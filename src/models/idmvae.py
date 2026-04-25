# Base IDMVAE class definition
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .diffusion_prior import LatentDiffusionPrior

from diffusers.models import AutoencoderKL # pip install diffusers transformers accelerate


class IDMVAE(nn.Module):
    """
    IDMVAE class definition.
    """
    def __init__(self, params, *vaes): # *: # Captures all remaining positional arguments
        super(IDMVAE, self).__init__()
        self.num_views = len(vaes)
        self.modelName = None  # Filled-in in subclass
        self.params = params # Model parameters (i.e. args passed to main script)

        self.diffusion_loss_weight = params.diffusion_loss_weight
        # Some scripts (e.g., older PolyMNIST configs) might not define this flag; default to False.
        self.use_pretrain_feats = getattr(params, "use_pretrain_feats", False)
        self.vaes = nn.ModuleList([vae(params) for vae in vaes]) # List of unimodal VAEs (one for each modality)

        device = torch.device("cuda" if torch.cuda.is_available() and not params.no_cuda else "cpu")

        vae_variant = getattr(params, "vae", "mse")
        self.pretrained_vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_variant}").to(device) # ema or mse (default)

        # Priors are currently fixed here; make configurable via params if needed.
        self.pz_params = self._init_prior_params(params.latent_dim_z,
                                                 device,
                                                 learnable_logvar=False) # Original: learnable_logvar=False
        self.pws_params = nn.ModuleList([
            self._init_prior_params(params.latent_dim_w,
                                    device,
                                    learnable_logvar=False)
            for _ in range(self.num_views)
        ])
        self.pws_params_aux = nn.ModuleList([
            self._init_prior_params(params.latent_dim_w,
                                    device,
                                    learnable_logvar=True)
            for _ in range(self.num_views)
        ])
        if self.diffusion_loss_weight > 0.0:
            self.pz_diffusion = LatentDiffusionPrior(params.latent_dim_z,
                                                     stop_grad_on_input=params.diffusion_stop_grad_on_input,
                                                     timesteps=1000, sampling_timesteps=100, device=device)
            self.pws_diffusion = nn.ModuleList([
                LatentDiffusionPrior(params.latent_dim_w,
                                     stop_grad_on_input=params.diffusion_stop_grad_on_input,
                                     timesteps=1000, sampling_timesteps=100, device=device)
                for _ in range(self.num_views)
            ])

    def _init_prior_params(self,
                          latent_dim: int,
                          device: torch.device = torch.device("cpu"),
                          learnable_logvar: bool = False
                         ) -> nn.ParameterList:
        """
        Create (and register) a pair of parameters [mu, logvar]:
          - mu:       fixed zero mean
          - logvar:   zero-initialized, learnable if learnable_logvar=True
        """
        return nn.ParameterList([
            nn.Parameter(torch.zeros(1, latent_dim, device=device), requires_grad=False),
            nn.Parameter(torch.zeros(1, latent_dim, device=device), requires_grad=learnable_logvar)
        ])

    def _compute_prior_from_params(self, mu, logvar, type):
        if  type == "Normal":
            scale = F.softplus(logvar) + self.eps
            return dist.Normal(mu, scale)
        elif type == "Laplace":
            scale = F.softmax(logvar, dim=-1) * logvar.size(-1) + self.eps
            return dist.Laplace(mu, scale)
        else:
            raise ValueError(f"Unknown prior type {self.params.priorposterior!r}")

    def get_simple_prior_z(self):
        """
        Build a fresh Normal or Laplace distribution from (mu, logvar),
        using self.params.priorposterior to decide which.
        """
        mu, logvar = self.pz_params
        return self._compute_prior_from_params(mu, logvar, self.params.priorposterior)

    def get_simple_prior_w(self, view, aux=False):
        if aux:
            mu, logvar = self.pws_params_aux[view]
        else:
            mu, logvar = self.pws_params[view]
        return self._compute_prior_from_params(mu, logvar, self.params.priorposterior)

    @staticmethod
    def getDataSets(batch_size, shuffle=True, device="cuda"):
        # Handle getting individual datasets appropriately in sub-class
        raise NotImplementedError

    @property
    def encoders(self):
        return [vae.enc for vae in self.vaes]

    @property
    def decoders(self):
        return [vae.dec for vae in self.vaes]

    def forward(self, x, K=1):
        """
        Forward function.
        Input:
            - x: list of data samples for each modality
            - K: number of samples for reparameterization in latent space

        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of self- and cross- reconstructions. px_zs[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note there latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)

                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
                    pw = self.get_simple_prior_w(view=d, aux=True) # Original MMVAE+ set up, learnable prior only used here.
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)

                    # Fixed for cuda (sorry)
                    if not self.params.no_cuda and torch.cuda.is_available():
                        latents_w.cuda()
                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=-1)
                    # Get cross-reconstruction likelihood
                    px_us[e][d] = vae.px_u(*vae.dec(us_combined))
        return qu_xs, px_us, uss

    def _decode_latents_to_pixels(self, latents):
        """
        Decode SD-VAE latents and rescale results from [-1, 1] back to [0, 1].
        """
        assert self.use_pretrain_feats
        device = next(self.pretrained_vae.parameters()).device
        decoded = self.pretrained_vae.decode((latents.to(device)) / 0.18215).sample
        return decoded.add(1).div(2).clamp(0, 1)

    def self_and_cross_modal_generation_forward(self, data, K=1):
        """
        Test-time self- and cross-model generation forward function.
        Args:
            data: Input

        Returns:
            Unimodal encoding distribution, Matrix of self- and cross-modal reconstruction distrubutions, Latent embeddings

        """
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(data[m], K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            _, latents_z = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
            for d, vae in enumerate(self.vaes):
                # Note the different from forward():
                # 1. Here we use diffusion prior when possible.
                # 2. Here we use aux=False, i.e., non-learnable prior instead of learnable auxiliary prior.
                #    This is consistent with the original MMVAE+ implementation.
                if self.diffusion_loss_weight > 0.0:
                    pw = self.pws_diffusion[d]
                else:
                    pw = self.get_simple_prior_w(view=d, aux=False)
                latents_w_new = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                us_new = torch.cat((latents_w_new, latents_z), dim=-1)
                if e != d:  # fill-in off-diagonal
                    px_us[e][d] = vae.px_u(*vae.dec(us_new))
        return qu_xs, px_us, uss
