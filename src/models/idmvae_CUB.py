# CUB Image-Captions IDMVAE model specification
# Deterministic behavior (optional):
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
from numpy import prod
from .idmvae import IDMVAE
from .vae_CUB_image_modality import CUB_Image
from .vae_CUB_captions_modality import CUB_Sentence
from utils import ContrastiveLoss

from eval_functions import (
    setup_pretrained_denoiser,
    has_pretrained_denoiser,
    run_pretrained_denoiser,
)

# Constants
maxSentLen = 32
minOccur = 3
eps = 1e-6


class CUB_Image_Captions(IDMVAE):
    """
    IDMVAE subclass for CUB Image-Captions Experiment
    """
    def __init__(self, vocab_size, params):
        super(CUB_Image_Captions, self).__init__(vocab_size, params, CUB_Image, CUB_Sentence)

        self.vaes[0].llik_scaling = self.vaes[1].maxSentLen / prod(self.vaes[0].dataSize)
        self.vaes[1].llik_scaling = params.llik_scaling_sent
        self.modelName = 'IDMVAE_CUB'

        self.mi_shared = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.mi_private = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.contrast_mi = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]

        self.params = params
        self.eps = eps
        self.img_size_original = params.img_size_original
        self.text2img_ratio = params.text2img_ratio
        self.fontsize = params.fontsize
        denoiser_device = next(self.pretrained_vae.parameters()).device
        (
            self.denoiser_model,
            self.denoiser_diffusion,
            self.denoiser_device,
            self.denoiser_condition_label,
        ) = setup_pretrained_denoiser(self.params, denoiser_device)

        self.enable_denoiser_outputs = self._has_denoiser()
        self.last_denoised_prior_grids = None
        self.last_denoised_prior_entries = None
        self.last_denoised_posterior_grids = None
        self.last_denoised_posterior_entries = None
        self.last_denoised_posterior_nonshuf_grids = None
        self.last_denoised_posterior_nonshuf_entries = None
        self.last_denoised_posterior_extended_grids = None

    def _has_denoiser(self):
        return has_pretrained_denoiser(self.denoiser_model, self.denoiser_diffusion)

    def _run_denoiser(self, noisy_latents):
        return run_pretrained_denoiser(
            self.denoiser_model,
            self.denoiser_diffusion,
            self.denoiser_device,
            self.denoiser_condition_label,
            noisy_latents,
        )

    def _denoise_recon_matrix(self, recons):
        if not self._has_denoiser():
            return None
        denoised = [[None for _ in row] for row in recons]
        for r, row in enumerate(recons):
            for c, tensor in enumerate(row):
                if tensor is None or c != 0:
                    continue
                latents = tensor.squeeze(0)
                refined = self._run_denoiser(latents)
                if refined is not None:
                    denoised[r][c] = refined.unsqueeze(0)
        return denoised
