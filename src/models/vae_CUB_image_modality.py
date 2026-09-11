# CUB Image-Captions Unimodal VAE Image model specification
# Deterministic behavior:
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

import torch
import torch.distributions as dist
from .base_vae import VAE
from .encoder_decoder_blocks.resnet_cub_image import EncoderImg, DecoderImg
from .encoder_decoder_blocks.siglip_cub_image import SigLIPEncoderImg
from .encoder_decoder_blocks.vitmae_cub_image import ViTMAEDecoderImg


class CUB_Image(VAE):
    """ Unimodal VAE subclass for Image modality CUB Image-Captions experiment """

    def __init__(self, params):
        image_encoder_arch = getattr(params, "image_encoder_arch", "cnn")

        if image_encoder_arch == "siglip":
            enc = SigLIPEncoderImg(
                params.latent_dim_w,
                params.latent_dim_z,
                dist=params.priorposterior,
                model_name=getattr(
                    params,
                    "siglip_model_name",
                    "google/siglip-base-patch16-256",
                ),
            )
        elif image_encoder_arch == "cnn":
            enc = EncoderImg(
                params.latent_dim_w,
                params.latent_dim_z,
                dist=params.priorposterior,
                img_size=params.img_size,
                in_channels=params.img_channels,
            )
        else:
            raise ValueError(
                f"Unknown image_encoder_arch={image_encoder_arch!r}; "
                "expected 'cnn' or 'siglip'."
            )

        image_decoder_arch = getattr(
            params,
            "image_decoder_arch",
            "cnn",
        )

        if image_decoder_arch == "cnn":

            # Existing decoder: unchanged
            dec = DecoderImg(
                params.latent_dim_u,
                img_size=params.img_size,
                out_channels=params.img_channels,
            )

        elif image_decoder_arch == "vitmae":

            dec = ViTMAEDecoderImg(
                ndim_w=params.latent_dim_w,
                ndim_z=params.latent_dim_z,
                model_name=getattr(
                    params,
                    "vitmae_model_name",
                    "facebook/vit-mae-base",
                ),
                cond_tokens_per_latent=getattr(
                    params,
                    "vitmae_cond_tokens_per_latent",
                    4,
                ),
                adapter_heads=getattr(
                    params,
                    "vitmae_adapter_heads",
                    12,
                ),
                adapter_mlp_ratio=getattr(
                    params,
                    "vitmae_adapter_mlp_ratio",
                    4.0,
                ),
            )

        else:

            raise ValueError(
                f"Unknown image_decoder_arch={image_decoder_arch!r}; "
                "expected 'cnn' or 'vitmae'."
            )

        # Intentionally unchanged: image reconstruction remains in the
        # existing 4x32x32 SD-VAE latent space.


        super(CUB_Image, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,          # prior
            likelihood_dist=dist.Laplace,                                                           # likelihood
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,           # posterior
            # Encoder model
            enc=enc,
            dec=dec,
            params=params                                                                           # Params (args passed to main)
        )
        self.modelName = 'cubI'

        if image_decoder_arch == "vitmae":

            self.dataSize = torch.Size([
                dec.num_channels,
                dec.image_size,
                dec.image_size,
            ])

            # Old image likelihood:
            #     4 * 32 * 32 = 4096 elements
            #
            # New RGB likelihood:
            #     3 * 224 * 224 = 150528 elements
            #
            # IDMVAE sums log_prob over all image elements,
            # therefore compensate for the dimensionality increase.
            self.llik_scaling = ((4 * 32 * 32)/(dec.num_channels * dec.image_size * dec.image_size))

        else:

            self.dataSize = torch.Size([params.img_channels,params.img_size,params.img_size,])

            self.llik_scaling = 1.



        self.params = params
        self.num_workers = params.num_workers if hasattr(params, 'num_workers') else 32


