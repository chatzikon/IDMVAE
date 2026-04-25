# CUB Image-Captions Unimodal VAE Image model specification
# Deterministic behavior:
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

import torch
import torch.distributions as dist
from .base_vae import VAE
from .encoder_decoder_blocks.resnet_cub_image import EncoderImg, DecoderImg

class CUB_Image(VAE):
    """ Unimodal VAE subclass for Image modality CUB Image-Captions experiment """

    def __init__(self, params):
        super(CUB_Image, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,          # prior
            likelihood_dist=dist.Laplace,                                                           # likelihood
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,           # posterior
            # Encoder model
            enc=EncoderImg(
                    params.latent_dim_w, params.latent_dim_z,
                    dist=params.priorposterior, img_size=params.img_size,
                    in_channels=params.img_channels),
            # Decoder model
            dec=DecoderImg(params.latent_dim_u, img_size=params.img_size,
                                out_channels=params.img_channels),
            params=params                                                                           # Params (args passed to main)
        )
        self.modelName = 'cubI'
        self.dataSize = torch.Size([params.img_channels, params.img_size, params.img_size])
        self.llik_scaling = 1.
        self.params = params
        self.num_workers = params.num_workers if hasattr(params, 'num_workers') else 32
