# CUB Image-Captions unimodal VAE (text / captions) specification
# Deterministic behavior:
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

import numpy as np
import torch
import torch.distributions as dist

from .base_vae import VAE
from .encoder_decoder_blocks.cnn_cub_text import Enc, Dec


# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


class CUB_Sentence(VAE):
    """ Unimodal VAE subclass for Text modality CUB experiment """

    def __init__(self, vocab_size, params):
        super(CUB_Sentence, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,      # prior (continuous)
            likelihood_dist=dist.OneHotCategorical,                                             # likelihood (discrete)
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,       # posterior
            enc=Enc(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior, vocab_size=vocab_size),      # Encoder model
            dec=Dec(params.latent_dim_w, params.latent_dim_z, vocab_size=vocab_size),                                  # Decoder model
            params=params)                                                                      # Params (args passed to main)

        self.modelName = 'cubC'
        self.llik_scaling = 1.

        self.fn_2i = lambda t: t.cpu().numpy().astype(int)
        self.fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s

        self.maxSentLen = maxSentLen
        self.vocabSize = vocabSize

        self.params = params
