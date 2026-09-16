# CUB Image-Captions unimodal VAE (text / captions) specification
# Deterministic behavior:
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

import numpy as np
import torch
import torch.distributions as dist

from .base_vae import VAE
from .encoder_decoder_blocks.cnn_ucf_text import Enc, Dec
from .encoder_decoder_blocks.bart_ucf_text import BartLatentDecoder
from .encoder_decoder_blocks.bert_ucf_text import BertTextEncoder, BertLatentDecoder


# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 2505


class UCF_Sentence(VAE):

    def __init__(
        self,
        vocab_size,
        params,
    ):

        self.text_encoder_arch = getattr(params,"text_encoder_arch", "cnn")
        self.text_decoder_arch = getattr(params,"text_decoder_arch", "cnn")

        # ==================================================
        # SELECT TEXT ENCODER
        # ==================================================

        if self.text_encoder_arch == "cnn":

            enc = Enc(
                params.latent_dim_w,
                params.latent_dim_z,
                dist=params.priorposterior,
                vocab_size=vocab_size,
            )

        elif self.text_encoder_arch == "bert":

            enc = BertTextEncoder(
                latent_dim_w=params.latent_dim_w,
                latent_dim_z=params.latent_dim_z,
                dist_name=params.priorposterior,
                model_name=params.bert_model_name,
            )

        else:

            raise ValueError(
                "Unknown text encoder architecture: "
                f"{self.text_encoder_arch}"
            )

        # ==================================================
        # SELECT TEXT DECODER
        # ==================================================

        if self.text_decoder_arch == "cnn":

            dec = Dec(
                params.latent_dim_w,
                params.latent_dim_z,
                vocab_size=vocab_size,
            )

        elif self.text_decoder_arch == "bart":

            dec = BartLatentDecoder(
                latent_dim_w=params.latent_dim_w,
                latent_dim_z=params.latent_dim_z,
                model_name=params.bart_model_name,
                memory_tokens_per_latent=(
                    params.bart_memory_tokens_per_latent
                ),
                token_dropout=params.bart_token_dropout,
            )

        elif self.text_decoder_arch == "bert":

            dec = BertLatentDecoder(
                latent_dim_w=params.latent_dim_w,
                latent_dim_z=params.latent_dim_z,
                model_name=params.bert_model_name,
                memory_tokens_per_latent=(
                    params.bert_memory_tokens_per_latent
                ),
                max_length=params.bert_max_length,
            )

        else:

            raise ValueError(
                "Unknown text decoder architecture: "
                f"{self.text_decoder_arch}"
            )

        super().__init__(
            prior_dist=(
                dist.Normal
                if params.priorposterior == "Normal"
                else dist.Laplace
            ),

            likelihood_dist=dist.OneHotCategorical,

            post_dist=(
                dist.Normal
                if params.priorposterior == "Normal"
                else dist.Laplace
            ),

            enc=enc,
            dec=dec,
            params=params,
        )

        self.modelName = "cubC"
        self.llik_scaling = 1.0

        self.fn_2i = (
            lambda t:
            t.cpu().numpy().astype(int)
        )

        self.fn_trun = (
            lambda s:
            s[:np.where(s == 2)[0][0] + 1]
            if 2 in s
            else s
        )

        # Your actual current CNN encoder length.
        self.maxSentLen = 32

        # Do NOT use the old hard-coded 1590.
        self.vocabSize = vocab_size

        self.params = params

    def decode_likelihood(
            self,
            u,
            reconstruction_target=None,
    ):

        if self.text_decoder_arch == "bart":

            if reconstruction_target is None:
                raise ValueError(
                    "BART decoder requires BART-tokenized "
                    "reconstruction targets during training."
                )

            return self.dec(
                u,
                reconstruction_target,
            )

        if self.text_decoder_arch == "bert":
            # IMPORTANT:
            # reconstruction_target is deliberately ignored.
            #
            # BERT receives ONLY the IDMVAE latent.
            return self.dec(u)

        # CNN: exact legacy behaviour.
        return super().decode_likelihood(
            u,
            reconstruction_target,
        )
