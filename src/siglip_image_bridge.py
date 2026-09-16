import torch


VAE_LATENT_SCALE = 0.18215





def get_reconstruction_targets(model,x):
    """
    Create reconstruction targets for the two actual
    IDMVAE modalities:

        targets[0] -> image reconstruction target
        targets[1] -> text reconstruction target

    Important:
    These targets do NOT replace the inputs seen by the
    encoders.

    For example, with SigLIP + ViT-MAE:

        encoder input:
            RGB image

        decoder target:
            RGB image resized for ViT-MAE
    """

    image_encoder_arch = getattr(
        model.params,
        "image_encoder_arch",
        "cnn",
    )

    image_decoder_arch = getattr(
        model.params,
        "image_decoder_arch",
        "cnn",
    )

    text_decoder_arch = getattr(
        model.params,
        "text_decoder_arch",
        "cnn",
    )

    # There are two actual modalities:
    #
    #   0 -> image
    #   1 -> text
    #
    # x may later contain x[2] as an auxiliary BART target,
    # but that is NOT a third modality.
    targets = [x[0], x[1]]


    # =====================================================
    # IMAGE RECONSTRUCTION TARGET
    # =====================================================

    # -----------------------------------------------------
    # Case 1:
    # SigLIP encoder + ViT-MAE decoder
    #
    # RGB -> SigLIP -> latent -> ViT-MAE -> RGB
    #
    # Therefore reconstruction target is RGB.
    # -----------------------------------------------------
    if image_encoder_arch == "siglip" and image_decoder_arch == "vitmae":

        targets[0] = (model.vaes[0].dec.prepare_target(x[0]))


    # -----------------------------------------------------
    # Case 2:
    # SigLIP encoder + old CNN decoder
    #
    # RGB -> SigLIP -> latent -> CNN -> SD-VAE latent
    #
    # Therefore the reconstruction target must also be
    # an SD-VAE latent.
    # -----------------------------------------------------
    elif (
        image_encoder_arch == "siglip"
        and image_decoder_arch == "cnn"
    ):

        vae = model.pretrained_vae

        if vae is None:
            raise RuntimeError(
                "SigLIP + CNN image decoder requires "
                "model.pretrained_vae, but it is None."
            )

        vae_device = next(vae.parameters()).device

        rgb = x[0].to(vae_device).float().mul(2).sub(1)

        # Reconstruction target only:
        # gradients are unnecessary.
        with torch.no_grad():

            image_target = (vae.encode(rgb).latent_dist.sample()* VAE_LATENT_SCALE)

        targets[0] = image_target


    # -----------------------------------------------------
    # Case 3:
    # CNN encoder + CNN decoder
    #
    # Existing legacy behaviour.
    #
    # targets[0] remains x[0].
    # -----------------------------------------------------
    elif image_encoder_arch == "cnn" and image_decoder_arch == "cnn":
        pass


    # -----------------------------------------------------
    # Any other combination has not been implemented.
    # -----------------------------------------------------
    else:

        raise ValueError(
            "Unsupported image architecture combination: "
            f"encoder={image_encoder_arch}, "
            f"decoder={image_decoder_arch}"
        )


    # =====================================================
    # TEXT RECONSTRUCTION TARGET
    # =====================================================

    # -----------------------------------------------------
    # Existing CNN text decoder:
    #
    # target is your current [32, vocab_size] one-hot
    # caption representation.
    # -----------------------------------------------------
    if text_decoder_arch == "cnn":

        targets[1] = x[1]


    # -----------------------------------------------------
    # New BART decoder:
    #
    # x[1] remains the custom one-hot representation used
    # by the EXISTING text encoder.
    #
    # x[2] is the auxiliary BART-tokenized target used only
    # by the BART decoder.
    # -----------------------------------------------------
    elif text_decoder_arch == "bart":

        if len(x) < 3:
            raise RuntimeError(
                "BART text decoder requires BART-tokenized "
                "target IDs in x[2]."
            )

        targets[1] = x[2]


    elif text_decoder_arch == "bert":

        # x[1] is both:
        #
        #   - the BERT text-encoder input
        #   - the reconstruction TARGET
        #
        # It is NOT passed into the BERT decoder itself.
        targets[1] = x[1]

    else:

        raise ValueError(
            "Unsupported text decoder architecture: "
            f"{text_decoder_arch}"
        )


    return targets

def prepare_augmented_data_for_encoder(
    model,
    aug_data,
    view_index,
):
    """
    Convert generated data into the representation expected
    by the corresponding encoder during generative augmentation.

    Cases
    -----
    Text modality:
        unchanged

    CNN image encoder + CNN image decoder:
        unchanged

    SigLIP image encoder + ViT-MAE image decoder:
        ViT-MAE already generates RGB -> feed RGB directly to SigLIP

    SigLIP image encoder + CNN image decoder:
        CNN generates SD-VAE latent -> decode latent to RGB
        -> feed RGB to SigLIP
    """

    image_encoder_arch = getattr(
        model.params,
        "image_encoder_arch",
        "cnn",
    )

    image_decoder_arch = getattr(
        model.params,
        "image_decoder_arch",
        "cnn",
    )

    # =========================================================
    # TEXT MODALITY
    # =========================================================
    #
    # view_index:
    #   0 -> image
    #   1 -> text
    #
    # Text augmentation is handled elsewhere and does not
    # require image-space conversion.
    # =========================================================

    if view_index != 0:
        return aug_data


    # =========================================================
    # IMAGE: CNN encoder + CNN decoder
    # =========================================================
    #
    # Generated output is already in the representation expected
    # by the CNN image encoder.
    # =========================================================

    if (
        image_encoder_arch == "cnn"
        and image_decoder_arch == "cnn"
    ):
        return aug_data


    # =========================================================
    # IMAGE: SigLIP encoder + ViT-MAE decoder
    # =========================================================
    #
    # ViT-MAE decoder already generates RGB:
    #
    #       latent
    #         ↓
    #      ViT-MAE
    #         ↓
    #       RGB
    #         ↓
    #      SigLIP
    #
    # Therefore no SD-VAE conversion is needed.
    # =========================================================

    if (
        image_encoder_arch == "siglip"
        and image_decoder_arch == "vitmae"
    ):

        # SigLIPEncoderImg expects RGB in [0,1].
        #
        # IMPORTANT:
        # Do NOT detach and do NOT use torch.no_grad().
        #
        # GenAug gradients must be able to propagate:
        #
        # GenAug loss
        #     ↓
        # SigLIP encoder
        #     ↓
        # generated RGB
        #     ↓
        # ViT-MAE decoder
        #     ↓
        # IDMVAE latent
        return aug_data.clamp(
            0.0,
            1.0,
        )


    # =========================================================
    # IMAGE: SigLIP encoder + CNN decoder
    # =========================================================
    #
    # The old CNN decoder generates an SD-VAE latent:
    #
    #       latent
    #         ↓
    #    CNN decoder
    #         ↓
    #   [4,32,32] SD latent
    #         ↓
    #   SD-VAE decoder
    #         ↓
    #        RGB
    #         ↓
    #      SigLIP
    #
    # Therefore we must decode the generated SD latent back
    # to RGB before passing it to SigLIP.
    # =========================================================

    if (
        image_encoder_arch == "siglip"
        and image_decoder_arch == "cnn"
    ):

        vae = model.pretrained_vae

        if vae is None:
            raise RuntimeError(
                "SigLIP + CNN image decoder requires "
                "model.pretrained_vae for generative "
                "augmentation, but it is None."
            )

        vae_device = next(
            vae.parameters()
        ).device

        latent = aug_data.to(
            vae_device
        )

        # IMPORTANT:
        # no torch.no_grad() here.
        #
        # The SD-VAE parameters themselves can be frozen,
        # but autograd must pass THROUGH its decoder so that
        # the GenAug loss reaches the IDMVAE decoder.
        rgb = vae.decode(
            latent / VAE_LATENT_SCALE
        ).sample

        rgb = (
            rgb
            .add(1)
            .div(2)
            .clamp(0.0, 1.0)
        )

        return rgb


    # =========================================================
    # Unsupported combination
    # =========================================================

    raise ValueError(
        "Unsupported image architecture combination "
        "during generative augmentation: "
        f"encoder={image_encoder_arch}, "
        f"decoder={image_decoder_arch}"
    )