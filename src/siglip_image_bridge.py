import torch


VAE_LATENT_SCALE = 0.18215


def uses_siglip_image_encoder(model):

    return (
        getattr(
            model.params,
            "image_encoder_arch",
            "cnn"
        )
        == "siglip"
    )


def uses_vitmae_image_decoder(model):

    return (
        getattr(
            model.params,
            "image_decoder_arch",
            "cnn"
        )
        == "vitmae"
    )


def get_reconstruction_targets(model, x):
    """
    Build reconstruction targets without changing the actual
    modality inputs passed to the encoders.

    Cases
    -----
    1. SigLIP + ViT-MAE:
       encoder input  = RGB
       decoder target = RGB resized to ViT-MAE resolution

    2. SigLIP + old CNN decoder:
       encoder input  = RGB
       decoder target = SD-VAE latent

    3. Legacy CNN setup:
       reconstruction target = original modality input
    """

    # =========================================================
    # NEW RGB ViT-MAE branch
    # =========================================================
    if uses_vitmae_image_decoder(model):

        # x[0] is RGB.
        #
        # ViT-MAE decoder reconstructs RGB directly.
        # prepare_target() handles the required resolution,
        # e.g. 256x256 -> 224x224 for facebook/vit-mae-base.
        image_target = (
            model.vaes[0]
            .dec
            .prepare_target(x[0])
        )

        targets = list(x)

        # Replace ONLY the image reconstruction target.
        targets[0] = image_target

        return targets

    # =========================================================
    # Legacy non-SigLIP branch
    # =========================================================
    if not uses_siglip_image_encoder(model):
        return x

    # =========================================================
    # Existing SigLIP + CNN decoder branch
    #
    # SigLIP encoder sees RGB, but the old image decoder
    # reconstructs an SD-VAE latent [4,32,32].
    # =========================================================

    rgb = x[0]

    vae = model.pretrained_vae

    if vae is None:
        raise RuntimeError(
            "SigLIP + non-ViTMAE image decoder requires "
            "model.pretrained_vae, but it is None."
        )

    vae_device = next(
        vae.parameters()
    ).device

    rgb = (
        rgb.to(vae_device)
        .float()
        .mul(2)
        .sub(1)
    )

    # Target only -> gradients unnecessary.
    with torch.no_grad():

        image_target = (
            vae
            .encode(rgb)
            .latent_dist
            .sample()
            * VAE_LATENT_SCALE
        )

    targets = list(x)

    # Replace ONLY image reconstruction target.
    targets[0] = image_target

    return targets


def prepare_augmented_data_for_encoder(
    model,
    aug_data,
    view_index,
):
    """
    Convert generated image data into the representation expected
    by the image encoder during generative augmentation.

    Text modality:
        unchanged

    CNN image encoder:
        unchanged

    SigLIP + ViT-MAE:
        decoder already outputs RGB -> feed RGB to SigLIP

    SigLIP + old CNN image decoder:
        decoder outputs SD latent -> SD-VAE decode -> RGB -> SigLIP
    """

    # Text modality or original CNN image encoder:
    # nothing changes.
    if (
        view_index != 0
        or not uses_siglip_image_encoder(model)
    ):
        return aug_data

    # =========================================================
    # NEW: SigLIP + ViT-MAE
    #
    # aug_data is already RGB from the ViT-MAE decoder.
    # No SD-VAE conversion is necessary.
    # =========================================================
    if uses_vitmae_image_decoder(model):

        # SigLIPEncoderImg expects RGB values in [0,1].
        #
        # Do NOT detach / use torch.no_grad():
        # GenAug gradients must propagate:
        #
        # SigLIP
        #   ↓
        # generated RGB
        #   ↓
        # ViT-MAE decoder
        #   ↓
        # IDMVAE latent
        return aug_data.clamp(0.0, 1.0)

    # =========================================================
    # Existing SigLIP + CNN decoder
    #
    # aug_data is [B,4,32,32] SD-VAE latent.
    # Convert it back to RGB before feeding SigLIP.
    # =========================================================

    vae = model.pretrained_vae

    if vae is None:
        raise RuntimeError(
            "SigLIP + non-ViTMAE image decoder requires "
            "model.pretrained_vae for GenAug, but it is None."
        )

    vae_device = next(
        vae.parameters()
    ).device

    latent = aug_data.to(vae_device)

    # IMPORTANT:
    # no torch.no_grad() here.
    #
    # SD-VAE parameters are frozen, but gradients must pass:
    #
    # SigLIP
    #   -> SD-VAE decoder
    #   -> IDMVAE image decoder
    rgb = vae.decode(
        latent / VAE_LATENT_SCALE
    ).sample

    rgb = (
        rgb
        .add(1)
        .div(2)
        .clamp(0, 1)
    )

    return rgb