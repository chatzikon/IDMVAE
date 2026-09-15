# Objectives
import torch
from utils import log_mean_exp
from siglip_image_bridge import (
    get_reconstruction_targets,
    prepare_augmented_data_for_encoder,
)

"""
Abbreviations:
# Names of different dimensions of shape tensors:
- B: Batch size
- D: Data dimension
- K: Number of samples for resampling in the latent space
- M: Number of modalities
- N: Number of samples
- Z: Shared latent dimension
- W: Private latent dimension (modality-specific)
- C: Number of channels
- H: Height of the image
- V: Width of the image
"""


def compute_elbo_loss(
    model,
    x,
    K=1,
    test=False,
):
    """
    Core ELBO computation for a single minibatch.

    Supports both:

        text_decoder_arch == "cnn"
        text_decoder_arch == "bart"

    Reconstruction targets are separated from encoder inputs.

    Examples
    --------
    Image:
        SigLIP + CNN:
            encoder input  = RGB
            decoder target = SD-VAE latent

        SigLIP + ViT-MAE:
            encoder input  = RGB
            decoder target = RGB prepared for ViT-MAE

    Text:
        CNN decoder:
            encoder input  = custom one-hot caption
            decoder target = same custom one-hot caption

        BART decoder:
            encoder input  = custom one-hot caption
            decoder target = BART token IDs
    """

    # =========================================================
    # 1. PREPARE RECONSTRUCTION TARGETS
    # =========================================================
    #
    # IMPORTANT:
    #
    # This must happen BEFORE model(...).
    #
    # The old CNN decoder only needs the latent u to construct
    # p(x|u).
    #
    # BART additionally needs the ground-truth BART token IDs
    # during training for teacher forcing.
    # =========================================================

    text_decoder_arch = getattr(model.params,"text_decoder_arch", "cnn")


    if not test:

        # =====================================================
        # TRAINING
        # =====================================================
        #
        # BART is allowed to use teacher forcing here.
        #
        # The reconstruction targets must therefore be built
        # BEFORE model.forward().
        # =====================================================

        reconstruction_targets = (
            get_reconstruction_targets(
                model,
                x,
            )
        )

        qu_xs, px_us, uss = model(
            x,
            K,
            reconstruction_targets=reconstruction_targets,
        )


    else:

        # =====================================================
        # TEST LOSS
        # =====================================================
        #
        # IMPORTANT:
        #
        # test=True means held-out ELBO / likelihood testing.
        #
        # This is NOT the same as autoregressive caption
        # generation evaluation.
        #
        # BART:
        #     Teacher forcing is allowed here so that we can
        #     compute p(x_text | u) / reconstruction likelihood.
        #
        # CNN:
        #     Keep the existing legacy test-time behaviour.
        # =====================================================

        if text_decoder_arch == "bart":

            # -------------------------------------------------
            # BART TEST ELBO
            # -------------------------------------------------
            #
            # Build targets BEFORE forward because the BART
            # decoder requires target token IDs for its
            # teacher-forced likelihood.
            #
            # x[1]:
            #     custom one-hot caption used by CNN encoder
            #
            # x[2]:
            #     BART-tokenized caption used as decoder target
            # -------------------------------------------------

            reconstruction_targets = (
                get_reconstruction_targets(
                    model,
                    x,
                )
            )

            qu_xs, px_us, uss = model(
                x,
                K,
                reconstruction_targets=reconstruction_targets,
            )

        else:

            # -------------------------------------------------
            # CNN TEST ELBO
            # -------------------------------------------------
            #
            # Preserve the existing behaviour.
            #
            # Generate first without reconstruction targets,
            # then use GT only afterward when evaluating
            # log_prob().
            # -------------------------------------------------

            qu_xs, px_us, uss = (
                model.self_and_cross_modal_generation_forward(
                    x,
                    K,
                )
            )

            reconstruction_targets = (
                get_reconstruction_targets(
                    model,
                    x,
                )
            )


    # =========================================================
    # 3. SPLIT POSTERIORS INTO PRIVATE w AND SHARED z
    # =========================================================

    shared_latents = []
    shared_dists = []

    private_latents = []
    private_dists = []

    qz_xs = []
    qw_xs = []

    for r, qu_x in enumerate(qu_xs):

        # Posterior parameters:
        #
        # [B,W+Z]
        qu_x_r_mean, qu_x_r_lv = (
            model.vaes[r].qu_x_params
        )

        # -----------------------------------------------------
        # Split posterior parameters:
        #
        # private w
        # shared  z
        # -----------------------------------------------------

        qw_x_mean, qz_x_mean = torch.split(
            qu_x_r_mean,
            [
                model.params.latent_dim_w,
                model.params.latent_dim_z,
            ],
            dim=-1,
        )

        qw_x_lv, qz_x_lv = torch.split(
            qu_x_r_lv,
            [
                model.params.latent_dim_w,
                model.params.latent_dim_z,
            ],
            dim=-1,
        )

        qw_x = model.vaes[r].qu_x(
            qw_x_mean,
            qw_x_lv,
        )

        qz_x = model.vaes[r].qu_x(
            qz_x_mean,
            qz_x_lv,
        )

        qz_xs.append(qz_x)
        qw_xs.append(qw_x)


    shared_dists = qz_xs
    private_dists = qw_xs


    # =========================================================
    # 4. ELBO TERMS
    # =========================================================

    lws = []
    KL_divs = []
    llik_recons = []




    for r, qu_x in enumerate(qu_xs):

        # -----------------------------------------------------
        # Split sampled latent:
        #
        # [K,B,W+Z]
        #       ↓
        # w: [K,B,W]
        # z: [K,B,Z]
        # -----------------------------------------------------

        ws, zs = torch.split(
            uss[r],
            [
                model.params.latent_dim_w,
                model.params.latent_dim_z,
            ],
            dim=-1,
        )

        shared_latents.append(zs)
        private_latents.append(ws)


        # =====================================================
        # PRIOR LOG-PROBABILITIES
        # =====================================================

        lpz = (
            model
            .get_simple_prior_z()
            .log_prob(zs)
            .sum(-1)
        )

        lpw = (
            model
            .get_simple_prior_w(
                view=r,
                aux=False,
            )
            .log_prob(ws)
            .sum(-1)
        )


        # =====================================================
        # POSTERIOR LOG-PROBABILITIES
        # =====================================================

        lqz_x = log_mean_exp(
            torch.stack([
                qz_x
                .log_prob(zs)
                .sum(-1)

                for qz_x in qz_xs
            ])
        )

        lqw_x = (
            qw_xs[r]
            .log_prob(ws)
            .sum(-1)
        )


        # =====================================================
        # RECONSTRUCTION LOG-LIKELIHOODS
        # =====================================================
        #
        # px_us[r][d]:
        #
        #   r = source modality supplying shared z
        #   d = target modality / decoder
        #
        # Examples:
        #
        #   px_us[0][0] = image -> image
        #   px_us[0][1] = image -> text
        #   px_us[1][0] = text  -> image
        #   px_us[1][1] = text  -> text
        #
        # CNN text decoder:
        #
        #   log_prob:
        #       [K,B,32]
        #
        # BART text decoder:
        #
        #   log_prob:
        #       [K,B,L_bart]
        #
        # Image decoder:
        #
        #   log_prob:
        #       [K,B,C,H,W]
        #
        # We therefore avoid relying on
        # torch.distributions.batch_shape and simply preserve
        # the first two dimensions [K,B], flattening everything
        # after them.
        # =====================================================

        lpx_u = []



        for d, px_u in enumerate(px_us[r]):

            target = reconstruction_targets[d]

            log_prob = px_u.log_prob(
                target
            )

            # -------------------------------------------------
            # Expected first dimensions:
            #
            #   [K,B,...]
            #
            # This works for:
            #   image distributions
            #   old OneHotCategorical text
            #   new BartTextLikelihood
            # -------------------------------------------------

            if log_prob.ndim < 2:

                raise RuntimeError(
                    "Reconstruction log_prob must have "
                    "at least [K,B] dimensions, but got "
                    f"{log_prob.shape} for target view {d}."
                )

            log_prob = log_prob.reshape(
                log_prob.shape[0],
                log_prob.shape[1],
                -1,
            )

            # Use scaling belonging to the TARGET modality d.
            log_prob = (
                log_prob
                .mul(
                    model.vaes[d].llik_scaling
                )
                .sum(-1)
            )



            # [K,B]
            lpx_u.append(log_prob)


        # Sum likelihoods over target modalities.
        #
        # [M,K,B] -> [K,B]
        lpx_u = (
            torch
            .stack(lpx_u)
            .sum(0)
        )


        # =====================================================
        # LOG IMPORTANCE WEIGHT / ELBO
        # =====================================================

        KL_div = (
            lpz
            + lpw
            - lqz_x
            - lqw_x
        )

        lw = (
            lpx_u
            + model.params.beta * KL_div
        )

        lws.append(lw)
        llik_recons.append(lpx_u)
        KL_divs.append(KL_div)



    # =========================================================
    # 5. STACK MODALITIES
    # =========================================================

    # [M,K,B]
    lws = torch.stack(lws)

    llik_recons_stk = torch.stack(
        llik_recons
    )

    KL_divs_stk = torch.stack(
        KL_divs
    )


    # =========================================================
    # 6. FINAL LOSSES
    # =========================================================

    # log_mean_exp over K
    # mean over source modalities M
    # mean over minibatch B

    elbo_loss = (
        -log_mean_exp(
            lws,
            dim=1,
        )
        .mean(0)
        .mean()
    )

    llik_recon_loss = (
        -log_mean_exp(
            llik_recons_stk,
            dim=1,
        )
        .mean(0)
        .mean()
    )



    KL_div_loss = (
        -log_mean_exp(
            KL_divs_stk,
            dim=1,
        )
        .mean(0)
        .mean()
    )

    recon_KL_sum_loss = (
        llik_recon_loss
        + model.params.beta
        * KL_div_loss
    )


    # =========================================================
    # 7. RETURN
    # =========================================================

    return (
        elbo_loss,
        recon_KL_sum_loss,
        llik_recon_loss,
        KL_div_loss,
        shared_latents,
        shared_dists,
        private_latents,
        private_dists,
    )


def compute_cross_mi_loss(dists_shared, mi_estimators, use_mean_for_mi=True):
    """
    Compute Cross MI loss for datasets with multiple views.

    Args:
        dists_shared (list): List of shared latent distributions (one for each view).
        mi_estimators: Mutual information estimator function.

    Returns:
        cross_mi_loss (torch.Tensor): Combined mutual information bottleneck loss across all pairs of views.
    """
    num_views = len(dists_shared)  # Number of views(cross_mi): 5
    cross_mi_loss = 0.0
    for i in range(num_views):
        mi_estimator = mi_estimators[i]
        for j in range(num_views):  # Iterate over unique pairs
            if i == j:
                continue

            p_z1_given_v1, p_z2_given_v2 = dists_shared[i], dists_shared[j]

            # Sample from the posteriors using reparameterization
            # z1, z2 shape: (B, Z)
            if use_mean_for_mi:
                z1 = p_z1_given_v1.loc
                z2 = p_z2_given_v2.loc
            else:
                z1 = p_z1_given_v1.rsample()
                z2 = p_z2_given_v2.rsample()

            # Estimate mutual information
            mi_for_grad, _ = mi_estimator(z1, z2)
            mi_for_grad = mi_for_grad.mean()

            cross_mi_loss += -mi_for_grad

    cross_mi_loss = cross_mi_loss / (num_views - 1)
    return cross_mi_loss


def compute_shared_z_alignment_loss(shared_dists):
    """
    Align paired shared posterior distributions across modalities.

    For diagonal Gaussian posteriors, this is the dimension-normalized
    squared 2-Wasserstein distance:

        ||mu_i - mu_j||^2 + ||sigma_i - sigma_j||^2

    averaged over latent dimensions, batch, and modality pairs.

    shared_dists[i].loc   : [B, Z]
    shared_dists[i].scale : [B, Z]
    """
    num_views = len(shared_dists)

    if num_views < 2:
        return shared_dists[0].loc.new_tensor(0.0)

    loss = 0.0
    num_pairs = 0

    mu_txt = shared_dists[1].loc.detach()
    sigma_txt = shared_dists[1].scale.detach()

    mu_img = shared_dists[0].loc
    sigma_img = shared_dists[0].scale

    z_alignment_loss = (
            (mu_img - mu_txt).pow(2)
            + (sigma_img - sigma_txt).pow(2)
    ).mean()

    return z_alignment_loss

    # for i in range(num_views):
    #     for j in range(i + 1, num_views):
    #
    #
    #         mu_txt = shared_dists[1].loc.detach()
    #         sigma_txt = shared_dists[1].scale.detach()
    #
    #         mu_img = shared_dists[0].loc
    #         sigma_img = shared_dists[0].scale
    #
    #         z_alignment_loss = (
    #                 (mu_img - mu_txt).pow(2)
    #                 + (sigma_img - sigma_txt).pow(2)
    #         ).mean()
    #
    #         mu_i = shared_dists[i].loc
    #         mu_j = shared_dists[j].loc
    #
    #         sigma_i = shared_dists[i].scale
    #         sigma_j = shared_dists[j].scale
    #
    #         pair_loss = (
    #             (mu_i - mu_j).pow(2)
    #             + (sigma_i - sigma_j).pow(2)
    #         ).mean()
    #
    #         loss += pair_loss
    #         num_pairs += 1
    #
    # return loss / num_pairs

def compute_gen_aug_loss(
    shared_latents,
    shared_dists,
    private_latents,
    private_dists,
    decoders,
    encoders,
    mi_shared,
    mi_private,
    model,
    n_samples,
    sampling_scheme,
):
    """
    Compute the generative augmentation loss.

    Args:
        shared_latents (list): List of shared latent tensors (one per view).
        shared_dists (list): List of distributions for shared latents (one per view).
            type: list[i] -> mean, scale.shape: (B, Z), (B, Z)
        private_latents (list): List of private latent tensors (one per view).
        private_dists (list): List of distributions for private latents (one per view).
        decoders (list): List of decoders (one per view).
        encoders (list): List of encoders (one per view).
        mi_shared (list): List of mutual information estimators for shared latents.
        mi_private (list): List of mutual information estimators for private latents.
        n_samples: number of samples for augmentation

    Returns:
        torch.Tensor: Generative augmentation information loss.
    """
    gen_aug_loss = 0.0

    # Compute gen aug Loss for each view
    for i in range(model.num_views):

        # roll_private mutual information for shared gen aug
        gen_aug_shared, _ = compute_gen_aug_loss_oneview(
            shared_latents[i],
            shared_dists[i],
            private_latents[i],
            private_dists[i],
            decoders[i],
            encoders[i],
            mi_shared[i],
            model,
            n_samples,
            view_index=i,
            role="roll_private",
            use_mean_for_mi=True,
            sampling_scheme=sampling_scheme,
        )
        # roll_shared mutual information for private gen aug
        gen_aug_private, _ = compute_gen_aug_loss_oneview(
            shared_latents[i],
            shared_dists[i],
            private_latents[i],
            private_dists[i],
            decoders[i],
            encoders[i],
            mi_private[i],
            model,
            n_samples,
            view_index=i,
            role="roll_shared",
            use_mean_for_mi=True,
            sampling_scheme=sampling_scheme,
        )
        gen_aug_loss += 0.5 * (gen_aug_shared + gen_aug_private)

    return gen_aug_loss


def compute_gen_aug_loss_oneview(
    shared_latent,
    dist_shared,
    private_latent,
    dist_private,
    decoder,
    encoder,
    mi_estimator,
    model,
    n_samples,
    view_index,
    role,
    use_mean_for_mi=True,
    sampling_scheme="posterior",
):
    """
    Compute the generative augmentation loss between shared and private latents.

    Args:
    Note: posterior sampling is currently used; prior-based variants can be explored separately.
        shared_latent: posterior samples of shared latent variable.
        dist_shared: posterior distribution for the shared latent variable (e.g., q(z|x)).
            type: dist -> mean, scale.shape: (B, Z), (B, Z)
        private_latent: posterior samples of private latent variable.
        dist_private: posterior distribution for the private latent variable (e.g., q(w|x)).
        decoder: Decoder network to map latents to reconstructed data.
        encoder: Encoder network for extracting latents from reconstructed data.
        mi_estimator: Mutual information estimator.
        n_samples: Number of samples for latent space for generation.
        role: Role of the latent variables ('roll_private' or 'roll_shared').

    Returns:
        gen_aug_subloss (torch.Tensor): Generative augmentation loss.
        aug_data (torch.Tensor): Reconstructed data from the augmented input.
    """

    # Augmented input: concatenate sampled shared and private latents
    # aug_in shape: (K, B, W+Z), roll in the batch dimension to mix and match.
    if role == "roll_private":
        if sampling_scheme == "posterior":
            aug_in = torch.cat(
                [
                    torch.roll(
                        dist_private.rsample(torch.Size([n_samples])), shifts=1, dims=1
                    ),
                    dist_shared.rsample(torch.Size([n_samples])),
                ],
                axis=-1,
            )
        elif sampling_scheme == "diffusion_prior":
            private_latent_prior = model.pws_diffusion[view_index]
            aug_in = torch.cat(
                [
                    private_latent_prior.rsample(
                        torch.Size([n_samples, private_latent.size()[1]])
                    ).squeeze(2),
                    dist_shared.rsample(torch.Size([n_samples])),
                ],
                axis=-1,
            )
        else:
            private_latent_prior = model.get_simple_prior_w(view=view_index, aux=False)
            aug_in = torch.cat(
                [
                    private_latent_prior.rsample(
                        torch.Size([n_samples, private_latent.size()[1]])
                    ).squeeze(2),
                    dist_shared.rsample(torch.Size([n_samples])),
                ],
                axis=-1,
            )

    elif role == "roll_shared":
        if sampling_scheme == "posterior":
            aug_in = torch.cat(
                [
                    dist_private.rsample(torch.Size([n_samples])),
                    torch.roll(
                        dist_shared.rsample(torch.Size([n_samples])), shifts=1, dims=1
                    ),
                ],
                axis=-1,
            )
        elif sampling_scheme == "diffusion_prior":
            shared_latent_prior = model.pz_diffusion
            aug_in = torch.cat(
                [
                    dist_private.rsample(torch.Size([n_samples])),
                    shared_latent_prior.rsample(
                        torch.Size([n_samples, shared_latent.size()[1]])
                    ).squeeze(2),
                ],
                axis=-1,
            )
        else:
            shared_latent_prior = model.get_simple_prior_z()
            aug_in = torch.cat(
                [
                    dist_private.rsample(torch.Size([n_samples])),
                    shared_latent_prior.rsample(
                        torch.Size([n_samples, shared_latent.size()[1]])
                    ).squeeze(2),
                ],
                axis=-1,
            )
    else:
        raise ValueError(
            f"Invalid role: {role}. Expected 'roll_private' or 'roll_shared'."
        )
    # ---------------------------------------------------------
    # Decode augmented latent and re-encode it.
    #
    # BART text uses a special fully differentiable continuous
    # GenAug path. All other decoders keep the original path.
    # ---------------------------------------------------------

    is_bart_text = (
            view_index == 1
            and getattr(
        model.params,
        "text_decoder_arch",
        "cnn",
    ) == "bart"
    )

    if is_bart_text:

        # -----------------------------------------------------
        # BART text GenAug
        # -----------------------------------------------------
        #
        # aug_in:
        #     [K, B, W+Z]
        #
        # forward_gen_aug returns continuous text features:
        #     [K, B, 32, 128]
        #
        # No target caption.
        # No teacher forcing.
        # No discrete tokens.
        # No argmax.
        # -----------------------------------------------------

        aug_data_K = decoder.forward_gen_aug(
            aug_in
        )

        # Flatten K and B:
        #
        # [K, B, 32, 128]
        #       ->
        # [K*B, 32, 128]
        aug_data = aug_data_K.reshape(
            -1,
            aug_data_K.size(-2),
            aug_data_K.size(-1),
        )

        # BART already produced 128-D continuous embeddings,
        # so bypass the CNN text encoder's vocab -> embedding
        # projection layer.
        mu, sigma = encoder.forward_from_embeddings(
            aug_data
        )

    else:

        # -----------------------------------------------------
        # Original GenAug path
        # -----------------------------------------------------

        aug_data_raw = decoder(
            aug_in
        )

        # Handle different decoder outputs.
        if isinstance(aug_data_raw, tuple):

            # Image decoder, e.g. reconstructed image mean.
            aug_data_K = aug_data_raw[0]

        elif isinstance(aug_data_raw, list):

            # Legacy CNN text decoder.
            aug_data_K = aug_data_raw[0]

        else:

            raise ValueError(
                "aug_data_raw is wrong!"
            )

        # Flatten K and B:
        #
        # [K,B,...] -> [K*B,...]
        aug_data = aug_data_K.reshape(
            -1,
            *aug_data_K.shape[2:],
        )

        # Architecture-specific preparation, mainly needed
        # for the image branch.
        aug_data_for_encoder = (
            prepare_augmented_data_for_encoder(
                model,
                aug_data,
                view_index,
            )
        )

        # Standard encoder path.
        mu, sigma = encoder(
            aug_data_for_encoder
        )

    # Split the mu and logvar into shared and private parts
    private_mu, shared_mu = torch.split(
        mu, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1
    )
    private_sigma, shared_sigma = torch.split(
        sigma, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1
    )

    # Expand dist_shared and dist_private to match the shape of the new latents
    mu_shared_expanded = dist_shared.loc.repeat_interleave(n_samples, dim=0)  # (K*B, Z)
    sigma_shared_expanded = dist_shared.scale.repeat_interleave(
        n_samples, dim=0
    )  # (K*B, Z)
    mu_private_expanded = dist_private.loc.repeat_interleave(
        n_samples, dim=0
    )  # (K*B, W)
    sigma_private_expanded = dist_private.scale.repeat_interleave(
        n_samples, dim=0
    )  # (K*B, W)

    vae_i = model.vaes[view_index]

    # Recreate the new distributions of dist_shared and dist_private
    dist_shared_expanded = vae_i.qu_x(
        mu_shared_expanded, sigma_shared_expanded
    )  # (K*B, Z)
    dist_private_expanded = vae_i.qu_x(
        mu_private_expanded, sigma_private_expanded
    )  # (K*B, W)

    # Recreate the distributions, then compute the mutual information loss
    if model.params.gen_aug_loss_type is None:
        # NOTE: CL: contrastive loss (original), ML: matching loss
        gen_aug_loss_type = "CL"
    else:
        gen_aug_loss_type = model.params.gen_aug_loss_type

    if role == "roll_private":
        rec_dist_shared = vae_i.qu_x(shared_mu, shared_sigma)
        if gen_aug_loss_type == "CL":
            gen_aug_subloss = compute_mi_loss_twoviews(
                [dist_shared_expanded, rec_dist_shared],
                mi_estimator,
                use_mean_for_mi=use_mean_for_mi,
            )
        else:
            assert gen_aug_loss_type == "ML"
            gen_aug_subloss = compute_lsq_matching_loss(
                dist_shared_expanded,
                rec_dist_shared,
            )
    elif role == "roll_shared":
        rec_dist_private = vae_i.qu_x(private_mu, private_sigma)
        if gen_aug_loss_type == "CL":
            gen_aug_subloss = compute_mi_loss_twoviews(
                [dist_private_expanded, rec_dist_private],
                mi_estimator,
                use_mean_for_mi=use_mean_for_mi,
            )
        else:
            assert gen_aug_loss_type == "ML"
            gen_aug_subloss = compute_lsq_matching_loss(
                dist_private_expanded,
                rec_dist_private,
            )

    return gen_aug_subloss, aug_data


def compute_mi_loss_twoviews(dists_shared, mi_estimator, use_mean_for_mi=True):
    """
    Compute MI loss between two views.
    Currently called by gen_aug Loss

    Args:
        dists_shared (list): List of shared latent distributions (one for each view).
        mi_estimator: Mutual information estimator function.
        use_mean_for_mi (bool): If True, use mean of distributions for MI estimator, otherwise use samples.

    Returns:
        loss (torch.Tensor): mutual information loss between two views.
    """
    p_z1_given_v1, p_z2_given_v2 = dists_shared

    # z1, z2 shape: (K*B, Z)
    if use_mean_for_mi:
        z1 = p_z1_given_v1.loc
        z2 = p_z2_given_v2.loc
    else:
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()

    # Estimate mutual information
    mi_for_grad, _ = mi_estimator(z1, z2)
    mi_for_grad = mi_for_grad.mean()

    # We would like to maximize the mutual information.
    return -mi_for_grad


def compute_lsq_matching_loss(q1, q2):
    return torch.square(q1.loc - q2.loc).sum(-1).mean()


def compute_diffusion_prior_loss(model, shared_dists, private_dists):
    num_views = len(shared_dists)
    loss = 0.0
    pz = model.pz_diffusion
    # Losses are computed with the mean of distributions.
    for i in range(num_views):
        loss += pz(shared_dists[i].loc)
        pw = model.pws_diffusion[i]
        loss += pw(private_dists[i].loc)
    return loss / num_views


def compute_idmvae_loss(model, x, K=1, test=False):  # , current_ iteration=None
    """
    Compute the IDMVAE loss by combining elbo, cross view MI losses, and generative augmentation loss.

    Args:
        model: The MMVAE+ model instance.
        x: Input data (list of modalities).
        K: Number of samples for latent space resampling.
        test: Boolean indicating whether this is a test run.

    Returns:
        torch.Tensor: Combined loss (scalar).
    """
    device = next(model.parameters()).device

    # Compute ELBO loss (latent + reconstruction loss), and extract shared and private latents and their distributions
    (
        elbo_loss,
        recon_KL_sum_loss,
        llik_recon_loss,
        KL_div_loss,
        shared_latents,
        shared_dists,
        private_latents,
        private_dists,
    ) = compute_elbo_loss(model, x, K, test)
    total_loss = elbo_loss

    # =========================
    # Regularization terms
    # =========================

    # --- Reg 1: Cross view MI loss ---
    if model.params.cross_mi_loss_scale > 0.0:
        cross_mi_loss = compute_cross_mi_loss(
            shared_dists, model.contrast_mi, use_mean_for_mi=True
        )
        total_loss += model.params.cross_mi_loss_scale * cross_mi_loss
    else:
        cross_mi_loss = torch.tensor(0.0).to(device)

    # --- Reg 1.5: Direct shared-posterior alignment ---
    if model.params.z_alignment_loss_scale > 0.0:

        if model.params.priorposterior != "Normal":
            raise ValueError(
                "z_alignment_loss currently assumes Gaussian/Normal posteriors."
            )

        z_alignment_loss = compute_shared_z_alignment_loss(
            shared_dists
        )

        total_loss += (
                model.params.z_alignment_loss_scale
                * z_alignment_loss
        )
    else:
        z_alignment_loss = torch.tensor(0.0).to(device)

    # --- Reg 2: Generative augmentation loss ---
    if model.params.gen_aug_loss_scale > 0.0:
        gen_aug_loss = compute_gen_aug_loss(
            shared_latents,
            shared_dists,
            private_latents,
            private_dists,
            model.decoders,
            model.encoders,
            model.mi_shared,
            model.mi_private,
            model=model,
            n_samples=1,  # K or 1
            sampling_scheme=model.params.gen_aug_sampling_scheme,  # 'posterior' or 'diffusion_prior'
        )
        total_loss += model.params.gen_aug_loss_scale * gen_aug_loss
    else:
        gen_aug_loss = torch.tensor(0.0).to(device)

    # --- Reg 3: Diffusion Loss ---
    if model.params.diffusion_loss_weight > 0.0:
        diffusion_loss = compute_diffusion_prior_loss(model, shared_dists, private_dists)
        # print(f"diffusion_loss={diffusion_loss}")
        total_loss += model.params.diffusion_loss_weight * diffusion_loss
    else:
        diffusion_loss = torch.tensor(0.0).to(device)

    return (
        total_loss,
        recon_KL_sum_loss,
        llik_recon_loss,
        KL_div_loss,
        cross_mi_loss,
        z_alignment_loss,
        gen_aug_loss,
        diffusion_loss,
    )
