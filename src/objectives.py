# Objectives
import torch
from utils import log_mean_exp

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


def compute_elbo_loss(model, x, K=1, test=False):
    """
    Core ELBO computation for a single minibatch.

    Parameters:
        - model: MMVAE+ model instance.
        - x: Input data for the minibatch.
        - K: Number of samples for latent space resampling (steps).
        - test: Boolean indicating whether this is for testing.

    Returns:
        - torch.Tensor: Log weights (lws) for the minibatch.
        - shared_latents (list): List of shared latent tensors (one per view).
        - shared_dists (list): List of distributions for shared latents (one per view).
        - private_latents (list): List of private latent tensors (one per view).
        - private_dists (list): List of distributions for private latents (one per view).

    Math & Logic:

    1. Forward Pass:
    Calls the model to perform self- and cross-modal forward passes, generating:
    - q(u∣x): Posterior distributions of latents.
    - p(x∣u): Decoded data distributions.
    - u∼q(u∣x): Sampled latent codes.

    2. Decompose Latents:
    Splits the latents u into:
    - w: Modality-specific / Private latents.
    - z: Shared latents.

    3. Compute Likelihoods:
    Likelihoods of reconstructed data x under p(x∣u) for each modality.

    4. Compute KL Divergences:
    Between q(u∣x)and p(u), with u split into w and z :
    KL[q(z∣x)∣∣p(z)]+KL[q(w∣x)∣∣p(w)]

    5. Log-Weight Calculation:
    Combines the likelihoods and KL terms:
    lqz_x = log(1/M*∑p(z|x_i))
    lw=logp(x∣u)+β[logp(z)+logp(w)−logq(z∣x)−logq(w∣x)]

    """
    if test:
        # Posterior qu_xs shape: dist_list[M] -> dist.sample() -> (B, W+Z)
        # Likelihood px_us shape: dist_list[M][M] -> dist.sample() -> (K, B, C, H, V)
        # uss shape: sample_list[M] -> (K, B, W+Z)
        qu_xs, px_us, uss = model.self_and_cross_modal_generation_forward(x, K)
    else:
        qu_xs, px_us, uss = model(x, K)

    # Initialize lists to store shared and private latents and latent distributions for generative augmentation.
    shared_latents, shared_dists = [], []  # prior, posterior
    private_latents, private_dists = [], []

    # List of latent distributions for shared and private latents
    # r: len M, qu_x: dist -> loc, scale.shape: (B, W+Z), (B, W+Z)
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs):
        # qu_x_r_mean and qx_r_lv shape: loc, scale, (B, W+Z), (B, W+Z)
        qu_x_r_mean, qu_x_r_lv = model.vaes[
            r
        ].qu_x_params
        # (B, W+Z) -> (B, W), (B, Z)
        qw_x_mean, qz_x_mean = torch.split(
            qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1
        )
        qw_x_lv, qz_x_lv = torch.split(
            qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1
        )
        # qw_x shape: dist.sample()->(B, W)
        qw_x = model.vaes[r].qu_x(
            qw_x_mean, qw_x_lv
        )
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)

    shared_dists = qz_xs
    private_dists = qw_xs

    lws = []
    KL_divs = []
    llik_recons = []
    for r, qu_x in enumerate(qu_xs):
        # ws, zs shape: (K, B, W+Z) -> (K, B, W), (K, B, Z)
        ws, zs = torch.split(
            uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1
        )
        shared_latents.append(zs)
        private_latents.append(ws)
        # lpz shape: (K, B)
        # distribution.log_prob(zs) -> (K, B, Z) -> .sum(-1) -> (K, B)
        lpz = model.get_simple_prior_z().log_prob(zs).sum(-1)
        lpw = model.get_simple_prior_w(view=r, aux=False).log_prob(ws).sum(-1)
        # shape (K, B)
        lqz_x = log_mean_exp(
            torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs])
        )  # Mean of the views
        lqw_x = (
            qw_xs[r].log_prob(ws).sum(-1)
        )  # Modality-specific without mean of the views

        # Each row of px_us contains K distributions, each corresponding to the reconstruction from
        #   the z of view r, combined with the w of view d
        #   the reconstruction used the decoder in view d, therefore the llik_scaling for view d is used.
        # px_u: dist, [K, B, channel, height, width]
        # Each element of lpx_u has shape [K, B]
        lpx_u = [
            px_u.log_prob(x[d])
            .view(*px_u.batch_shape[:2], -1)
            .mul(model.vaes[d].llik_scaling)
            .sum(-1)
            for d, px_u in enumerate(px_us[r])
        ]

        # The reconstruction log_probs for different target view d, and the same source view r, are summed.
        lpx_u = torch.stack(lpx_u).sum(0)

        # shape (K, B)
        lw = lpx_u + model.params.beta * (lpz + lpw - lqz_x - lqw_x)
        lws.append(lw)

        # Reconstruction term
        llik_recon = lpx_u
        llik_recons.append(llik_recon)

        # KL divergence term
        KL_div = lpz + lpw - lqz_x - lqw_x
        KL_divs.append(KL_div)

    # shape: (M, K, B)
    lws = torch.stack(lws)
    llik_recons_stk = torch.stack(llik_recons)
    KL_divs_stk = torch.stack(KL_divs)

    # log_mean_exp removes the "K" dimension, mean(0) removes "M" by averaging over (source) views, mean() average over batch
    elbo_loss = -log_mean_exp(lws, dim=1).mean(0).mean()
    llik_recon_loss = -log_mean_exp(llik_recons_stk, dim=1).mean(0).mean()
    KL_div_loss = -log_mean_exp(KL_divs_stk, dim=1).mean(0).mean()

    recon_KL_sum_loss = llik_recon_loss + model.params.beta * KL_div_loss

    # shared_latents and private_latents are lists of length M, where each element is of shape [K, B, Z/W]
    # shared_dists and private_dists are lists of length M, where each element is a distribution with parameters of the shape [B, Z/W]
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

    aug_data_raw = decoder(aug_in)

    # Handle different decoder outputs:
    if isinstance(aug_data_raw, tuple):
        aug_data_K = aug_data_raw[0]  # Extract reconstructed image mean
    elif isinstance(aug_data_raw, list):
        aug_data_K = aug_data_raw[
            0
        ]  # Extract the first tensor (for CUB text), [1, 32, 32, 1590]
    else:
        raise ValueError("aug_data_raw is wrong!")

    # Flatten `aug_data` to (K*B, C, H, V) or (K*B, vocab_size)
    aug_data = aug_data_K.view(-1, *aug_data_K.shape[2:])

    # Encode augmented data to get new latents
    # mu, sigma shape: (K*B, W+Z)
    mu, sigma = encoder(aug_data)  # Normal or Laplace: mu(loc) and sigma(scale)

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
        gen_aug_loss,
        diffusion_loss,
    )
