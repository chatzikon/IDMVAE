import os
import shutil
import numpy as np
from statistics import mean
from scipy import stats
import math
from collections import defaultdict

from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, unused import

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import DataLoader, Subset

import gc

import warnings

import wandb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision.utils import save_image, make_grid

from utils import unpack_data_PM_quadrant as unpack_data_PolyMNIST
from utils import unpack_data_CUBcluster8, make_mlp

from utils import EmbeddedDatasetWithPriors, build_matrix, EmbeddedDataset_visualization as EmbeddedDataset

import random

from utils import CrossModalEvalForwardMode, get_mean


def _decode_matrix_entries(model, matrix, entries):
    if matrix is None or not model.use_pretrain_feats:
        return
    for r, c in entries:
        if matrix[r][c] is None:
            continue
        decoded = model._decode_latents_to_pixels(matrix[r][c].squeeze(0))
        matrix[r][c] = decoded.unsqueeze(0)


def _decode_eval_latents(
    model,
    vae,
    modality_idx,
    latents,
):
    """
    Decode latent variables during evaluation.

    CNN/image decoders:
        return a likelihood distribution.

    BART text decoder:
        return autoregressively generated token IDs.
        No ground-truth text is supplied.
    """

    text_decoder_arch = getattr(
        model.params,
        "text_decoder_arch",
        "cnn",
    )

    if (
        modality_idx == 1
        and text_decoder_arch == "bart"
    ):
        return vae.dec.generate(
            latents,
            max_new_tokens=model.params.bart_max_length,
        )

    return vae.decode_likelihood(
        latents
    )


def idmvae_generate_unconditional(
    model,
    N,
):
    with torch.no_grad():

        data = []

        # -----------------------------------------------------
        # Shared latent z
        # -----------------------------------------------------

        if model.diffusion_loss_weight > 0.0:
            pz = model.pz_diffusion
        else:
            pz = model.get_simple_prior_z()

        latents_z = pz.rsample(
            torch.Size([N])
        )

        # -----------------------------------------------------
        # Generate each modality
        # -----------------------------------------------------

        for d, vae in enumerate(model.vaes):

            if model.diffusion_loss_weight > 0.0:
                pw = model.pws_diffusion[d]
            else:
                pw = model.get_simple_prior_w(
                    view=d,
                    aux=False,
                )

            latents_w = pw.rsample(
                [latents_z.size(0)]
            )

            latents = torch.cat(
                (
                    latents_w,
                    latents_z,
                ),
                dim=-1,
            )

            decoded = _decode_eval_latents(
                model,
                vae,
                d,
                latents,
            )

            # -------------------------------------------------
            # BART:
            #
            # generated IDs approximately [N,1,L]
            # -> [N,L]
            # -------------------------------------------------

            if torch.is_tensor(decoded):

                generated = decoded.reshape(
                    -1,
                    decoded.size(-1),
                )

                data.append(generated)

            # -------------------------------------------------
            # Existing image / CNN text behaviour.
            # -------------------------------------------------

            else:

                mean = get_mean(decoded)

                generated = mean.view(
                    -1,
                    *mean.size()[2:]
                )

                data.append(generated)

    return data


def idmvae_self_and_cross_modal_generation_impl(
    model,
    data,
    K=1,
    condition_type=None,
    mode=CrossModalEvalForwardMode.PRIOR,
    data_ctrl=None,
):

    if (
        mode == CrossModalEvalForwardMode.POSTERIOR_CTRL
        and data_ctrl is None
    ):
        raise ValueError(
            "data_ctrl is required when mode is POSTERIOR_CTRL."
        )

    M = len(model.vaes)

    dw = model.params.latent_dim_w
    dz = model.params.latent_dim_z

    qu_xs = []
    uss = []

    px_us = [
        [None for _ in range(M)]
        for _ in range(M)
    ]

    uss_ctrl = None

    # =========================================================
    # 1. ENCODE NORMAL INPUTS
    # =========================================================
    #
    # Important:
    #
    # Do NOT call:
    #
    #     vae(data[m], K=K)
    #
    # because that also performs decoding.
    #
    # During BART evaluation decoding must be autoregressive
    # and must not receive GT target tokens.
    # =========================================================

    for m, vae in enumerate(model.vaes):

        vae._qu_x_params = vae.enc(
            data[m]
        )

        qu_x = vae.qu_x(
            *vae._qu_x_params
        )

        us = qu_x.rsample(
            torch.Size([K])
        )

        qu_xs.append(qu_x)
        uss.append(us)

        # -----------------------------------------------------
        # POSTERIOR_NONSHUF keeps the diagonal reconstruction.
        # -----------------------------------------------------

        if mode == CrossModalEvalForwardMode.POSTERIOR_NONSHUF:

            px_us[m][m] = _decode_eval_latents(
                model,
                vae,
                m,
                us,
            )

    # =========================================================
    # 2. ENCODE CONTROL INPUTS
    # =========================================================

    if mode == CrossModalEvalForwardMode.POSTERIOR_CTRL:

        uss_ctrl = []

        for m, vae in enumerate(model.vaes):

            ctrl_params = vae.enc(
                data_ctrl[m]
            )

            qu_ctrl = vae.qu_x(
                *ctrl_params
            )

            us_ctrl = qu_ctrl.rsample(
                torch.Size([K])
            )

            uss_ctrl.append(
                us_ctrl
            )

    # =========================================================
    # 3. SELF / CROSS-MODAL GENERATION
    # =========================================================

    for e, us_e in enumerate(uss):

        latents_w_e, latents_z_e = torch.split(
            us_e,
            [dw, dz],
            dim=-1,
        )

        for d, vae_d in enumerate(model.vaes):

            if (
                mode == CrossModalEvalForwardMode.POSTERIOR_NONSHUF
                and e == d
            ):
                continue

            # =================================================
            # PRIOR / PRIOR_CTRL
            # =================================================

            if mode in (
                CrossModalEvalForwardMode.PRIOR,
                CrossModalEvalForwardMode.PRIOR_CTRL,
            ):

                if model.diffusion_loss_weight > 0.0:

                    pz = model.pz_diffusion
                    pw = model.pws_diffusion[d]

                else:

                    pz = model.get_simple_prior_z()

                    pw = model.get_simple_prior_w(
                        view=d,
                        aux=False,
                    )

                if mode == CrossModalEvalForwardMode.PRIOR:

                    latents_w_new = pw.rsample(
                        torch.Size([
                            us_e.size(0),
                            us_e.size(1),
                        ])
                    ).squeeze(2)

                    latents_z_new = pz.rsample(
                        torch.Size([
                            us_e.size(0),
                            us_e.size(1),
                        ])
                    ).squeeze(2)

                else:
                    # PRIOR_CTRL

                    latent_w_new = pw.rsample(
                        torch.Size([
                            us_e.size(0),
                            1,
                        ])
                    ).squeeze(2)

                    latent_z_new = pz.rsample(
                        torch.Size([
                            us_e.size(0),
                            1,
                        ])
                    ).squeeze(2)

                    latents_w_new = latent_w_new.repeat(
                        1,
                        us_e.size(1),
                        1,
                    )

                    latents_z_new = latent_z_new.repeat(
                        1,
                        us_e.size(1),
                        1,
                    )

                # ---------------------------------------------
                # Preserve shared z from source e.
                # Decode using TARGET modality d.
                # ---------------------------------------------

                if (
                    condition_type is None
                    or condition_type == "shared"
                ):

                    us_new = torch.cat(
                        (
                            latents_w_new,
                            latents_z_e,
                        ),
                        dim=-1,
                    )

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        vae_d,
                        d,
                        us_new,
                    )

                # ---------------------------------------------
                # Preserve private w from source e.
                #
                # Existing IDMVAE behaviour decodes with the
                # SOURCE modality e.
                # ---------------------------------------------

                elif condition_type == "private":

                    us_new = torch.cat(
                        (
                            latents_w_e,
                            latents_z_new,
                        ),
                        dim=-1,
                    )

                    src_vae = model.vaes[e]

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        src_vae,
                        e,
                        us_new,
                    )

            # =================================================
            # POSTERIOR
            # =================================================

            elif mode == CrossModalEvalForwardMode.POSTERIOR:

                us_rs = qu_xs[d].rsample(
                    torch.Size([K])
                )

                latents_w_d, latents_z_d = torch.split(
                    us_rs,
                    [dw, dz],
                    dim=-1,
                )

                shift_w = random.randint(
                    1,
                    10,
                )

                shift_z = random.randint(
                    1,
                    10,
                )

                if (
                    condition_type is None
                    or condition_type == "shared"
                ):

                    us_new = torch.cat(
                        (
                            torch.roll(
                                latents_w_d,
                                shifts=shift_w,
                                dims=1,
                            ),
                            latents_z_e,
                        ),
                        dim=-1,
                    )

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        vae_d,
                        d,
                        us_new,
                    )

                elif condition_type == "private":

                    us_new = torch.cat(
                        (
                            latents_w_e,
                            torch.roll(
                                latents_z_d,
                                shifts=shift_z,
                                dims=1,
                            ),
                        ),
                        dim=-1,
                    )

                    src_vae = model.vaes[e]

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        src_vae,
                        e,
                        us_new,
                    )

            # =================================================
            # POSTERIOR_CTRL
            # =================================================

            elif mode == CrossModalEvalForwardMode.POSTERIOR_CTRL:

                us_d_ctrl = uss_ctrl[d]

                (
                    latents_w_d_ctrl,
                    latents_z_d_ctrl,
                ) = torch.split(
                    us_d_ctrl,
                    [dw, dz],
                    dim=-1,
                )

                if (
                    condition_type is None
                    or condition_type == "shared"
                ):

                    us_new = torch.cat(
                        (
                            latents_w_d_ctrl,
                            latents_z_e,
                        ),
                        dim=-1,
                    )

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        vae_d,
                        d,
                        us_new,
                    )

                elif condition_type == "private":

                    us_new = torch.cat(
                        (
                            latents_w_e,
                            latents_z_d_ctrl,
                        ),
                        dim=-1,
                    )

                    src_vae = model.vaes[e]

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        src_vae,
                        e,
                        us_new,
                    )

            # =================================================
            # POSTERIOR_NONSHUF
            # =================================================

            else:

                us_rs = qu_xs[d].rsample(
                    torch.Size([K])
                )

                latents_w_d, latents_z_d = torch.split(
                    us_rs,
                    [dw, dz],
                    dim=-1,
                )

                if (
                    condition_type is None
                    or condition_type == "shared"
                ):

                    us_new = torch.cat(
                        (
                            latents_w_d,
                            latents_z_e,
                        ),
                        dim=-1,
                    )

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        vae_d,
                        d,
                        us_new,
                    )

                elif condition_type == "private":

                    us_new = torch.cat(
                        (
                            latents_w_e,
                            latents_z_d,
                        ),
                        dim=-1,
                    )

                    src_vae = model.vaes[e]

                    px_us[e][d] = _decode_eval_latents(
                        model,
                        src_vae,
                        e,
                        us_new,
                    )

    return (
        qu_xs,
        px_us,
        uss,
    )
def idmvae_recon_matrix_from_px_us(
    px_us,
):
    """
    Convert decoder outputs into reconstruction tensors.

    Normal likelihood distributions:
        use get_mean(...)

    BART generation:
        output is already a tensor of generated token IDs.
    """

    recons = []

    for row in px_us:

        recon_row = []

        for px_u in row:

            if torch.is_tensor(px_u):
                recon = px_u
            else:
                recon = get_mean(px_u)

            recon_row.append(
                recon
            )

        recons.append(
            recon_row
        )

    return recons


def idmvae_finalize_cross_modal_recons(model, recons, condition_type, return_denoised):
    denoised = None
    if return_denoised and hasattr(model, "_denoise_recon_matrix"):
        denoised = model._denoise_recon_matrix(recons)
    if condition_type == "shared" and model.use_pretrain_feats:
        _decode_matrix_entries(model, recons, [(0, 0), (1, 0)])
        if denoised is not None:
            _decode_matrix_entries(model, denoised, [(0, 0), (1, 0)])
    elif condition_type == "private" and model.use_pretrain_feats:
        _decode_matrix_entries(model, recons, [(0, 0), (0, 1)])
        if denoised is not None:
            _decode_matrix_entries(model, denoised, [(0, 0), (0, 1)])
    if return_denoised:
        return recons, denoised
    return recons


def idmvae_self_and_cross_modal_generation_eval(
    model,
    data,
    *,
    mode=None,
    condition_type=None,
    return_denoised=False,
    data_ctrl=None,
    K=1,
):
    if mode is None:
        with torch.no_grad():
            _, px_us, _ = model.self_and_cross_modal_generation_forward(data, K=K)
        return idmvae_recon_matrix_from_px_us(px_us)

    if mode == CrossModalEvalForwardMode.POSTERIOR_CTRL and data_ctrl is None:
        raise ValueError("data_ctrl is required when mode is POSTERIOR_CTRL.")
    if condition_type is not None and condition_type not in ("shared", "private"):
        raise ValueError(f"Unknown condition_type '{condition_type}'")

    with torch.no_grad():
        _, px_us, _ = idmvae_self_and_cross_modal_generation_impl(
            model,
            data,
            K=K,
            condition_type=condition_type,
            mode=mode,
            data_ctrl=data_ctrl,
        )
    recons = idmvae_recon_matrix_from_px_us(px_us)
    return idmvae_finalize_cross_modal_recons(model, recons, condition_type, return_denoised)


def setup_pretrained_denoiser(params, device):
    """Optionally load a pretrained DiT denoiser for latent refinement."""
    denoiser_model = None
    denoiser_diffusion = None
    denoiser_device = device
    denoiser_condition_label = getattr(params, "denoiser_class_label", None)
    ckpt_path = getattr(params, "denoiser_ckpt", None)
    if not ckpt_path:
        return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Denoiser checkpoint not found at {ckpt_path}.")
        return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label
    try:
        from models.dit_denoiser import DiT_models
        from models.dit_diffusion import create_diffusion
    except Exception as exc:
        print(f"[WARN] Unable to import DiT denoiser modules: {exc}")
        return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label

    model_name = getattr(params, "denoiser_model", "DiT-XL/2")
    if model_name not in DiT_models:
        print(f"[WARN] Unknown denoiser model '{model_name}'. Available: {list(DiT_models.keys())}")
        return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label

    latent_size = getattr(params, "img_size", 32)
    num_classes = getattr(params, "denoiser_num_classes", 1000)
    try:
        model = DiT_models[model_name](input_size=latent_size, num_classes=num_classes).to(device)
    except Exception as exc:
        print(f"[WARN] Failed to construct denoiser model '{model_name}': {exc}")
        return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label

    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict):
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        print(f"[WARN] Unable to load denoiser weights from {ckpt_path}: {exc}")
        return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label

    model.eval()
    steps = str(getattr(params, "denoiser_num_sampling_steps", 250))
    denoiser_model = model
    denoiser_diffusion = create_diffusion(steps)
    return denoiser_model, denoiser_diffusion, denoiser_device, denoiser_condition_label


def has_pretrained_denoiser(denoiser_model, denoiser_diffusion):
    return denoiser_model is not None and denoiser_diffusion is not None


def run_pretrained_denoiser(
    denoiser_model,
    denoiser_diffusion,
    denoiser_device,
    denoiser_condition_label,
    noisy_latents,
):
    if not has_pretrained_denoiser(denoiser_model, denoiser_diffusion):
        return None
    if noisy_latents is None:
        return None
    if noisy_latents.dim() != 4:
        return None

    latent = noisy_latents.to(denoiser_device)
    b, c, h, w = latent.shape
    z = torch.randn(b, c, h, w, device=denoiser_device)
    if denoiser_condition_label is None:
        y = denoiser_model.num_classes * torch.ones(b, dtype=torch.int32, device=denoiser_device)
    else:
        y = torch.full((b,), int(denoiser_condition_label), dtype=torch.int32, device=denoiser_device)

    model_kwargs = dict(y=y, noisy_x=latent)
    with torch.no_grad():
        samples = denoiser_diffusion.p_sample_loop(
            denoiser_model.forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=denoiser_device,
        )
    return samples.detach().cpu()


def train_latent_logistic(model, dataloader, device, args):
    """
    Gather the *mean* shared‐latent (Z) representations over `dataloader`
    and fit a sklearn LogisticRegression to predict the binary label.
    """
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for views_list, mask, labels in dataloader:
            # move to device
            views_list = [v.to(device) for v in views_list]
            labels = labels.view(-1).cpu().numpy()
            # for each view, encode to get posterior mean of z
            z_means = []
            for v, vae in enumerate(model.vaes):
                mu, logvar = vae.enc(views_list[v])
                # split into w and z
                _, qz_mu = torch.split(
                    mu, [args.latent_dim_w, args.latent_dim_z], dim=-1
                )
                z_means.append(qz_mu)
            # fuse by averaging across views
            z_mean = torch.stack(z_means, dim=0).mean(dim=0)  # (B, Z)
            zs.append(z_mean.cpu().numpy())
            ys.append(labels)
    X = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0)
    clf = LogisticRegression(random_state=args.seed, solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    return clf


def evaluate_latent_logistic(model, dataloader, clf, device, args):
    """
    Run the sklearn classifier on held-out latents and return accuracy + confusion matrix.
    """
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for views_list, mask, labels in dataloader:
            views_list = [v.to(device) for v in views_list]
            labels = labels.view(-1).cpu().numpy()
            z_means = []
            for v, vae in enumerate(model.vaes):
                mu, logvar = vae.enc(views_list[v])
                _, qz_mu = torch.split(
                    mu, [args.latent_dim_w, args.latent_dim_z], dim=-1
                )
                z_means.append(qz_mu)
            z_mean = torch.stack(z_means, dim=0).mean(dim=0)
            zs.append(z_mean.cpu().numpy())
            ys.append(labels)
    X = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return acc, cm

def classify_linear_latent_representations(
    clf_lr, data, labels, split=False
):  # dict, [[B,U],[B,W],[B,Z]][M], [B,]
    gt = labels.cpu().numpy()
    accuracies = dict()

    # Create a dictionary to store predictions
    predictions = dict()

    y_preds_z = []
    y_preds_w = []
    for k, data_k in enumerate(data):
        if split:
            data_rep_u, data_rep_w, data_rep_z = data_k  # [B,U],[B,W],[B,Z]
        else:
            data_rep_u = data[0][0]

        clf_key_u = "m" + str(k) + "_" + "u"
        clf_lr_rep_u = clf_lr[clf_key_u]
        y_pred_rep_u = clf_lr_rep_u.predict(
            data_rep_u
        )  # (B,), predicted labels (hard prediction)

        # ADDED: Store confidence scores
        probas_rep_u = clf_lr_rep_u.predict_proba(data_rep_u)
        confidence_rep_u = np.max(probas_rep_u, axis=1)  # (B,), confidence scores

        accuracy_rep_u = accuracy_score(
            gt, y_pred_rep_u.ravel()
        )  # same as y_pred_rep_u
        accuracies[clf_key_u] = accuracy_rep_u
        # Store predictions
        predictions[clf_key_u] = y_pred_rep_u
        # Store confidence scores
        predictions[clf_key_u + "_confidence"] = confidence_rep_u

        if split:

            clf_key_z = "m" + str(k) + "_" + "z"
            clf_lr_rep_z = clf_lr[clf_key_z]
            y_pred_rep_z = clf_lr_rep_z.predict(data_rep_z)
            accuracy_rep_z = accuracy_score(gt, y_pred_rep_z.ravel())
            accuracies[clf_key_z] = accuracy_rep_z
            # Store predictions
            predictions[clf_key_z] = y_pred_rep_z
            # ADDED: Store confidence scores
            probas_rep_z = clf_lr_rep_z.predict_proba(data_rep_z)
            confidence_rep_z = np.max(probas_rep_z, axis=1)  # (B,), confidence scores
            predictions[clf_key_z + "_confidence"] = confidence_rep_z

            clf_key_w = "m" + str(k) + "_" + "w"
            clf_lr_rep_w = clf_lr[clf_key_w]
            y_pred_rep_w = clf_lr_rep_w.predict(data_rep_w)
            accuracy_rep_w = accuracy_score(gt, y_pred_rep_w.ravel())
            accuracies[clf_key_w] = accuracy_rep_w
            # Store predictions
            predictions[clf_key_w] = y_pred_rep_w
            # ADDED: Store confidence scores
            probas_rep_w = clf_lr_rep_w.predict_proba(data_rep_w)
            confidence_rep_w = np.max(probas_rep_w, axis=1)  # (B,), confidence scores
            predictions[clf_key_w + "_confidence"] = confidence_rep_w

            y_preds_z.append(y_pred_rep_z)
            y_preds_w.append(y_pred_rep_w)
    overall_preds_z = stats.mode(np.stack(y_preds_z))[0]
    overall_preds_w = stats.mode(np.stack(y_preds_w))[0]
    accuracy_rep_all_z = accuracy_score(gt, overall_preds_z.ravel())
    accuracy_rep_all_w = accuracy_score(gt, overall_preds_w.ravel())
    # accuracies['all'] = accuracy_rep_all
    accuracies["all_z"] = accuracy_rep_all_z
    accuracies["all_w"] = accuracy_rep_all_w

    # Store ground truth
    predictions["ground_truths"] = gt

    return accuracies, predictions


# t-SNE or UMAP latent visualization
def visualize_latents_with_priors(
    model,
    vae,
    encoder,
    valid_loader,
    test_loader,
    encoder_name,
    mismatch_type,
    device="cuda",
    figure=1,
    n_prior=2500,
    condition_type=None,
    view_index=None,
    batch_size=128,
    visualize_ratio=0.25,  # Ratio of val/test set to visualize
    use_pca=False,
    plot_method="umap",  # 'umap' or 'tsne'
    plot_3d=False,
    save_file=None,
):
    """
    Embed posterior and prior latent samples, then visualize together.
    Prior samples are drawn from standard normal in chosen subspace ('shared' or 'private').
    """

    # Get prior distributions
    # Parameter suffix names kept as-is for backward compatibility.
    pz = model.get_simple_prior_z()  # non-learnable
    pw = model.get_simple_prior_w(view=view_index, aux=False)  # non-learnable
    pw_aux = model.get_simple_prior_w(
        view=view_index, aux=True
    )  # learnable or aux prior
    # pw_learned = model.get_prior_w(view=view_index)
    if model.diffusion_loss_weight > 0.0:
        print("Using diffusion prior for visualization...(diff_lw=%.2f)" % model.diffusion_loss_weight)
        pz_diffusion = model.pz_diffusion
        pw_diffusion = model.pws_diffusion[view_index]
    else:
        print("pz None or using pw_aux prior for visualization...(diff_lw=%.2f)" % model.diffusion_loss_weight)
        # pz_diffusion = None  # pz_learned
        # pw_diffusion = None  # pw_learned
        pz_learned = None  # flag to disable ax4 in the fig.
        pz_diffusion = pz
        pw_diffusion = (
            pw_aux  # NOTE:if want to load the checkpoint to double check the pw_aux,
        )
        #      change the .sh diffusion_loss_weight to 0.0 and load the diffusion checkpoint.

    # 1) Embed posterior: same as before
    # embedded_valid_set = EmbeddedDataset(
    #     base_dataloader=valid_loader,
    #     vae=vae,
    #     encoder=encoder,
    #     device=device,
    #     condition_type=condition_type,
    #     use_mean=model.params.use_mean_in_latent_visualization,
    #     view_idx=view_index,
    #     batch_size=batch_size,
    # )
    # ---------------------------------------------------------
    # Subsample BEFORE latent/RGB materialization
    # ---------------------------------------------------------
    base_test_dataset = test_loader.dataset
    total = len(base_test_dataset)

    sample_size_float = total * visualize_ratio
    sample_size_int = min(
        total,
        math.floor(sample_size_float),
    )

    if sample_size_int < total:
        sample_size_int = (sample_size_int // batch_size) * batch_size


    print(
        f"Total samples: {total}, "
        f"Sample size for visualization: "
        f"{sample_size_int}({sample_size_float})"
    )

    indices = np.random.choice(
        total,
        sample_size_int,
        replace=False,
    )

    # Keep deterministic/original ordering inside the subset.
    indices = np.sort(indices)

    sampled_test_dataset = Subset(
        base_test_dataset,
        indices.tolist(),
    )

    sampled_test_loader = DataLoader(
        sampled_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
        if isinstance(device, torch.device)
        else str(device).startswith("cuda"),
    )

    n_prior = min(n_prior, sample_size_int)



    embedded_test_set = EmbeddedDataset(
        base_dataloader=sampled_test_loader,
        vae=vae,
        encoder=encoder,
        device=device,
        condition_type=condition_type,
        use_mean=model.params.use_mean_in_latent_visualization,
        view_idx=view_index,
        batch_size=batch_size,
    )
    # original_valid_set = EmbeddedDataset(
    #     base_dataloader=valid_loader,
    #     vae=vae,
    #     encoder=None,
    #     device=device,
    #     condition_type=condition_type,
    #     use_mean=model.params.use_mean_in_latent_visualization,
    #     view_idx=view_index,
    #     batch_size=batch_size,
    # )
    # original_test_set = EmbeddedDataset(
    #     base_dataloader=test_loader,
    #     vae=vae,
    #     encoder=None,
    #     device=device,
    #     condition_type=condition_type,
    #     use_mean=model.params.use_mean_in_latent_visualization,
    #     view_idx=view_index,
    #     batch_size=batch_size,
    # )

    # -- Step 2: Convert to matrices
    # Convert the two sets into 2D matrices for evaluation
    #FX_valid, Y_valid = build_matrix(
    #    embedded_valid_set
    #)  # numpy.ndarray, shape: (N_v, W+Z), (N_v,)
    FX_test, Y_test = build_matrix(embedded_test_set)  # shape: (N_t, W+Z), (N_t,)
    #X_valid, _ = build_matrix(original_valid_set)  # shape: (N_t, C, H, W)
    #X_test, _ = build_matrix(original_test_set)  # shape: (N_t, C, H, W)
    # X_test, Y_test_orig = build_matrix(original_test_set)

    # 2) Split into shared (Z) and private (W)
    # --Step 3: Random subsample the test set
    # total = len(FX_test)
    # # total = len(FX_valid)
    # sample_size_float = total * visualize_ratio #// 4  # Adjust ratio as needed
    # sample_size_int = min(total, math.floor(sample_size_float))
    # print(f"Total samples: {total}, Sample size for visualization: {sample_size_int}({sample_size_float})")
    # indices = np.random.choice(total, sample_size_int, replace=False)
    # n_prior = min(n_prior, sample_size_int)  # Ensure n_prior does not exceed sample size
    #
    # FX_test = FX_test[indices]
    # #X_test = X_test[indices]
    # Y_test = Y_test[indices]  # = Y_test_orig[indices]

    X_test = None
    original_test_set = None

    if view_index == 0:
        original_test_set = EmbeddedDataset(
            base_dataloader=sampled_test_loader,
            vae=vae,
            encoder=None,
            device=device,
            condition_type=condition_type,
            use_mean=model.params.use_mean_in_latent_visualization,
            view_idx=view_index,
            batch_size=batch_size,
        )

        X_test, _ = build_matrix(original_test_set)

        # Keep the same samples as the latent representation
        #X_test = X_test[indices]


    W_dim = model.params.latent_dim_w
    Z_dim = model.params.latent_dim_z
    #FX_valid_Z = FX_valid[:, W_dim:]
    FX_test_Z = FX_test[:, W_dim:]
    #FX_valid_W = FX_valid[:, :W_dim]
    FX_test_W = FX_test[:, :W_dim]

    # 3) Choose latent subspace (shared)
    # sub_valid = FX_test_Z
    # sub_valid_labels = Y_test

    # --Step 5: Choose latent space to visualize
    # Prior sampling currently uses rsample; mean-based prior option is not enabled.
    if condition_type is None or condition_type == "shared":
        # -> standard visualization: shared label - latent space Z
        #latent_space_valid = FX_valid_Z
        latent_space_test = FX_test_Z
        # Sample non-learnable prior
        prior_non = (
            pz.rsample(torch.Size((n_prior,))).squeeze(1).cpu().numpy()
        )  # .squeeze(2)
        # Sample learnable prior
        prior_diffusion = (
            pz_diffusion.rsample(torch.Size((n_prior,)))
            .squeeze(1)
            .cpu()
            .detach()
            .numpy()
        )  # .squeeze(2)

        # -> mismatch visualization: shared label - latent space W
        #latent_space_valid_mismatch = FX_valid_W
        latent_space_test_mismatch = FX_test_W
        # Sample non-learnable prior
        prior_non_mismatch = (
            pw.rsample(torch.Size((n_prior,))).squeeze(1).cpu().numpy()
        )
        # Sample learnable prior
        prior_diffusion_mismatch = (
            pw_diffusion.rsample(torch.Size((n_prior,)))
            .squeeze(1)
            .cpu()
            .detach()
            .numpy()
        )
        # n_labels = 10
    elif condition_type == "private":
        # -> standard visualization: private label - latent space W
        #latent_space_valid = FX_valid_W
        latent_space_test = FX_test_W
        # Sample non-learnable prior
        prior_non = pw.rsample(torch.Size((n_prior,))).squeeze(1).cpu().numpy()
        # Sample learnable prior
        prior_diffusion = (
            pw_diffusion.rsample(torch.Size((n_prior,)))
            .squeeze(1)
            .cpu()
            .detach()
            .numpy()
        )

        # -> mismatch visualization: private label - latent space Z
        #latent_space_valid_mismatch = FX_valid_Z
        latent_space_test_mismatch = FX_test_Z
        # Sample non-learnable prior
        prior_non_mismatch = pz.rsample(torch.Size((n_prior,))).squeeze(1).cpu().numpy()
        # Sample learnable prior
        prior_diffusion_mismatch = (
            pz_diffusion.rsample(torch.Size((n_prior,)))
            .squeeze(1)
            .cpu()
            .detach()
            .numpy()
        )

        # n_labels = 4
    else:
        raise ValueError("Unsupported condition_type: must be 'shared' or 'private'")

    # 5) Concatenate
    # # X_all = np.vstack([sub_valid, prior_samples])
    # # labels_all = np.concatenate([sub_valid_labels, [-1]*n_prior])  # prior label = -1
    # # X_all = np.vstack([latent_space_valid, prior_samples])
    # # labels_all = np.concatenate([Y_valid, [-1]*n_prior])  # prior label = -1
    # X_all = np.vstack([latent_space_test, prior_samples])
    # labels_all = np.concatenate([Y_test, [-1]*n_prior])  # prior label = -1

    # 6) Prepare projections for non-learnable and learnable priors
    # # Sample non-learnable prior
    # prior_non = pw.rsample(torch.Size((n_prior,))).squeeze(1).cpu().detach().numpy()
    # # Sample learnable prior
    # prior_learn = pw_learnable.rsample(torch.Size((n_prior,))).squeeze(1).cpu().detach().numpy()

    # Stack posterior and priors
    # -> Standard visualization
    X_all_non = np.vstack([latent_space_test, prior_non])
    X_all_diffusion = np.vstack([latent_space_test, prior_diffusion])
    # -> Mismatch visualization
    X_all_non_mismatch = np.vstack([latent_space_test_mismatch, prior_non_mismatch])
    X_all_diffusion_mismatch = np.vstack(
        [latent_space_test_mismatch, prior_diffusion_mismatch]
    )

    # Labels: posterior labels Y_test, priors as -1
    labels_all_non = np.concatenate([Y_test, [-1] * n_prior])
    labels_all_learn = np.concatenate([Y_test, [-1] * n_prior])

    # parameters:
    umap_n_neighbors = model.params.lv_umap_n_neighbors
    umap_min_dist = model.params.lv_umap_min_dist
    print(f"UMAP parameters: n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}")

    # Helper to reduce dimensionality
    def reduce_space(X):
        if use_pca:
            X = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
        if plot_method == "tsne":
            return TSNE(n_components=3 if plot_3d else 2, random_state=0).fit_transform(
                X
            )
        reducer = umap.UMAP(
            n_components=3 if plot_3d else 2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=0,
        )
        return reducer.fit_transform(X)

    # Compute projections
    # -> Standard visualization
    X_proj_non = reduce_space(X_all_non)
    X_proj_diffusion = reduce_space(X_all_diffusion)
    # -> Mismatch visualization
    X_proj_non_mismatch = reduce_space(X_all_non_mismatch)
    X_proj_diffusion_mismatch = reduce_space(X_all_diffusion_mismatch)
    # Posterior-only coordinates (first len(Y_test) points)
    N_test = len(Y_test)
    X_proj_post = X_proj_non[:N_test]
    #X_proj_post_diffusion = X_proj_diffusion[:N_test]  # Added to keep posterior-only diffusion coordinates aligned.
    X_proj_post_mismatch = X_proj_non_mismatch[:N_test]
    #X_proj_post_diffusion_mismatch = X_proj_diffusion_mismatch[:N_test]

    # 7) Plot: four panels side by side
    if plot_3d:
        # -> Standard visualization
        fig = plt.figure(figsize=(24, 6))
        ax1 = fig.add_subplot(1, 4, 1, projection="3d")  # 1x4 layout used for side-by-side comparison.
        ax2 = fig.add_subplot(1, 4, 2, projection="3d")
        ax3 = fig.add_subplot(1, 4, 3, projection="3d")
        ax4 = fig.add_subplot(1, 4, 4, projection="3d")
        # -> Mismatch visualization
        fig_mis = plt.figure(figsize=(24, 6))
        ax1_mis = fig_mis.add_subplot(1, 4, 1, projection="3d")
        ax2_mis = fig_mis.add_subplot(1, 4, 2, projection="3d")
        ax3_mis = fig_mis.add_subplot(1, 4, 3, projection="3d")
        ax4_mis = fig_mis.add_subplot(1, 4, 4, projection="3d")

    else:
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(24,6))
        # -> Standard visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        ax1, ax2, ax3, ax4 = axes.flatten()
        # -> Mismatch visualization
        fig_mis, axes_mis = plt.subplots(2, 2, figsize=(14, 11))
        ax1_mis, ax2_mis, ax3_mis, ax4_mis = axes_mis.flatten()

    # Panel 1: posterior only
    # -> Standard visualization
    for lbl in np.unique(Y_test):
        pts = X_proj_post[Y_test == lbl]
        if plot_3d:
            ax1.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.6, s=10, label=f"class {lbl}"
            )
        else:
            ax1.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10, label=f"class {lbl}")
    ax1.set_title(f"{encoder_name} posterior only")
    ax1.legend(loc="best")
    # -> Mismatch visualization
    for lbl in np.unique(Y_test):
        pts = X_proj_post_mismatch[Y_test == lbl]
        if plot_3d:
            ax1_mis.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.6, s=10, label=f"class {lbl}"
            )
        else:
            ax1_mis.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10, label=f"class {lbl}")
    ax1_mis.set_title(f"{mismatch_type} posterior only (mismatch)")
    ax1_mis.legend(loc="best")

    # Panel 2: posterior + overlay thumbnails
    # for lbl in np.unique(Y_test):
    #     pts = X_proj_post[Y_test == lbl]
    #     if plot_3d:
    #         ax2.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.6, s=10)
    # Panel 2: posterior + overlay thumbnails using AnnotationBbox
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # -> Standard visualization
    ax2.set_title(f"{encoder_name} posterior + thumbnails")
    # plot posterior colored by class
    for lbl in np.unique(Y_test):
        pts = X_proj_post[Y_test == lbl]
        ax2.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)  # , label=f'class {lbl}')
    # overlay thumbnails every 50 points
    if X_test is not None and X_test.ndim == 4:
        for i in range(0, N_test, 50):
            x0, y0 = X_proj_post[i]
            img = np.transpose(X_test[i], (1, 2, 0))
            im = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
            ax2.add_artist(ab)
    ax2.legend(loc="best")
    # -> Mismatch visualization
    ax2_mis.set_title(f"{mismatch_type} posterior + thumbnails (mismatch)")
    # plot posterior colored by class
    for lbl in np.unique(Y_test):
        pts = X_proj_post_mismatch[Y_test == lbl]
        ax2_mis.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)  # , label=f'class {lbl}')
    # overlay thumbnails every 50 points
    if X_test is not None and X_test.ndim == 4:
        for i in range(0, N_test, 50):
            x0, y0 = X_proj_post_mismatch[i]
            img = np.transpose(X_test[i], (1, 2, 0))
            im = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
            ax2_mis.add_artist(ab)
    ax2_mis.legend(loc="best")

    # Panel 3: posterior + non-learnable prior
    # -> Standard visualization
    ax3.set_title(f"{encoder_name} posterior + non-learnable prior")
    for lbl in np.unique(labels_all_non):
        pts = X_proj_non[labels_all_non == lbl]
        if lbl < 0:
            # priors as black 'x'
            ax3.scatter(
                pts[:, 0],
                pts[:, 1],
                c="black",
                marker="x",
                s=20,
                alpha=0.8,
                label="prior",
            )
        else:
            ax3.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10, label=f"class {lbl}")
    ax3.legend(loc="best")
    # -> Mismatch visualization
    ax3_mis.set_title(f"{mismatch_type} posterior + non-learnable prior (mismatch)")
    for lbl in np.unique(labels_all_non):
        pts = X_proj_non_mismatch[labels_all_non == lbl]
        if lbl < 0:
            # priors as black 'x'
            ax3_mis.scatter(
                pts[:, 0],
                pts[:, 1],
                c="black",
                marker="x",
                s=20,
                alpha=0.8,
                label="prior",
            )
        else:
            ax3_mis.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10, label=f"class {lbl}")
    ax3_mis.legend(loc="best")

    # Panel 4: posterior + non-learnable prior
    if model.diffusion_loss_weight > 0.0:
        # -> Standard visualization
        ax4.set_title(f"{encoder_name} posterior + diffusion prior")
        for lbl in np.unique(labels_all_learn):
            pts = X_proj_diffusion[labels_all_learn == lbl]
            if lbl < 0:
                ax4.scatter(
                    pts[:, 0], pts[:, 1], c="black", marker="x", s=20, alpha=0.8
                )
            else:
                ax4.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)
        ax4.legend(loc="best")
        # -> Mismatch visualization
        ax4_mis.set_title(f"{mismatch_type} posterior + diffusion prior (mismatch)")
        for lbl in np.unique(labels_all_learn):
            pts = X_proj_diffusion_mismatch[labels_all_learn == lbl]
            if lbl < 0:
                ax4_mis.scatter(
                    pts[:, 0], pts[:, 1], c="black", marker="x", s=20, alpha=0.8
                )
            else:
                ax4_mis.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)
        ax4_mis.legend(loc="best")
    else:
        if condition_type is None or condition_type == "shared":
            if pz_learned is None:
                # ax4 is not available, set it to blank
                # -> Standard visualization
                ax4.set_title(
                    f"{encoder_name} posterior + learnable prior is not available"
                )
                ax4.axis("off")
                # -> Mismatch visualization
                ax4_mis.set_title(
                    f"{mismatch_type} posterior + learnable prior is not available (mismatch)"
                )
                ax4_mis.axis("off")
            else:
                # -> Standard visualization
                ax4.set_title(f"{encoder_name} posterior + learnable prior")
                for lbl in np.unique(labels_all_learn):
                    pts = X_proj_diffusion[labels_all_learn == lbl]
                    if lbl < 0:
                        ax4.scatter(
                            pts[:, 0], pts[:, 1], c="black", marker="x", s=20, alpha=0.8
                        )
                    else:
                        ax4.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)
                ax4.legend(loc="best")
                # -> Mismatch visualization
                ax4_mis.set_title(f"{mismatch_type} posterior + learnable prior (mismatch)")
                for lbl in np.unique(labels_all_learn):
                    pts = X_proj_diffusion_mismatch[labels_all_learn == lbl]
                    if lbl < 0:
                        ax4_mis.scatter(
                            pts[:, 0],
                            pts[:, 1],
                            c="black",
                            marker="x",
                            s=20,
                            alpha=0.8,
                        )
                    else:
                        ax4_mis.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)
                ax4_mis.legend(loc="best")
        elif condition_type == "private":
            # -> Standard visualization
            ax4.set_title(f"{encoder_name} posterior + learnable aux prior")
            for lbl in np.unique(labels_all_learn):
                pts = X_proj_diffusion[labels_all_learn == lbl]
                if lbl < 0:
                    ax4.scatter(
                        pts[:, 0], pts[:, 1], c="black", marker="x", s=20, alpha=0.8
                    )
                else:
                    ax4.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)
            ax4.legend(loc="best")
            # -> Mismatch visualization
            ax4_mis.set_title(f"{mismatch_type} posterior + learnable aux prior (mismatch)")
            for lbl in np.unique(labels_all_learn):
                pts = X_proj_diffusion_mismatch[labels_all_learn == lbl]
                if lbl < 0:
                    ax4_mis.scatter(
                        pts[:, 0], pts[:, 1], c="black", marker="x", s=20, alpha=0.8
                    )
                else:
                    ax4_mis.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=10)
            ax4_mis.legend(loc="best")

    plt.tight_layout()

    del embedded_test_set, original_test_set

    del FX_test, FX_test_W, FX_test_Z, Y_test, X_test, latent_space_test, latent_space_test_mismatch
    del X_all_non, X_all_diffusion, X_all_non_mismatch, X_all_diffusion_mismatch
    del labels_all_learn, labels_all_non


    del X_proj_non, X_proj_diffusion, X_proj_non_mismatch, X_proj_diffusion_mismatch, X_proj_post, X_proj_post_mismatch




    gc.collect()
    torch.cuda.empty_cache()



    # # 9) Save
    # # if save_file:
    # #     os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # #     fig.savefig(save_file)
    return fig, fig_mis

