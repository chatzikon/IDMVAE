"""CelebA-mask IDMVAE cross-modal eval grids (moved from models.idmvae_CelebAMask)."""
import torch
from torchvision.utils import make_grid

from eval_functions import idmvae_self_and_cross_modal_generation_eval
from utils import CrossModalEvalForwardMode


def celeba_self_and_cross_modal_generation_eval(
    model,
    data,
    num=8,
    N=8,
    *,
    mode=None,
    condition_type=None,
    prune_cols=None,
    prune_rows=None,
    return_denoised=False,
    data_ctrl=None,
    K=1,
):
    """
    CelebA extended posterior grids when ``mode`` is ``POSTERIOR_CTRL``; otherwise defers to ``IDMVAE``.
    """
    if mode != CrossModalEvalForwardMode.POSTERIOR_CTRL:
        return idmvae_self_and_cross_modal_generation_eval(
            model,
            data,
            mode=mode,
            condition_type=condition_type,
            return_denoised=return_denoised,
            data_ctrl=data_ctrl,
            K=K,
        )
    num_modalities = len(data)
    recon_triess = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
    recon_triess_denoised = (
        [[[] for _ in range(num_modalities)] for _ in range(num_modalities)]
        if model.enable_denoiser_outputs
        else None
    )

    control_inputs = [[] for _ in range(num_modalities)]

    for i in range(N):
        data_ctrl_local = []
        for d in data:
            sample = d[i]
            repeat_shape = [num] + [1] * sample.dim()
            data_ctrl_local.append(sample.unsqueeze(0).repeat(*repeat_shape))

        recons_call = idmvae_self_and_cross_modal_generation_eval(
            model,
            [d[:num] for d in data],
            mode=CrossModalEvalForwardMode.POSTERIOR_CTRL,
            data_ctrl=data_ctrl_local,
            condition_type=condition_type,
            return_denoised=model.enable_denoiser_outputs,
        )

        if model.enable_denoiser_outputs:
            recons_mat, recons_mat_denoised = recons_call
        else:
            recons_mat = recons_call
            recons_mat_denoised = None

        for r in range(num_modalities):
            control_inputs[r].append(data[r][i].cpu())

        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                if condition_type == "private" and r != o:
                    continue
                recon = recon.cpu()
                if recon.dim() == 5:
                    recon = recon.squeeze(0)
                elif recon.dim() == 3:
                    recon = recon.squeeze(0)
                recon_triess[r][o].append(recon)
                if recons_mat_denoised is not None:
                    den_entry = recons_mat_denoised[r][o]
                    if den_entry is not None:
                        den_recon = den_entry.cpu()
                        if den_recon.dim() == 5:
                            den_recon = den_recon.squeeze(0)
                        elif den_recon.dim() == 3:
                            den_recon = den_recon.squeeze(0)
                        recon_triess_denoised[r][o].append(den_recon)

    def to_3ch(t):
        if t.dim() == 4:
            if t.size(1) == 1:
                return t.repeat(1, 3, 1, 1)
        elif t.dim() == 3:
            if t.size(0) == 1:
                return t.repeat(3, 1, 1)
        return t

    outputss = [[None for i in range(num_modalities)] for j in range(num_modalities)]
    outputss_extended = [[None for i in range(num_modalities)] for j in range(num_modalities)]
    outputss_extended_pruned = [[None for i in range(num_modalities)] for j in range(num_modalities)]
    outputss_denoised = (
        [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
        if recon_triess_denoised
        else None
    )
    outputss_extended_denoised = (
        [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
        if recon_triess_denoised
        else None
    )
    outputss_extended_pruned_denoised = (
        [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
        if recon_triess_denoised
        else None
    )

    visualizable_modalities = [0, 1]

    for r in range(num_modalities):
        if r not in visualizable_modalities:
            continue

        input_data = data[r][:num].cpu()
        for o in range(num_modalities):
            if o not in visualizable_modalities:
                continue

            if not recon_triess[r][o]:
                continue

            gen_images = torch.cat(recon_triess[r][o], dim=0)

            curr_input = to_3ch(input_data)
            curr_gen = to_3ch(gen_images)

            all_images = torch.cat([curr_input, curr_gen], dim=0)

            outputss[r][o] = make_grid(all_images, nrow=num)

            placeholder = torch.full_like(curr_input[0], 0.5)
            all_images_ext = [placeholder] + list(curr_input)
            ctrl_rows = control_inputs[r]
            for ii in range(N):
                if ii < len(ctrl_rows):
                    all_images_ext.append(to_3ch(ctrl_rows[ii]))
                else:
                    all_images_ext.append(placeholder)

                if ii < len(recon_triess[r][o]):
                    all_images_ext.extend(list(to_3ch(recon_triess[r][o][ii])))
                else:
                    all_images_ext.extend([placeholder] * num)

            outputss_extended[r][o] = make_grid(all_images_ext, nrow=num + 1)

            if prune_cols is not None or prune_rows is not None:
                keep_cols = [0] + [
                    c + 1 for c in range(num) if (prune_cols is None or c not in prune_cols)
                ]
                keep_rows = [0] + [
                    r + 1 for r in range(N) if (prune_rows is None or r not in prune_rows)
                ]

                all_images_ext_pruned = []

                stride = num + 1

                for row_idx in keep_rows:
                    start_idx = row_idx * stride
                    for col_idx in keep_cols:
                        all_images_ext_pruned.append(all_images_ext[start_idx + col_idx])

                outputss_extended_pruned[r][o] = make_grid(
                    all_images_ext_pruned, nrow=len(keep_cols)
                )

    if recon_triess_denoised is not None and outputss_denoised is not None:
        for r in range(num_modalities):
            if r not in visualizable_modalities:
                continue
            input_data_den = data[r][:num].cpu()
            for o in range(num_modalities):
                if o not in visualizable_modalities:
                    continue
                if not recon_triess_denoised[r][o]:
                    continue
                gen_images_den = torch.cat(recon_triess_denoised[r][o], dim=0)
                curr_input = to_3ch(input_data_den)
                curr_gen_den = to_3ch(gen_images_den)
                all_images_den = torch.cat([curr_input, curr_gen_den], dim=0)
                outputss_denoised[r][o] = make_grid(all_images_den, nrow=num)

                placeholder = torch.full_like(curr_input[0], 0.5)
                all_images_ext_den = [placeholder] + list(curr_input)
                ctrl_rows = control_inputs[r]
                for ii in range(N):
                    if ii < len(ctrl_rows):
                        all_images_ext_den.append(to_3ch(ctrl_rows[ii]))
                    else:
                        all_images_ext_den.append(placeholder)
                    if ii < len(recon_triess_denoised[r][o]):
                        all_images_ext_den.extend(list(to_3ch(recon_triess_denoised[r][o][ii])))
                    else:
                        all_images_ext_den.extend([placeholder] * num)
                outputss_extended_denoised[r][o] = make_grid(
                    all_images_ext_den, nrow=num + 1
                )

                if prune_cols is not None or prune_rows is not None:
                    keep_cols = [0] + [
                        c + 1 for c in range(num) if (prune_cols is None or c not in prune_cols)
                    ]
                    keep_rows = [0] + [
                        r_ + 1 for r_ in range(N) if (prune_rows is None or r_ not in prune_rows)
                    ]
                    all_images_ext_pruned_den = []
                    stride = num + 1
                    for row_idx in keep_rows:
                        start_idx = row_idx * stride
                        for col_idx in keep_cols:
                            all_images_ext_pruned_den.append(
                                all_images_ext_den[start_idx + col_idx]
                            )
                    outputss_extended_pruned_denoised[r][o] = make_grid(
                        all_images_ext_pruned_den, nrow=len(keep_cols)
                    )

    if model.enable_denoiser_outputs:
        model.last_denoised_posterior_grids = outputss_denoised
        model.last_denoised_posterior_extended_grids = outputss_extended_denoised
        model.last_denoised_posterior_extended_pruned_grids = outputss_extended_pruned_denoised
    else:
        model.last_denoised_posterior_grids = None
        model.last_denoised_posterior_extended_grids = None
        model.last_denoised_posterior_extended_pruned_grids = None

    return outputss, outputss_extended, outputss_extended_pruned, None
