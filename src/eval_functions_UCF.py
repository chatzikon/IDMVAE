import os
import shutil

import numpy as np
import wandb
import glob
from statistics import mean
from collections import defaultdict
from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance
from sklearn.linear_model import LogisticRegression
import pandas as pd
import torch
from torchvision.utils import save_image, make_grid

from eval_functions import (
    classify_linear_latent_representations,
    idmvae_generate_unconditional,
    idmvae_self_and_cross_modal_generation_eval,
)
from utils import CrossModalEvalForwardMode
from utils import unpack_data_CUBcluster8
from utils import plot_text_as_image_tensor
from dataset_UCF import load_cub_i2w


def _plot_sentences_as_tensor(model, batched_text_modality, i2w=None):
    if i2w is None:
        i2w = load_cub_i2w(
            model.params.datadir,
            getattr(model.params, "cub_vocab_file", None),
            getattr(model.params, "cub_vocab_min_occ", 1),
        )
    sentences_processed = _sent_process(model, batched_text_modality.argmax(-1))
    sentences_worded = [
        " ".join(
            i2w[str(word)] for word in sent
            if i2w[str(word)] != "<pad>"
        )
        for sent in sentences_processed
    ]
    return plot_text_as_image_tensor(
        sentences_worded,
        pixel_width=model.img_size_original,
        pixel_height=model.img_size_original * model.text2img_ratio,
        fontsize=model.fontsize,
    )


def _sent_process(model, sentences):
    return [model.vaes[1].fn_trun(model.vaes[1].fn_2i(s)) for s in sentences]


def train_clf_lr_CUB_multi_labelTypes(model, dl, device, args, condition_type=None):
    """Train linear classifiers on CUB latents for shared/private targets."""
    latent_rep = {
        "m0": {"us": [], "zs": [], "ws": []},
        "m1": {"us": [], "zs": [], "ws": []},
    }
    labels_all = []
    for dataT_lr in dl:
        data, labels_batch = unpack_data_CUBcluster8(dataT_lr, device=device)
        mask = None

        if condition_type is None:
            labels = labels_batch.cpu().data.numpy()
        elif condition_type == "shared":
            labels = labels_batch[0].cpu().data.numpy()
        elif condition_type == "private":
            color_labels_tensor = labels_batch[1]
            mask = (color_labels_tensor == 0) | (color_labels_tensor == 1)
            labels = color_labels_tensor[mask].cpu().data.numpy()

        labels_all.append(labels)
        for v, vae in enumerate(model.vaes):
            with torch.no_grad():
                if args.use_mean_for_latent_clf:
                    mu_v, _ = vae.enc(data[v]); us_v = mu_v
                else:
                    us_v = vae.qu_x(*vae.enc(data[v])).rsample()
                ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)

                if mask is not None:
                    us_v_filtered = us_v[mask]
                    zs_v_filtered = zs_v[mask]
                    ws_v_filtered = ws_v[mask]
                else:
                    us_v_filtered = us_v
                    zs_v_filtered = zs_v
                    ws_v_filtered = ws_v

                latent_rep[f"m{v}"]["us"].append(us_v_filtered.cpu().data.numpy())
                latent_rep[f"m{v}"]["zs"].append(zs_v_filtered.cpu().data.numpy())
                latent_rep[f"m{v}"]["ws"].append(ws_v_filtered.cpu().data.numpy())

    gt = np.concatenate(labels_all, axis=0)
    if gt.shape[0] == 0:
        print("Warning: No valid samples found for training the classifier after filtering. Returning None.")
        return None

    clf_lr = {}
    for v, _ in enumerate(model.vaes):
        latent_rep_u = np.concatenate(latent_rep[f"m{v}"]["us"], axis=0)
        latent_rep_w = np.concatenate(latent_rep[f"m{v}"]["ws"], axis=0)
        latent_rep_z = np.concatenate(latent_rep[f"m{v}"]["zs"], axis=0)
        clf_lr_rep_u = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000)
        clf_lr_rep_z = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000)
        clf_lr_rep_w = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000)
        clf_lr_rep_u.fit(latent_rep_u, gt.ravel()); clf_lr[f"m{v}_u"] = clf_lr_rep_u
        clf_lr_rep_w.fit(latent_rep_w, gt.ravel()); clf_lr[f"m{v}_w"] = clf_lr_rep_w
        clf_lr_rep_z.fit(latent_rep_z, gt.ravel()); clf_lr[f"m{v}_z"] = clf_lr_rep_z
    return clf_lr


def linear_latent_classification_CUB_multi_labelTypes(model, test_loader, clf_lr, device, args, condition_type=None):
    """Evaluate CUB latent classifiers and keep valid/ambiguous splits."""
    model.eval()
    lr_acc_all_z = []
    lr_acc_m0_z, lr_acc_m1_z = [], []
    lr_acc_m0_w, lr_acc_m1_w = [], []
    lr_acc_m0_u, lr_acc_m1_u = [], []
    collected_data = defaultdict(list)

    with torch.no_grad():
        for dataT in test_loader:
            data, targets = unpack_data_CUBcluster8(dataT, device)
            cluster_lbls, color_lbls, category_lbls, img_ids, dataset_indices = targets
            base_validity_mask = (
                (color_lbls == 0) | (color_lbls == 1)
                if condition_type == "private"
                else torch.ones_like(img_ids, dtype=torch.bool)
            )
            manual_ambiguous_mask = torch.zeros_like(
                base_validity_mask,
                dtype=torch.bool,
                device=device,
            )
            valid_mask = base_validity_mask & ~manual_ambiguous_mask
            ambiguous_mask = base_validity_mask & manual_ambiguous_mask

            for is_ambiguous, mask in [(False, valid_mask), (True, ambiguous_mask)]:
                if not mask.any():
                    continue
                filtered_data = [d[mask] for d in data]
                labels_batch_for_clf = color_lbls[mask] if condition_type == "private" else cluster_lbls[mask]
                if clf_lr is None:
                    continue
                latent_reps = []
                for v, vae in enumerate(model.vaes):
                    if args.use_mean_for_latent_clf:
                        mu_v, _ = vae.enc(filtered_data[v]); us_v = mu_v
                    else:
                        us_v = vae.qu_x(*vae.enc(filtered_data[v])).rsample()
                    ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                    latent_reps.append([us_v.cpu().numpy(), ws_v.cpu().numpy(), zs_v.cpu().numpy()])
                accuracies, predictions = classify_linear_latent_representations(
                    clf_lr, latent_reps, labels_batch_for_clf, split=True
                )

                if not is_ambiguous:
                    lr_acc_m0_u.append(np.mean(accuracies["m0_u"]))
                    lr_acc_m1_u.append(np.mean(accuracies["m1_u"]))
                    lr_acc_m0_w.append(np.mean(accuracies["m0_w"]))
                    lr_acc_m1_w.append(np.mean(accuracies["m1_w"]))
                    lr_acc_m0_z.append(np.mean(accuracies["m0_z"]))
                    lr_acc_m1_z.append(np.mean(accuracies["m1_z"]))
                    lr_acc_all_z.append(np.mean(accuracies["all_z"]))

                collected_data["images"].append(filtered_data[0].cpu())
                collected_data["is_ambiguous"].append(np.full(len(labels_batch_for_clf), is_ambiguous))
                collected_data["gts"].append(predictions["ground_truths"])
                collected_data["dataset_indices"].append(dataset_indices[mask].cpu().numpy())
                collected_data["img_ids"].append(img_ids[mask].cpu().numpy())
                collected_data["category_labels"].append(category_lbls[mask].cpu().numpy())
                collected_data["cluster_labels"].append(cluster_lbls[mask].cpu().numpy())
                collected_data["color_labels"].append(color_lbls[mask].cpu().numpy())
                for key, value in predictions.items():
                    if key != "ground_truths":
                        collected_data[f"preds_{key}"].append(value)

    if not lr_acc_m0_u:
        print(f"Warning: No valid samples were processed for latent classification (condition: {condition_type}).")
        return {}

    accuracies_lr = {
        "m0_u": mean(lr_acc_m0_u),
        "m1_u": mean(lr_acc_m1_u),
        "m0_w": mean(lr_acc_m0_w),
        "m1_w": mean(lr_acc_m1_w),
        "m0_z": mean(lr_acc_m0_z),
        "m1_z": mean(lr_acc_m1_z),
        "_mean_u": mean([mean(lr_acc_m0_u), mean(lr_acc_m1_u)]),
        "_mean_w": mean([mean(lr_acc_m0_w), mean(lr_acc_m1_w)]),
        "_mean_z": mean([mean(lr_acc_m0_z), mean(lr_acc_m1_z)]),
        "z_all": mean(lr_acc_all_z),
    }

    if collected_data:
        final_data = {"images": torch.cat(collected_data["images"], dim=0)}
        for key, val_list in collected_data.items():
            if key != "images":
                final_data[key] = np.concatenate(val_list, axis=0)
        final_preds = {}
        for key, val in final_data.items():
            if key.startswith("preds_"):
                final_preds[key.replace("preds_", "")] = val
        final_data["preds"] = final_preds
        accuracies_lr["prediction_data"] = final_data

        df_full = pd.DataFrame({
            "gt": final_data["gts"],
            "img_id": final_data["img_ids"],
            "is_ambiguous": final_data["is_ambiguous"],
            "task_subset_index": np.arange(len(final_data["gts"])),
        })
        df_valid = df_full[~df_full["is_ambiguous"]].copy()
        if condition_type == "shared":
            df_valid["pred"] = final_preds["m0_z"][~final_data["is_ambiguous"]]
            df_valid["confidence"] = final_preds["m0_z_confidence"][~final_data["is_ambiguous"]]
        elif condition_type == "private":
            df_valid["pred"] = final_preds["m0_w"][~final_data["is_ambiguous"]]
            df_valid["confidence"] = final_preds["m0_w_confidence"][~final_data["is_ambiguous"]]

        top_samples = {}
        if condition_type in ["shared", "private"]:
            correct_df = df_valid[df_valid["gt"] == df_valid["pred"]].sort_values("confidence", ascending=False)
            unique_top_df = correct_df.drop_duplicates(subset=["img_id"], keep="first")
            x_count = 50 if condition_type == "shared" else 100
            top_x_per_class = unique_top_df.groupby("gt").head(x_count)
            key_name = "cluster" if condition_type == "shared" else "color"
            top_samples[key_name] = {}
            for _, row in top_x_per_class.iterrows():
                label = int(row["gt"])
                idx = int(row["task_subset_index"])
                if label not in top_samples[key_name]:
                    top_samples[key_name][label] = []
                sample_data = {
                    "image": final_data["images"][idx],
                    "gt": final_data["gts"][idx],
                    "dataset_index": final_data["dataset_indices"][idx],
                    "img_id": final_data["img_ids"][idx],
                    "cluster_label": final_data["cluster_labels"][idx],
                    "category_label": final_data["category_labels"][idx],
                    "color_label": final_data["color_labels"][idx],
                    "pred_img_w": final_data["preds"]["m0_w"][idx],
                    "conf_img_w": final_data["preds"]["m0_w_confidence"][idx],
                    "pred_img_z": final_data["preds"]["m0_z"][idx],
                    "conf_img_z": final_data["preds"]["m0_z_confidence"][idx],
                }
                if condition_type == "shared":
                    sample_data["pred_txt_z"] = final_data["preds"]["m1_z"][idx]
                    sample_data["conf_txt_z"] = final_data["preds"]["m1_z_confidence"][idx]
                    sample_data["pred_txt_w"] = final_data["preds"]["m1_w"][idx]
                    sample_data["conf_txt_w"] = final_data["preds"]["m1_w_confidence"][idx]
                top_samples[key_name][label].append(sample_data)
        accuracies_lr["top_confidence_samples"] = top_samples

    return accuracies_lr


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_images_with_labels(imgs, labels_tuple, save_dir, prefix="orig"):
    if imgs is None:
        return
    _ensure_dir(save_dir)
    lbl_cluster, lbl_color, lbl_category, img_id, dataset_index = labels_tuple
    for i in range(imgs.size(0)):
        fname = (
            f"{prefix}_clu_{int(lbl_cluster[i])}"
            f"_cat_{int(lbl_category[i])}"
            f"_dir_{int(lbl_color[i])}"
            f"_id_{int(img_id[i])}"
            f"_idx_{int(dataset_index[i])}.png"
        )
        save_image(imgs[i].cpu(), os.path.join(save_dir, fname))


def save_recon_sequences_with_labels(recon_sequences, labels_tuple, save_dir, prefix="gen"):
    if recon_sequences is None:
        return
    for gen_idx, tensor in enumerate(recon_sequences):
        if tensor is None:
            continue
        gen_dir = os.path.join(save_dir, f"{prefix}_g{gen_idx}")
        save_images_with_labels(tensor, labels_tuple, gen_dir, prefix=f"{prefix}_g{gen_idx}")

def calculate_inception_features_for_gen_evaluation(inception_state_dict_path, device, dir_fid_base, datadir, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=inception_state_dict_path)
    model = model.to(device)

    # for moddality_num in range(0):
    moddality_num = 0
    moddality = 'm{}'.format(moddality_num)
    filename_act_real_calc = os.path.join(dir_fid_base, 'test','real_activations_{}.npy'.format(moddality))
    if not os.path.exists(filename_act_real_calc):
        files_real_calc = glob.glob(os.path.join(dir_fid_base, 'test', moddality, '*' + '.png'))
        act_real_calc = get_activations(files_real_calc, model, device, batch_size, dims, verbose=False)
        np.save(filename_act_real_calc, act_real_calc)

    for prefix  in ['random', 'm0', 'm1']:
        dir_gen = os.path.join(dir_fid_base, prefix)
        if not os.path.exists(dir_gen):
            raise RuntimeError('Invalid path: %s' % dir_gen)
        # for modality in ['m{}'.format(m) for m in range(5)]:
        modality = 'm{}'.format(0)
        files_gen = glob.glob(os.path.join(dir_gen, modality, '*' + '.png'))
        filename_act = os.path.join(dir_gen,
                                       modality + '_activations.npy')
        act_rand_gen = get_activations(files_gen, model, device, batch_size, dims, verbose=False)
        np.save(filename_act, act_rand_gen)

def calculate_fid(feats_real, feats_gen):
    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid;

def calculate_fid_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_fid(feats_real, feats_gen);
    return dict_fid;

def get_clf_activations(flags, data, model):
    model.eval();
    act = model.get_activations(data);
    act = act.cpu().data.numpy().reshape(flags.batch_size, -1)
    return act


# --- CUB IDMVAE: unconditional sampling, cross-modal grids, FID I/O ---


def cub_self_and_cross_modal_generation_for_fid_calculation(model, data, savePath, i):
    recons_mat = idmvae_self_and_cross_modal_generation_eval(model, [d for d in data])
    for r, recons_list in enumerate(recons_mat):
        for o, recon in enumerate(recons_list):
            if o == 0:
                recon = recon.squeeze(0).cpu()
                for image in range(recon.size(0)):
                    save_image(
                        recon[image, :, :, :],
                        "{}/m{}/m{}/{}_{}.png".format(savePath, r, o, image, i),
                    )


def cub_save_test_samples_for_fid_calculation(model, data, savePath, i):
    o = 0
    imgs = data[0].cpu()
    for image in range(imgs.size(0)):
        save_image(
            imgs[image, :, :, :],
            "{}/test/m{}/{}_{}.png".format(savePath, o, image, i),
        )


def cub_generate_unconditional(
    model,
    N=100,
    coherence_calculation=False,
    fid_calculation=False,
    savePath=None,
    tranche=None,
):
    outputs = []
    samples_list = idmvae_generate_unconditional(model, N)
    if coherence_calculation:
        return [samples.data.cpu() for samples in samples_list]
    if fid_calculation:
        for i, samples in enumerate(samples_list):
            if i == 0:
                samples = samples.data.cpu()
                for image in range(samples.size(0)):
                    save_image(
                        samples[image, :, :, :],
                        "{}/random/m{}/{}_{}.png".format(savePath, i, tranche, image),
                    )
            else:
                continue
    else:
        for i, samples in enumerate(samples_list):
            if i == 0:
                samples = samples.data.cpu()
                samples = samples.view(samples.size()[0], *samples.size()[1:])
                outputs.append(make_grid(samples, nrow=int(np.sqrt(N))))
            else:
                samples = samples.data.cpu()
                samples = samples.view(samples.size()[0], *samples.size()[1:])
                outputs.append(
                    make_grid(_plot_sentences_as_tensor(model, samples), nrow=int(np.sqrt(N)))
                )

    return outputs


def calculate_fid_routine(
    datadirCUB,
    fid_path,
    num_fid_samples,
    epoch,
    model,
    test_time_loader,
    device,
    inception_path,
):
    """Compute FID for unconditional and conditional CUB generations; log to wandb."""
    total_cond = 0
    for j in [0]:
        if os.path.exists(os.path.join(fid_path, "test", "m{}".format(j))):
            shutil.rmtree(os.path.join(fid_path, "test", "m{}".format(j)))
            os.makedirs(os.path.join(fid_path, "test", "m{}".format(j)))
        else:
            os.makedirs(os.path.join(fid_path, "test", "m{}".format(j)))
        if os.path.exists(os.path.join(fid_path, "random", "m{}".format(j))):
            shutil.rmtree(os.path.join(fid_path, "random", "m{}".format(j)))
            os.makedirs(os.path.join(fid_path, "random", "m{}".format(j)))
        else:
            os.makedirs(os.path.join(fid_path, "random", "m{}".format(j)))
        for i in [0, 1]:
            if os.path.exists(os.path.join(fid_path, "m{}".format(i), "m{}".format(j))):
                shutil.rmtree(os.path.join(fid_path, "m{}".format(i), "m{}".format(j)))
                os.makedirs(os.path.join(fid_path, "m{}".format(i), "m{}".format(j)))
            else:
                os.makedirs(os.path.join(fid_path, "m{}".format(i), "m{}".format(j)))
    with torch.no_grad():
        for tranche in range(num_fid_samples // 100):
            kwargs_uncond = {"savePath": fid_path, "tranche": tranche}
            cub_generate_unconditional(
                model,
                N=100,
                coherence_calculation=False,
                fid_calculation=True,
                **kwargs_uncond,
            )
        for i, dataT in enumerate(test_time_loader):
            data, _ = unpack_data_CUBcluster8(dataT, device=device)

            if total_cond < num_fid_samples:
                cub_self_and_cross_modal_generation_for_fid_calculation(model, data, fid_path, i)
                cub_save_test_samples_for_fid_calculation(model, data, fid_path, i)
                total_cond += data[0].size(0)
        calculate_inception_features_for_gen_evaluation(
            inception_path, device, fid_path, datadirCUB
        )
        modality_target = "m{}".format(0)
        file_activations_real = os.path.join(
            fid_path, "test", "real_activations_{}.npy".format(modality_target)
        )
        feats_real = np.load(file_activations_real)
        file_activations_randgen = os.path.join(
            fid_path, "random", modality_target + "_activations.npy"
        )
        feats_randgen = np.load(file_activations_randgen)
        fid_randval = calculate_fid(feats_real, feats_randgen)
        wandb.log({"FID/Random/{}".format(modality_target): fid_randval}, step=epoch)
        fid_condgen_target_list = []
        for modality_source in ["m{}".format(m) for m in [0, 1]]:
            file_activations_gen = os.path.join(
                fid_path, modality_source, modality_target + "_activations.npy"
            )
            feats_gen = np.load(file_activations_gen)
            fid_val = calculate_fid(feats_real, feats_gen)
            wandb.log(
                {"FID/{}/{}".format(modality_source, modality_target): fid_val}, step=epoch
            )
            fid_condgen_target_list.append(fid_val)
    if os.path.exists(fid_path):
        shutil.rmtree(fid_path)
        os.makedirs(fid_path)


def cub_self_and_cross_modal_generation_eval(
    model,
    data,
    num=10,
    N=10,
    *,
    mode=None,
    condition_type=None,
    return_denoised=False,
    data_ctrl=None,
    K=1,
):
    """
    CUB-specific multi-row cross-modal grids. ``mode`` selects the base recon path
    (see ``CrossModalEvalForwardMode``); ``None`` matches val-style ``forward``.
    """
    if mode is None:
        num_modalities = len(data)
        recon_triess = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        outputss = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        for i in range(N):
            recons_mat = idmvae_self_and_cross_modal_generation_eval(
                model, [d[:num] for d in data]
            )
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.squeeze(0).cpu()
                    if o == 0:
                        recon_triess[r][o].append(recon)
                    else:
                        if i < 3:
                            recon_triess[r][o].append(_plot_sentences_as_tensor(model, recon))
        for r, recons_list in enumerate(recons_mat):
            if r == 1:
                input_data = _plot_sentences_as_tensor(model, data[r][:num]).cpu()
            else:
                input_data = data[r][:num]
                if model.params.use_pretrain_feats:
                    vae_device = next(model.pretrained_vae.parameters()).device
                    input_data = model.pretrained_vae.decode(
                        (input_data / 0.18215).to(vae_device)
                    ).sample
                    input_data = input_data.add(1).div(2).clamp(0, 1)
                input_data = input_data.cpu()
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([input_data] + recon_triess[r][o], dim=2), nrow=num
                )
        return outputss

    if mode == CrossModalEvalForwardMode.PRIOR_CTRL:
        num_modalities = len(data)
        recon_triess = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        recon_triess_to_table = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        recon_triess_denoised = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        outputss = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        outputss_denoised = (
            [[[] for i in range(num_modalities)] for j in range(num_modalities)]
            if model.enable_denoiser_outputs
            else None
        )
        for i in range(N):
            recons_call = idmvae_self_and_cross_modal_generation_eval(
                model,
                [d[:num] for d in data],
                mode=CrossModalEvalForwardMode.PRIOR_CTRL,
                condition_type=condition_type,
                return_denoised=model.enable_denoiser_outputs,
            )
            if model.enable_denoiser_outputs:
                recons_mat, recons_mat_denoised = recons_call
            else:
                recons_mat = recons_call
                recons_mat_denoised = None
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.squeeze(0).cpu()
                    recon_triess_to_table[r][o].append(recon)
                    if condition_type == "shared":
                        if o == 0:
                            recon_triess[r][o].append(recon)
                        else:
                            if i < 3:
                                recon_triess[r][o].append(_plot_sentences_as_tensor(model, recon))
                    elif condition_type == "private":
                        if r == 0 and o == 0:
                            recon_triess[r][o].append(recon)
                        elif r == 1 and o == 1:
                            if i < 3:
                                recon_triess[r][o].append(_plot_sentences_as_tensor(model, recon))
                    if model.enable_denoiser_outputs and recons_mat_denoised is not None:
                        den_entry = recons_mat_denoised[r][o]
                        if den_entry is not None:
                            recon_den = den_entry.squeeze(0).cpu()
                            if condition_type == "shared":
                                if o == 0:
                                    recon_triess_denoised[r][o].append(recon_den)
                                elif i < 3:
                                    recon_triess_denoised[r][o].append(
                                        _plot_sentences_as_tensor(model, recon_den)
                                    )
                            elif condition_type == "private":
                                if (r == 0 and o == 0) or (r == 1 and o == 1 and i < 3):
                                    target = (
                                        recon_den
                                        if o == 0
                                        else _plot_sentences_as_tensor(model, recon_den)
                                    )
                                    recon_triess_denoised[r][o].append(target)
        for r, recons_list in enumerate(recons_mat):
            if r == 1:
                input_data = _plot_sentences_as_tensor(model, data[r][:num]).cpu()
            else:
                input_data = data[r][:num]
                if model.params.use_pretrain_feats:
                    vae_device = next(model.pretrained_vae.parameters()).device
                    input_data = model.pretrained_vae.decode(
                        (input_data / 0.18215).to(vae_device)
                    ).sample
                    input_data = input_data.add(1).div(2).clamp(0, 1)
                input_data = input_data.cpu()
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([input_data] + recon_triess[r][o], dim=2), nrow=num
                )
                if model.enable_denoiser_outputs and recon_triess_denoised[r][o]:
                    outputss_denoised[r][o] = make_grid(
                        torch.cat([input_data] + recon_triess_denoised[r][o], dim=2), nrow=num
                    )
                elif model.enable_denoiser_outputs:
                    outputss_denoised[r][o] = None
        if model.enable_denoiser_outputs:
            model.last_denoised_prior_grids = outputss_denoised
            model.last_denoised_prior_entries = recon_triess_denoised
        else:
            model.last_denoised_prior_grids = None
            model.last_denoised_prior_entries = None
        return outputss, input_data, recon_triess_to_table

    if mode == CrossModalEvalForwardMode.POSTERIOR_CTRL:
        num_modalities = len(data)
        recon_triess = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        recon_triess_denoised = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        outputss = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        outputss_denoised = (
            [[[] for i in range(num_modalities)] for j in range(num_modalities)]
            if model.enable_denoiser_outputs
            else None
        )

        def _prepare_for_display(mod_idx, batch_tensor):
            if mod_idx == 1:
                return _plot_sentences_as_tensor(model, batch_tensor).cpu()
            processed = batch_tensor
            if mod_idx == 0 and model.params.use_pretrain_feats:
                vae_device = next(model.pretrained_vae.parameters()).device
                processed = model.pretrained_vae.decode(
                    (processed / 0.18215).to(vae_device)
                ).sample
                processed = processed.add(1).div(2).clamp(0, 1)
            return processed.cpu()

        prepared_inputs = []
        control_inputs = [[] for _ in range(num_modalities)]
        for i in range(N):
            data_ctrl = []
            for d in data:
                sample = d[i]
                repeat_shape = [num] + [1] * sample.dim()
                data_ctrl.append(sample.unsqueeze(0).repeat(*repeat_shape))
            recons_call = idmvae_self_and_cross_modal_generation_eval(
                model,
                [d[:num] for d in data],
                mode=CrossModalEvalForwardMode.POSTERIOR_CTRL,
                data_ctrl=data_ctrl,
                condition_type=condition_type,
                return_denoised=model.enable_denoiser_outputs,
            )
            if model.enable_denoiser_outputs:
                recons_mat, recons_mat_denoised = recons_call
            else:
                recons_mat = recons_call
                recons_mat_denoised = None
            for r in range(num_modalities):
                ctrl_sample = data[r][i].unsqueeze(0)
                control_inputs[r].append(_prepare_for_display(r, ctrl_sample).squeeze(0))
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    if condition_type == "private" and r != o:
                        continue
                    recon = recon.squeeze(0).cpu()
                    if o == 0:
                        recon_triess[r][o].append(recon)
                    else:
                        if i < 3:
                            recon_triess[r][o].append(_plot_sentences_as_tensor(model, recon))
                    if model.enable_denoiser_outputs and recons_mat_denoised is not None:
                        den_entry = recons_mat_denoised[r][o]
                        if den_entry is not None:
                            recon_den = den_entry.squeeze(0).cpu()
                            if o == 0:
                                recon_triess_denoised[r][o].append(recon_den)
                            elif i < 3:
                                recon_triess_denoised[r][o].append(
                                    _plot_sentences_as_tensor(model, recon_den)
                                )
        for r in range(num_modalities):
            prepared_inputs.append(_prepare_for_display(r, data[r][:num]))
        for r, recons_list in enumerate(recons_mat):
            input_data = prepared_inputs[r]
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([input_data] + recon_triess[r][o], dim=2), nrow=num
                )
                if model.enable_denoiser_outputs and recon_triess_denoised[r][o]:
                    outputss_denoised[r][o] = make_grid(
                        torch.cat([input_data] + recon_triess_denoised[r][o], dim=2), nrow=num
                    )
                elif model.enable_denoiser_outputs:
                    outputss_denoised[r][o] = None
        outputss_extended = [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
        outputss_extended_denoised = (
            [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
            if model.enable_denoiser_outputs
            else None
        )
        for r in range(num_modalities):
            if r != 0:
                continue
            input_row = prepared_inputs[r]
            placeholder = torch.full_like(input_row[0], 0.5)
            input_row_list = list(input_row)
            ctrl_rows = control_inputs[r]
            rows_to_show = min(N, len(ctrl_rows))
            for o in range(num_modalities):
                if o != 0:
                    continue
                generated_imgs_list = recon_triess[r][o]
                all_images_for_grid = [placeholder] + input_row_list
                for i in range(rows_to_show):
                    all_images_for_grid.append(ctrl_rows[i])
                    if i < len(generated_imgs_list):
                        all_images_for_grid.extend(list(generated_imgs_list[i]))
                    else:
                        all_images_for_grid.extend([placeholder] * num)
                outputss_extended[r][o] = make_grid(
                    all_images_for_grid, nrow=num + 1, padding=2, pad_value=0.0
                )
                if model.enable_denoiser_outputs and recon_triess_denoised[r][o]:
                    den_all_images_for_grid = [placeholder] + input_row_list
                    for i in range(rows_to_show):
                        den_all_images_for_grid.append(ctrl_rows[i])
                        if i < len(recon_triess_denoised[r][o]):
                            den_all_images_for_grid.extend(list(recon_triess_denoised[r][o][i]))
                        else:
                            den_all_images_for_grid.extend([placeholder] * num)
                    outputss_extended_denoised[r][o] = make_grid(
                        den_all_images_for_grid, nrow=num + 1, padding=2, pad_value=0.0
                    )
        if model.enable_denoiser_outputs:
            model.last_denoised_posterior_grids = outputss_denoised
            model.last_denoised_posterior_entries = recon_triess_denoised
            model.last_denoised_posterior_extended_grids = outputss_extended_denoised
        else:
            model.last_denoised_posterior_grids = None
            model.last_denoised_posterior_entries = None
            model.last_denoised_posterior_extended_grids = None
        return outputss, outputss_extended, outputss_extended_denoised

    if mode == CrossModalEvalForwardMode.POSTERIOR_NONSHUF:
        num_modalities = len(data)
        recon_triess = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        recon_triess_denoised = (
            [[[] for i in range(num_modalities)] for j in range(num_modalities)]
            if model.enable_denoiser_outputs
            else None
        )
        outputss = [[[] for i in range(num_modalities)] for j in range(num_modalities)]
        outputss_denoised = (
            [[[] for i in range(num_modalities)] for j in range(num_modalities)]
            if model.enable_denoiser_outputs
            else None
        )
        for i in range(N):
            recons_call = idmvae_self_and_cross_modal_generation_eval(
                model,
                [d[:num] for d in data],
                mode=CrossModalEvalForwardMode.POSTERIOR_NONSHUF,
                condition_type=condition_type,
                return_denoised=model.enable_denoiser_outputs,
            )
            if model.enable_denoiser_outputs:
                recons_mat, recons_mat_denoised = recons_call
            else:
                recons_mat = recons_call
                recons_mat_denoised = None
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    if condition_type == "private" and r != o:
                        continue
                    recon = recon.squeeze(0).cpu()
                    if o == 0:
                        recon_triess[r][o].append(recon)
                    else:
                        if i < 3:
                            recon_triess[r][o].append(_plot_sentences_as_tensor(model, recon))
                    if model.enable_denoiser_outputs and recons_mat_denoised is not None:
                        den_entry = recons_mat_denoised[r][o]
                        if den_entry is not None:
                            recon_den = den_entry.squeeze(0).cpu()
                            if o == 0:
                                recon_triess_denoised[r][o].append(recon_den)
                            elif i < 3:
                                recon_triess_denoised[r][o].append(
                                    _plot_sentences_as_tensor(model, recon_den)
                                )
        for r, recons_list in enumerate(recons_mat):
            if r == 1:
                input_data = _plot_sentences_as_tensor(model, data[r][:num]).cpu()
            else:
                input_data = data[r][:num]
                if model.params.use_pretrain_feats:
                    vae_device = next(model.pretrained_vae.parameters()).device
                    input_data = model.pretrained_vae.decode(
                        (input_data / 0.18215).to(vae_device)
                    ).sample
                    input_data = input_data.add(1).div(2).clamp(0, 1)
                input_data = input_data.cpu()
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([input_data] + recon_triess[r][o], dim=2), nrow=num
                )
                if model.enable_denoiser_outputs and recon_triess_denoised[r][o]:
                    outputss_denoised[r][o] = make_grid(
                        torch.cat([input_data] + recon_triess_denoised[r][o], dim=2), nrow=num
                    )
                elif model.enable_denoiser_outputs:
                    outputss_denoised[r][o] = None
        if model.enable_denoiser_outputs:
            model.last_denoised_posterior_nonshuf_grids = outputss_denoised
            model.last_denoised_posterior_nonshuf_entries = recon_triess_denoised
        else:
            model.last_denoised_posterior_nonshuf_grids = None
            model.last_denoised_posterior_nonshuf_entries = None
        return outputss

    raise ValueError(
        f"Unsupported mode {mode!r} for cub_self_and_cross_modal_generation_eval"
    )
