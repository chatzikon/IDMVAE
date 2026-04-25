import os
import shutil
import numpy as np
import glob
from statistics import mean
from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
import torch
import torch.nn as nn
import itertools

import matplotlib.pyplot as plt
import wandb

from utils import unpack_data_PM_quadrant as unpack_data_PolyMNIST
from utils import CrossModalEvalForwardMode
from eval_functions import (
    idmvae_generate_unconditional,
    idmvae_self_and_cross_modal_generation_eval,
)
from numpy import sqrt
from torchvision.utils import save_image, make_grid


def cross_coherence(model, test_loader, clfs, device):
    """Compute cross coherence."""
    model.eval()
    corrs = [[0 for _ in model.vaes] for _ in model.vaes]
    total = 0
    with torch.no_grad():
        for dataT in test_loader:
            data, targets = unpack_data_PolyMNIST(dataT, device)
            total += targets.size(0)
            _, px_us, _ = model.self_and_cross_modal_generation_forward(data)
            for idx_srt, _ in enumerate(model.vaes):
                for idx_trg, _ in enumerate(model.vaes):
                    clfs_results = torch.argmax(
                        clfs[idx_trg](px_us[idx_srt][idx_trg].mean.squeeze(0)),
                        dim=-1,
                    )
                    corrs[idx_srt][idx_trg] += (clfs_results == targets).sum().item()

        for idx_trgt, _ in enumerate(model.vaes):
            for idx_strt, _ in enumerate(model.vaes):
                corrs[idx_strt][idx_trgt] = corrs[idx_strt][idx_trgt] / total

        means_target = [0 for _ in model.vaes]
        for idx_target, _ in enumerate(model.vaes):
            means_target[idx_target] = mean(
                [
                    corrs[idx_start][idx_target]
                    for idx_start, _ in enumerate(model.vaes)
                    if idx_start != idx_target
                ]
            )
    return corrs, means_target, mean(means_target)


def self_and_cross_coherence_calculation(
    model,
    test_loader,
    clfs_main,
    clfs_cross,
    device,
    condition_type=None,
    rsample_type="prior",
):
    """Compute quadrant reconstruction cross coherence."""
    model.eval()
    num_modalities = len(model.vaes)
    coherence_main = [[0] * num_modalities for _ in range(num_modalities)]
    coherence_cross = [[0] * num_modalities for _ in range(num_modalities)]
    total = 0
    with torch.no_grad():
        for dataT in test_loader:
            data, targets = unpack_data_PolyMNIST(dataT, device)

            if condition_type is None:
                total += targets[0].size(0)
                main_targets = targets
                if rsample_type == "prior":
                    px_us_mean = idmvae_self_and_cross_modal_generation_eval(model,
                        data, mode=CrossModalEvalForwardMode.PRIOR
                    )
                elif rsample_type == "posterior":
                    px_us_mean = idmvae_self_and_cross_modal_generation_eval(model,
                        data, mode=CrossModalEvalForwardMode.POSTERIOR
                    )
            elif condition_type == "shared":
                total += targets[0][0].size(0)
                main_targets = targets[0]
                cross_targets = targets[1]
                if rsample_type == "prior":
                    px_us_mean = idmvae_self_and_cross_modal_generation_eval(model,
                        data,
                        condition_type="shared",
                        mode=CrossModalEvalForwardMode.PRIOR,
                    )
                elif rsample_type == "posterior":
                    px_us_mean = idmvae_self_and_cross_modal_generation_eval(model,
                        data,
                        condition_type="shared",
                        mode=CrossModalEvalForwardMode.POSTERIOR,
                    )
            elif condition_type == "private":
                total += targets[1][0].size(0)
                main_targets = targets[1]
                cross_targets = targets[0]
                if rsample_type == "prior":
                    px_us_mean = idmvae_self_and_cross_modal_generation_eval(model,
                        data,
                        condition_type="private",
                        mode=CrossModalEvalForwardMode.PRIOR,
                    )
                elif rsample_type == "posterior":
                    px_us_mean = idmvae_self_and_cross_modal_generation_eval(model,
                        data,
                        condition_type="private",
                        mode=CrossModalEvalForwardMode.POSTERIOR,
                    )
            else:
                raise ValueError("Invalid condition_type. Choose 'shared' or 'private'.")

            for idx_srt, _ in enumerate(model.vaes):
                for idx_trg, _ in enumerate(model.vaes):
                    generated_img = px_us_mean[idx_srt][idx_trg].squeeze(0)
                    if condition_type == "shared":
                        clf_main = clfs_main[idx_trg]
                        clf_cross = clfs_cross[idx_trg]
                    else:
                        clf_main = clfs_main[idx_srt]
                        clf_cross = clfs_cross[idx_srt]

                    main_results = torch.argmax(clf_main(generated_img), dim=-1)
                    cross_results = torch.argmax(clf_cross(generated_img), dim=-1)
                    ground_truth_main = main_targets[idx_srt]
                    ground_truth_cross = cross_targets[idx_srt]
                    coherence_main[idx_srt][idx_trg] += (main_results == ground_truth_main).sum().item()
                    coherence_cross[idx_srt][idx_trg] += (
                        cross_results == ground_truth_cross
                    ).sum().item()

        if total == 0:
            return ([[0] * num_modalities for _ in range(num_modalities)], 0, [0] * num_modalities, 0, 0)

        for idx_trgt, _ in enumerate(model.vaes):
            for idx_strt, _ in enumerate(model.vaes):
                coherence_main[idx_strt][idx_trgt] /= total
                coherence_cross[idx_strt][idx_trgt] /= total

        means_selfcoh_main = mean([coherence_main[idx][idx] for idx, _ in enumerate(model.vaes)])
        means_selfcoh_cross = mean([coherence_cross[idx][idx] for idx, _ in enumerate(model.vaes)])
        means_target_main = [0 for _ in model.vaes]
        means_target_cross = [0 for _ in model.vaes]
        for idx_target, _ in enumerate(model.vaes):
            means_target_main[idx_target] = mean(
                [coherence_main[idx_start][idx_target] for idx_start, _ in enumerate(model.vaes) if idx_start != idx_target]
            )
            means_target_cross[idx_target] = mean(
                [coherence_cross[idx_start][idx_target] for idx_start, _ in enumerate(model.vaes) if idx_start != idx_target]
            )
            means_all_main = mean([coherence_main[idx_start][idx_target] for idx_start, _ in enumerate(model.vaes)])
            means_all_cross = mean([coherence_cross[idx_start][idx_target] for idx_start, _ in enumerate(model.vaes)])

    return (
        coherence_main,
        coherence_cross,
        means_selfcoh_main,
        means_selfcoh_cross,
        means_target_main,
        means_target_cross,
        mean(means_target_main),
        mean(means_target_cross),
        means_all_main,
        means_all_cross,
    )


def unconditional_coherence(model, test_loader, clfs, device):
    """Compute unconditional coherence."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for dataT in test_loader:
            data, _ = unpack_data_PolyMNIST(dataT, device)
            b_size = data[0].size(0)
            uncond_gens = polymnist_generate_unconditional(
                model, N=b_size, coherence_calculation=True, fid_calculation=False
            )
            uncond_gens = [elem.to(device) for elem in uncond_gens]
            clfs_resultss = []
            for idx_trg, _ in enumerate(model.vaes):
                clfs_results = torch.argmax(clfs[idx_trg](uncond_gens[idx_trg]), dim=-1)
                if idx_trg == 0:
                    total += b_size
                clfs_resultss.append(clfs_results)
            clfs_resultss_tensor = torch.stack(clfs_resultss, dim=-1)
            for dim in range(clfs_resultss_tensor.size(0)):
                if torch.unique(clfs_resultss_tensor[dim, :]).size(0) == 1:
                    correct += 1

        uncond_coherence = correct / total
        print(f"correct: {correct}, total: {total}")
        print(f"uncond_coherence: {uncond_coherence}")

    return uncond_coherence


def train_clf_lr(model, dl, device, args):
    """Train linear classifier on latent representations."""
    latent_rep = {f"m{i}": {"us": [], "zs": [], "ws": []} for i in range(5)}
    labels_all = []
    for dataT_lr in dl:
        data, labels_batch = unpack_data_PolyMNIST(dataT_lr, device=device)
        b_size = data[0].size(0)
        labels_batch = nn.functional.one_hot(labels_batch, num_classes=10).float()
        labels_all.append(labels_batch.cpu().data.numpy().reshape(b_size, 10))
        for v, vae in enumerate(model.vaes):
            with torch.no_grad():
                if args.use_mean_for_latent_clf:
                    mu_v, _ = vae.enc(data[v])
                    us_v = mu_v
                else:
                    us_v = vae.qu_x(*vae.enc(data[v])).rsample()
                ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                latent_rep[f"m{v}"]["us"].append(us_v.cpu().data.numpy())
                latent_rep[f"m{v}"]["zs"].append(zs_v.cpu().data.numpy())
                latent_rep[f"m{v}"]["ws"].append(ws_v.cpu().data.numpy())

    gt = np.argmax(np.concatenate(labels_all, axis=0), axis=1).astype(int)
    clf_lr = {}
    for v, _ in enumerate(model.vaes):
        latent_rep_u = np.concatenate(latent_rep[f"m{v}"]["us"], axis=0)
        latent_rep_w = np.concatenate(latent_rep[f"m{v}"]["ws"], axis=0)
        latent_rep_z = np.concatenate(latent_rep[f"m{v}"]["zs"], axis=0)
        clf_lr_rep_u = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=1000)
        clf_lr_rep_z = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=1000)
        clf_lr_rep_w = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=1000)
        clf_lr_rep_u.fit(latent_rep_u, gt.ravel()); clf_lr[f"m{v}_u"] = clf_lr_rep_u
        clf_lr_rep_w.fit(latent_rep_w, gt.ravel()); clf_lr[f"m{v}_w"] = clf_lr_rep_w
        clf_lr_rep_z.fit(latent_rep_z, gt.ravel()); clf_lr[f"m{v}_z"] = clf_lr_rep_z
    return clf_lr


def train_clf_lr_multi_labelType(model, subset_loader, device, args, condition_type=None):
    """Train linear classifier with shared/private labels."""
    latent_rep = {f"m{i}": {"us": [], "zs": [], "ws": []} for i in range(5)}
    labels_all = {v: [] for v in range(len(model.vaes))}
    for dataT_lr in subset_loader:
        data, labels_batch = unpack_data_PolyMNIST(dataT_lr, device=device)
        for v, vae in enumerate(model.vaes):
            if condition_type is None:
                labels = labels_batch[v].cpu().data.numpy()
            elif condition_type == "shared":
                labels = labels_batch[0][v].cpu().data.numpy()
            elif condition_type == "private":
                labels = labels_batch[1][v].cpu().data.numpy()
            else:
                raise ValueError(f"Invalid condition_type: {condition_type}. Choose 'shared' or 'private'.")
            labels_all[v].append(labels)
            with torch.no_grad():
                if args.use_mean_for_latent_clf:
                    mu_v, _ = vae.enc(data[v]); us_v = mu_v
                else:
                    us_v = vae.qu_x(*vae.enc(data[v])).rsample()
                ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                latent_rep[f"m{v}"]["us"].append(us_v.cpu().data.numpy())
                latent_rep[f"m{v}"]["zs"].append(zs_v.cpu().data.numpy())
                latent_rep[f"m{v}"]["ws"].append(ws_v.cpu().data.numpy())

    clf_lr = {}
    for v, _ in enumerate(model.vaes):
        gt = np.concatenate(labels_all[v], axis=0)
        latent_rep_u = np.concatenate(latent_rep[f"m{v}"]["us"], axis=0)
        latent_rep_w = np.concatenate(latent_rep[f"m{v}"]["ws"], axis=0)
        latent_rep_z = np.concatenate(latent_rep[f"m{v}"]["zs"], axis=0)
        clf_lr_rep_u = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=1000)
        clf_lr_rep_z = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=1000)
        clf_lr_rep_w = LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=1000)
        clf_lr_rep_u.fit(latent_rep_u, gt); clf_lr[f"m{v}_u"] = clf_lr_rep_u
        clf_lr_rep_w.fit(latent_rep_w, gt); clf_lr[f"m{v}_w"] = clf_lr_rep_w
        clf_lr_rep_z.fit(latent_rep_z, gt); clf_lr[f"m{v}_z"] = clf_lr_rep_z
    return clf_lr


def linear_latent_classification(model, test_loader, clf_lr, device, args):
    """Linear latent classification."""
    model.eval()
    lr_acc_all_z = []
    lr_acc_m0_z, lr_acc_m1_z, lr_acc_m2_z, lr_acc_m3_z, lr_acc_m4_z = [], [], [], [], []
    lr_acc_m0_w, lr_acc_m1_w, lr_acc_m2_w, lr_acc_m3_w, lr_acc_m4_w = [], [], [], [], []
    lr_acc_m0_u, lr_acc_m1_u, lr_acc_m2_u, lr_acc_m3_u, lr_acc_m4_u = [], [], [], [], []
    accuracies_lr = {}
    with torch.no_grad():
        for dataT in test_loader:
            data, targets = unpack_data_PolyMNIST(dataT, device)
            b_size = data[0].size(0)
            labels = nn.functional.one_hot(targets, num_classes=10).float().cpu().data.numpy().reshape(b_size, 10)
            if clf_lr is not None:
                latent_reps = []
                for v, vae in enumerate(model.vaes):
                    with torch.no_grad():
                        if args.use_mean_for_latent_clf:
                            mu_v, _ = vae.enc(data[v]); us_v = mu_v
                        else:
                            us_v = vae.qu_x(*vae.enc(data[v])).rsample()
                        ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                        latent_reps.append([us_v.cpu().data.numpy(), ws_v.cpu().data.numpy(), zs_v.cpu().data.numpy()])
                accuracies = classify_latent_representations(clf_lr, latent_reps, labels, split=True)
                lr_acc_m0_u.append(np.mean(accuracies["m0_u"])); lr_acc_m1_u.append(np.mean(accuracies["m1_u"]))
                lr_acc_m2_u.append(np.mean(accuracies["m2_u"])); lr_acc_m3_u.append(np.mean(accuracies["m3_u"]))
                lr_acc_m4_u.append(np.mean(accuracies["m4_u"]))
                lr_acc_m0_w.append(np.mean(accuracies["m0_w"])); lr_acc_m1_w.append(np.mean(accuracies["m1_w"]))
                lr_acc_m2_w.append(np.mean(accuracies["m2_w"])); lr_acc_m3_w.append(np.mean(accuracies["m3_w"]))
                lr_acc_m4_w.append(np.mean(accuracies["m4_w"]))
                lr_acc_m0_z.append(np.mean(accuracies["m0_z"])); lr_acc_m1_z.append(np.mean(accuracies["m1_z"]))
                lr_acc_m2_z.append(np.mean(accuracies["m2_z"])); lr_acc_m3_z.append(np.mean(accuracies["m3_z"]))
                lr_acc_m4_z.append(np.mean(accuracies["m4_z"]))
                lr_acc_all_z.append(np.mean(accuracies["all"]))

        accuracies_lr["m0_u"] = mean(lr_acc_m0_u); accuracies_lr["m1_u"] = mean(lr_acc_m1_u)
        accuracies_lr["m2_u"] = mean(lr_acc_m2_u); accuracies_lr["m3_u"] = mean(lr_acc_m3_u)
        accuracies_lr["m4_u"] = mean(lr_acc_m4_u)
        accuracies_lr["m0_w"] = mean(lr_acc_m0_w); accuracies_lr["m1_w"] = mean(lr_acc_m1_w)
        accuracies_lr["m2_w"] = mean(lr_acc_m2_w); accuracies_lr["m3_w"] = mean(lr_acc_m3_w)
        accuracies_lr["m4_w"] = mean(lr_acc_m4_w)
        accuracies_lr["m0_z"] = mean(lr_acc_m0_z); accuracies_lr["m1_z"] = mean(lr_acc_m1_z)
        accuracies_lr["m2_z"] = mean(lr_acc_m2_z); accuracies_lr["m3_z"] = mean(lr_acc_m3_z)
        accuracies_lr["m4_z"] = mean(lr_acc_m4_z)
        accuracies_lr["_mean_u"] = mean([accuracies_lr[f"m{n}_u"] for n in range(5)])
        accuracies_lr["_mean_w"] = mean([accuracies_lr[f"m{n}_w"] for n in range(5)])
        accuracies_lr["_mean_z"] = mean([accuracies_lr[f"m{n}_z"] for n in range(5)])
        accuracies_lr["z_all"] = mean(lr_acc_all_z)
    return accuracies_lr


def linear_latent_classification_multi_labelType(
    model, valid_loader, clf_lr, device, args, condition_type=None, plot_confidence=False
):
    """Linear latent classification in quadrant labels."""
    model.eval()
    lr_acc_all_z, lr_acc_all_w = [], []
    lr_acc_m0_z, lr_acc_m1_z, lr_acc_m2_z, lr_acc_m3_z, lr_acc_m4_z = [], [], [], [], []
    lr_acc_m0_w, lr_acc_m1_w, lr_acc_m2_w, lr_acc_m3_w, lr_acc_m4_w = [], [], [], [], []
    lr_acc_m0_u, lr_acc_m1_u, lr_acc_m2_u, lr_acc_m3_u, lr_acc_m4_u = [], [], [], [], []
    accuracies_lr = {}
    confidence_data_all = {}
    confidence_figs_all = {}

    with torch.no_grad():
        for dataT in valid_loader:
            data, targets = unpack_data_PolyMNIST(dataT, device)
            if condition_type is None:
                labels_batch = targets
            elif condition_type == "shared":
                labels_batch = targets[0]
            elif condition_type == "private":
                labels_batch = targets[1]

            if clf_lr is not None:
                latent_reps = []
                for v, vae in enumerate(model.vaes):
                    with torch.no_grad():
                        if args.use_mean_for_latent_clf:
                            mu_v, _ = vae.enc(data[v]); us_v = mu_v
                        else:
                            us_v = vae.qu_x(*vae.enc(data[v])).rsample()
                        ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                        latent_reps.append([us_v.cpu().data.numpy(), ws_v.cpu().data.numpy(), zs_v.cpu().data.numpy()])
                accuracies = classify_latent_representations_multi_labelType(
                    clf_lr, latent_reps, labels_batch, split=True
                )
                lr_acc_m0_u.append(np.mean(accuracies["m0_u"])); lr_acc_m1_u.append(np.mean(accuracies["m1_u"]))
                lr_acc_m2_u.append(np.mean(accuracies["m2_u"])); lr_acc_m3_u.append(np.mean(accuracies["m3_u"]))
                lr_acc_m4_u.append(np.mean(accuracies["m4_u"]))
                lr_acc_m0_w.append(np.mean(accuracies["m0_w"])); lr_acc_m1_w.append(np.mean(accuracies["m1_w"]))
                lr_acc_m2_w.append(np.mean(accuracies["m2_w"])); lr_acc_m3_w.append(np.mean(accuracies["m3_w"]))
                lr_acc_m4_w.append(np.mean(accuracies["m4_w"]))
                lr_acc_m0_z.append(np.mean(accuracies["m0_z"])); lr_acc_m1_z.append(np.mean(accuracies["m1_z"]))
                lr_acc_m2_z.append(np.mean(accuracies["m2_z"])); lr_acc_m3_z.append(np.mean(accuracies["m3_z"]))
                lr_acc_m4_z.append(np.mean(accuracies["m4_z"]))
                lr_acc_all_z.append(np.mean(accuracies["all_z"]))
                lr_acc_all_w.append(np.mean(accuracies["all_w"]))

                if "confidence_data" in accuracies:
                    for key, new_values in accuracies["confidence_data"].items():
                        if key in confidence_data_all:
                            confidence_data_all[key] = np.concatenate([confidence_data_all[key], new_values], axis=0)
                        else:
                            confidence_data_all[key] = new_values

        accuracies_lr["m0_u"] = mean(lr_acc_m0_u); accuracies_lr["m1_u"] = mean(lr_acc_m1_u)
        accuracies_lr["m2_u"] = mean(lr_acc_m2_u); accuracies_lr["m3_u"] = mean(lr_acc_m3_u)
        accuracies_lr["m4_u"] = mean(lr_acc_m4_u)
        accuracies_lr["m0_w"] = mean(lr_acc_m0_w); accuracies_lr["m1_w"] = mean(lr_acc_m1_w)
        accuracies_lr["m2_w"] = mean(lr_acc_m2_w); accuracies_lr["m3_w"] = mean(lr_acc_m3_w)
        accuracies_lr["m4_w"] = mean(lr_acc_m4_w)
        accuracies_lr["m0_z"] = mean(lr_acc_m0_z); accuracies_lr["m1_z"] = mean(lr_acc_m1_z)
        accuracies_lr["m2_z"] = mean(lr_acc_m2_z); accuracies_lr["m3_z"] = mean(lr_acc_m3_z)
        accuracies_lr["m4_z"] = mean(lr_acc_m4_z)
        accuracies_lr["_mean_u"] = mean([accuracies_lr[f"m{n}_u"] for n in range(5)])
        accuracies_lr["_mean_w"] = mean([accuracies_lr[f"m{n}_w"] for n in range(5)])
        accuracies_lr["_mean_z"] = mean([accuracies_lr[f"m{n}_z"] for n in range(5)])
        accuracies_lr["z_all"] = None if condition_type == "private" else mean(lr_acc_all_z)
        accuracies_lr["w_all"] = None

        if plot_confidence:
            for key, values in confidence_data_all.items():
                fig = plt.figure()
                plt.hist(values, bins=50, density=True)
                plt.xlabel("Confidence")
                plt.ylabel("Density")
                plt.title(f"Confidence Distribution for {key}")
                plt.grid()
                plt.tight_layout()
                confidence_figs_all[key] = fig
                plt.close(fig)
            accuracies_lr["confidence_data"] = confidence_data_all
            accuracies_lr["confidence_figs"] = confidence_figs_all

    return accuracies_lr


def calculate_fid_routine(datadirPM, fid_path, num_fid_samples, epoch, model, test_loader, device, args):
    """Calculate FID scores for unconditional and conditional generation."""
    total_cond = 0
    for j in [0, 1, 2, 3, 4]:
        if os.path.exists(os.path.join(fid_path, "random", f"m{j}")):
            shutil.rmtree(os.path.join(fid_path, "random", f"m{j}"))
        os.makedirs(os.path.join(fid_path, "random", f"m{j}"), exist_ok=True)
        for i in [0, 1, 2, 3, 4]:
            if os.path.exists(os.path.join(fid_path, f"m{j}", f"m{i}")):
                shutil.rmtree(os.path.join(fid_path, f"m{j}", f"m{i}"))
            os.makedirs(os.path.join(fid_path, f"m{j}", f"m{i}"), exist_ok=True)

    with torch.no_grad():
        for tranche in range(num_fid_samples // 100):
            polymnist_generate_unconditional(
                model,
                N=100,
                coherence_calculation=False,
                fid_calculation=True,
                savePath=fid_path,
                tranche=tranche,
            )
        for i, dataT in enumerate(test_loader):
            data, _ = unpack_data_PolyMNIST(dataT, device=device)
            if total_cond < num_fid_samples:
                polymnist_self_and_cross_modal_generation_for_fid_calculation(model, data, fid_path, i)
                total_cond += data[0].size(0)

        calculate_inception_features_for_gen_evaluation(args.inception_path, device, fid_path, datadirPM)
        fid_randm_list, fid_condgen_list = [], []
        fid_self_condgen_list, fid_cross_condgen_list = [], []
        for modality_target in [f"m{m}" for m in range(5)]:
            feats_real = np.load(
                os.path.join(args.datadir_fid, "PolyMNIST", "test", f"real_activations_{modality_target}.npy")
            )
            feats_randgen = np.load(os.path.join(fid_path, "random", modality_target + "_activations.npy"))
            fid_randval = calculate_fid(feats_real, feats_randgen)
            wandb.log({f"FID/Random/{modality_target}": fid_randval}, step=epoch)
            fid_randm_list.append(fid_randval)

            fid_condgen_target_list = []
            for modality_source in [f"m{m}" for m in range(5)]:
                feats_gen = np.load(
                    os.path.join(fid_path, modality_source, modality_target + "_activations.npy")
                )
                fid_val = calculate_fid(feats_real, feats_gen)
                wandb.log({f"FID/{modality_source}/{modality_target}": fid_val}, step=epoch)
                fid_condgen_target_list.append(fid_val)
                if modality_source == modality_target:
                    fid_self_condgen_list.append(fid_val)
                else:
                    fid_cross_condgen_list.append(fid_val)
            fid_condgen_list.append(mean(fid_condgen_target_list))

        wandb.log({"FID/random_meanall": mean(fid_randm_list)}, step=epoch)
        wandb.log({"FID/condgen_meanall": mean(fid_condgen_list)}, step=epoch)
        wandb.log({"FID/condgen_self_mean": mean(fid_self_condgen_list)}, step=epoch)
        wandb.log({"FID/condgen_cross_mean": mean(fid_cross_condgen_list)}, step=epoch)

    if os.path.exists(fid_path):
        shutil.rmtree(fid_path)
        os.makedirs(fid_path)

def calculate_inception_features_for_gen_evaluation(inception_state_dict_path, device, dir_fid_base, datadir, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=inception_state_dict_path)
    model = model.to(device)

    for moddality_num in range(5):
        moddality = 'm{}'.format(moddality_num)
        filename_act_real_calc = os.path.join(datadir, 'test','real_activations_{}.npy'.format(moddality))
        if not os.path.exists(filename_act_real_calc):
            files_real_calc = glob.glob(os.path.join(datadir,  'test', moddality, '*' + '.png'))
            act_real_calc = get_activations(files_real_calc, model, device, batch_size, dims, verbose=False)
            np.save(filename_act_real_calc, act_real_calc)

    for prefix  in ['random', 'm0', 'm1', 'm2', 'm3', 'm4']:
        dir_gen = os.path.join(dir_fid_base, prefix)
        if not os.path.exists(dir_gen):
            raise RuntimeError('Invalid path: %s' % dir_gen)
        for modality in ['m{}'.format(m) for m in range(5)]:
            files_gen = glob.glob(os.path.join(dir_gen, modality, '*' + '.png'))
            filename_act = os.path.join(dir_gen,
                                           modality + '_activations.npy')
            act_rand_gen = get_activations(files_gen, model, device, batch_size, dims, verbose=False)
            np.save(filename_act, act_rand_gen)

def calculate_inception_features_for_gen_evaluation_unimodal(inception_state_dict_path, device, dir_fid_base, datadir, modal, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=inception_state_dict_path)
    model = model.to(device)

    #for moddality_num in range(5):
    moddality_num = modal
    moddality = 'm{}'.format(moddality_num)
    filename_act_real_calc = os.path.join(datadir, 'test','real_activations_{}.npy'.format(moddality))
    if not os.path.exists(filename_act_real_calc):
        files_real_calc = glob.glob(os.path.join(datadir,  'test', moddality, '*' + '.png'))
        act_real_calc = get_activations(files_real_calc, model, device, batch_size, dims, verbose=False)
        np.save(filename_act_real_calc, act_real_calc)

    for prefix  in ['random', moddality]:
        dir_gen = os.path.join(dir_fid_base, prefix)
        if not os.path.exists(dir_gen):
            raise RuntimeError('Invalid path: %s' % dir_gen)
        #for modality in ['m{}'.format(m) for m in range(5)]:
        modality = moddality
        files_gen = glob.glob(os.path.join(dir_gen, modality, '*' + '.png'))
        filename_act = os.path.join(dir_gen, modality + '_activations.npy')
        act_rand_gen = get_activations(files_gen, model, device, batch_size, dims, verbose=False)
        np.save(filename_act, act_rand_gen)



def load_inception_activations(flags, modality=None, num_modalities=2, conditionals=None):
    if modality is None:
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_img_activations.npy')
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_img_activations.npy')
        filename_conditional = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'conditional_img_activations.npy')
        feats_real = np.load(filename_real)
        feats_random = np.load(filename_random)
        feats_cond = np.load(filename_conditional)
        feats = [feats_real, feats_random, feats_cond]
    else:
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + modality + '_activations.npy')
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + modality + '_activations.npy')
        feats_real = np.load(filename_real)
        feats_random = np.load(filename_random)

        #if num_modalities == 2:
            #filename_cond_gen = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'cond_gen_' + modality + '_activations.npy')
            #feats_cond_gen = np.load(filename_cond_gen)
            #feats = [feats_real, feats_random, feats_cond_gen]
        #elif num_modalities > 2:
            #if conditionals is None:
                #raise RuntimeError('conditionals are needed for num(M) > 2...')
        feats_cond_1a2m = dict()
        for k, key in enumerate(conditionals[0].keys()):
            filename_cond_1a2m = os.path.join(conditionals[0][key], key + '_' + modality + '_activations.npy')
            feats_cond_key = np.load(filename_cond_1a2m)
            feats_cond_1a2m[key] = feats_cond_key
        '''
            feats_cond_2a1m = dict()
            for k, key in enumerate(conditionals[1].keys()):
                filename_cond_1a2m = os.path.join(conditionals[1][key], key + '_' + modality + '_activations.npy')
                feats_cond_key = np.load(filename_cond_1a2m);
                feats_cond_2a1m[key] = feats_cond_key

            if flags.modality_jsd:
                if conditionals is None:
                    raise RuntimeError('conditionals are needed for num(M) > 2...')
                feats_cond_dyn_prior_2a1m = dict()
                for k, key in enumerate(conditionals[2].keys()):
                    filename_dp_2a1m = os.path.join(conditionals[2][key], key + '_' + modality + '_activations.npy')
                    feats_dp_key = np.load(filename_dp_2a1m);
                    feats_cond_dyn_prior_2a1m[key] = feats_dp_key
            else:
                feats_cond_dyn_prior_2a1m = None;
        '''
        feats = [feats_real, feats_random, feats_cond_1a2m] #, feats_cond_2a1m, feats_cond_dyn_prior_2a1m]
    return feats

def calculate_fid(feats_real, feats_gen):
    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid


def calculate_fid_dict(feats_real, dict_feats_gen):
    dict_fid = dict()
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key]
        dict_fid[key] = calculate_fid(feats_real, feats_gen)
    return dict_fid


def get_clf_activations(flags, data, model):
    act = model.get_activations(data)
    act = act.cpu().data.numpy().reshape(flags.batch_size, -1)
    return act


def classify_latent_representations(clf_lr, data, labels, split=False): # dict, [[B,U],[B,W],[B,Z]][M], [B,10]
    # import pdb; pdb.set_trace()
    gt = np.argmax(labels, axis=1).astype(int) # (B,), shared by all modalities, pick one
    accuracies = dict()

    y_preds_z = []
    y_preds_w = []
    for k, data_k in enumerate(data):
        if split:
            data_rep_u, data_rep_w, data_rep_z = data_k # [B,U],[B,W],[B,Z]
        else:
            data_rep_u = data[0][0]

        clf_key_u = 'm' + str(k) + '_'+'u'
        clf_lr_rep_u = clf_lr[clf_key_u]
        y_pred_rep_u = clf_lr_rep_u.predict(data_rep_u) # (B,), predicted labels (hard prediction)
        accuracy_rep_u = accuracy_score(gt, y_pred_rep_u.ravel()) # same as y_pred_rep_u
        accuracies[clf_key_u] = accuracy_rep_u

        if split:

            clf_key_z = 'm' + str(k) + '_' + 'z'
            clf_lr_rep_z = clf_lr[clf_key_z]
            y_pred_rep_z = clf_lr_rep_z.predict(data_rep_z)
            accuracy_rep_z = accuracy_score(gt, y_pred_rep_z.ravel())
            accuracies[clf_key_z] = accuracy_rep_z

            clf_key_w = 'm' + str(k) + '_' + 'w'
            clf_lr_rep_w = clf_lr[clf_key_w]
            y_pred_rep_w = clf_lr_rep_w.predict(data_rep_w)
            accuracy_rep_w = accuracy_score(gt, y_pred_rep_w.ravel())
            accuracies[clf_key_w] = accuracy_rep_w
            y_preds_z.append(y_pred_rep_z)
            y_preds_w.append(y_pred_rep_w)
    # overall_preds = stats.mode(np.stack(y_preds_z))[0]
    overall_preds_z = stats.mode(np.stack(y_preds_z))[0]
    overall_preds_w = stats.mode(np.stack(y_preds_w))[0]
    # accuracy_rep_all = accuracy_score(gt, overall_preds.ravel())
    accuracy_rep_all_z = accuracy_score(gt, overall_preds_z.ravel())
    accuracy_rep_all_w = accuracy_score(gt, overall_preds_w.ravel())
    # accuracies['all'] = accuracy_rep_all
    accuracies['all_z'] = accuracy_rep_all_z
    accuracies['all_w'] = accuracy_rep_all_w

    return accuracies

def classify_latent_representations_multi_labelType(clf_lr, data, labels, split=False, plot_confidence_bs=False
                                             , wandb_log=False, wandb_run=None
                                             ):  # dict, [[B,W+Z],[B,W],[B,Z]][M], [B][M]
    # convert labels (5-view list of tensors) to numpy array
    # import pdb; pdb.set_trace()
    gt = [label.cpu().numpy() for label in labels] # (B,)[M]
    accuracies = dict()
    confidence_figs = dict() if plot_confidence_bs else None
    confidence_data = dict()

    y_preds_z = []
    y_preds_w = []
    for k, data_k in enumerate(data):
        if split:
            data_rep_u, data_rep_w, data_rep_z = data_k
        else:
            data_rep_u = data[0][0]
        # import pdb; pdb.set_trace()

        # --- U ---
        clf_key_u = 'm' + str(k) + '_'+'u'
        clf_lr_rep_u = clf_lr[clf_key_u]
        y_proba_rep_u = clf_lr_rep_u.predict_proba(data_rep_u)
        # y_pred_rep_u = np.argmax(y_proba_rep_u, axis=1)
        y_pred_rep_u = clf_lr_rep_u.predict(data_rep_u)
        accuracy_rep_u = accuracy_score(gt[k], y_pred_rep_u.ravel())
        accuracies[clf_key_u] = accuracy_rep_u
        confidences_u = np.max(y_proba_rep_u, axis=1)
        confidence_data[f"{clf_key_u}"] = confidences_u

        # Confidence histogram plotting is optional for minibatch debugging.
        # Plot confidence histogram for U
        if plot_confidence_bs:
            # confidences_u = np.max(y_proba_rep_u, axis=1)
            fig_u = plt.figure()
            plt.hist(confidences_u, bins=20, alpha=0.6)
            plt.title(f"Confidence distribution (U latent, m{k})")
            plt.xlabel("Confidence")
            plt.ylabel("Frequency")
            plt.tight_layout()
            confidence_figs[f"{clf_key_u}"] = fig_u
            plt.close(fig_u)

        if split:
            # --- Z ---
            clf_key_z = 'm' + str(k) + '_' + 'z'
            clf_lr_rep_z = clf_lr[clf_key_z]
            y_proba_rep_z = clf_lr_rep_z.predict_proba(data_rep_z)
            # y_pred_rep_z = np.argmax(y_proba_rep_z, axis=1)
            y_pred_rep_z = clf_lr_rep_z.predict(data_rep_z)
            accuracy_rep_z = accuracy_score(gt[k], y_pred_rep_z.ravel())
            accuracies[clf_key_z] = accuracy_rep_z
            confidences_z = np.max(y_proba_rep_z, axis=1)
            confidence_data[f"{clf_key_z}"] = confidences_z

            # Plot confidence histogram for Z
            if plot_confidence_bs:
                # confidences_z = np.max(y_proba_rep_z, axis=1)
                # plt.figure()
                fig_z = plt.figure()
                plt.hist(confidences_z, bins=20, alpha=0.6)
                plt.title(f"Confidence distribution (Z latent, m{k})")
                plt.xlabel("Confidence")
                plt.ylabel("Frequency")
                plt.tight_layout()
                confidence_figs[f"{clf_key_z}"] = fig_z
                # plt.show()
                # if wandb_log and wandb_run:
                #     wandb_run.log({f"confidence/{clf_key_z}": wandb.Image(fig_z)})
                plt.close(fig_z)

            # --- W ---
            clf_key_w = 'm' + str(k) + '_' + 'w'
            clf_lr_rep_w = clf_lr[clf_key_w]
            y_proba_rep_w = clf_lr_rep_w.predict_proba(data_rep_w)
            # y_pred_rep_w = np.argmax(y_proba_rep_w, axis=1)
            y_pred_rep_w = clf_lr_rep_w.predict(data_rep_w)
            accuracy_rep_w = accuracy_score(gt[k], y_pred_rep_w.ravel())
            accuracies[clf_key_w] = accuracy_rep_w
            confidences_w = np.max(y_proba_rep_w, axis=1)
            confidence_data[f"{clf_key_w}"] = confidences_w

            # import pdb; pdb.set_trace()

            # Plot confidence histogram for W
            if plot_confidence_bs:
                # confidences_w = np.max(y_proba_rep_w, axis=1)
                # plt.figure()
                fig_w = plt.figure()
                plt.hist(confidences_w, bins=20, alpha=0.6)
                plt.title(f"Confidence distribution (W latent, m{k})")
                plt.xlabel("Confidence")
                plt.ylabel("Frequency")
                plt.tight_layout()
                confidence_figs[f"{clf_key_w}"] = fig_w
                plt.close(fig_w)

            y_preds_z.append(y_pred_rep_z)
            y_preds_w.append(y_pred_rep_w)

    overall_preds_z = stats.mode(np.stack(y_preds_z))[0]
    overall_preds_w = stats.mode(np.stack(y_preds_w))[0] # meaningless parameter for quadrant classification just using gt[0] for all
    
    # NOTE: meaningless parameter for quadrant classification just using gt[0] for all
    accuracy_rep_all_z = accuracy_score(gt[0], overall_preds_z.ravel()) 
    accuracy_rep_all_w = accuracy_score(gt[0], overall_preds_w.ravel()) # meaningless parameter for quadrant classification just using gt[0] for all

    accuracies['all_z'] = accuracy_rep_all_z
    accuracies['all_w'] = accuracy_rep_all_w # meaningless parameter for quadrant classification just using gt[0] for all
    accuracies['confidence_data'] = confidence_data
    if plot_confidence_bs:
        accuracies['confidence_figs'] = confidence_figs

    return accuracies


def polymnist_generate_unconditional(
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
            samples = samples.data.cpu()
            for image in range(samples.size(0)):
                save_image(
                    samples[image, :, :, :],
                    "{}/random/m{}/{}_{}.png".format(savePath, i, tranche, image),
                )
    else:
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            samples = samples.view(samples.size()[0], *samples.size()[1:])
            outputs.append(make_grid(samples, nrow=int(sqrt(N))))

    return outputs


def polymnist_generate_unconditional_plot(model, num_rows=10, num_cols=10):
    outputs = []
    samples_tensor_list = []
    samples_list_to_one_grid = []
    N = num_rows * num_cols
    samples_list = idmvae_generate_unconditional(model, N)
    for i, samples in enumerate(samples_list):
        samples = samples.data.cpu()
        samples = samples.view(samples.size()[0], *samples.size()[1:])
        outputs.append(
            make_grid(samples, nrow=num_cols, padding=2, pad_value=0.0)
        )

        samples_tensor_list.append(samples)

    for n in range(N):
        for i, samples_tensor in enumerate(samples_tensor_list):
            samples_list_to_one_grid.append(samples_tensor[n])

    combined_grid = make_grid(
        samples_list_to_one_grid, nrow=5, padding=2, pad_value=0.0
    )

    return outputs, combined_grid


def polymnist_self_and_cross_modal_generation_for_fid_calculation(model, data, savePath, i):
    recons_mat = idmvae_self_and_cross_modal_generation_eval(model, [d for d in data])
    for r, recons_list in enumerate(recons_mat):
        for o, recon in enumerate(recons_list):
            recon = recon.squeeze(0).cpu()

            for image in range(recon.size(0)):
                save_image(
                    recon[image, :, :, :],
                    "{}/m{}/m{}/{}_{}.png".format(savePath, r, o, image, i),
                )


def polymnist_self_and_cross_modal_generation_eval(
    model,
    data,
    num=10,
    N=10,
    *,
    mode=None,
    condition_type=None,
    num_comb=5,
    N_comb=5,
    return_denoised=False,
    data_ctrl=None,
    K=1,
):
    if mode is None:
        recon_triess = [[[] for i in range(num)] for j in range(num)]
        outputss = [[[] for i in range(num)] for j in range(num)]
        for i in range(N):
            recons_mat = idmvae_self_and_cross_modal_generation_eval(
                model, [d[:num] for d in data]
            )
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.squeeze(0).cpu()
                    recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([data[r][:num].cpu()] + recon_triess[r][o]), nrow=num
                )
        return outputss

    num_modalities = len(data)

    if mode == CrossModalEvalForwardMode.PRIOR_CTRL:
        recon_triess = [
            [[] for i in range(num_modalities)] for j in range(num_modalities)
        ]
        outputss = [
            [[] for i in range(num_modalities)] for j in range(num_modalities)
        ]
        for i in range(N):
            recons_mat = idmvae_self_and_cross_modal_generation_eval(
                model,
                [d[:num] for d in data],
                mode=CrossModalEvalForwardMode.PRIOR_CTRL,
                condition_type=condition_type,
                return_denoised=return_denoised,
            )
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.squeeze(0).cpu()
                    recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([data[r][:num].cpu()] + recon_triess[r][o]), nrow=num
                )
        recon_grid_combined_final = []
        recon_grid_combined = []
        recon_grid = [[] for _ in range(num_modalities)]
        for n in range(N):
            for r in range(num_modalities):
                for o in range(num_modalities):
                    recon_triess[r][o][n] = recon_triess[r][o][n][:num_comb]
        for r in range(num_modalities):
            for o in range(num_modalities):
                tensors_r_o = torch.cat(recon_triess[r][o][:N_comb], dim=0)
                recon_grid[r].append(tensors_r_o)
            tensors_r = torch.cat(recon_grid[r], dim=0)
            recon_grid_combined.append(tensors_r)
            tensors_r_with_input = torch.cat(
                [data[r][:num_comb].cpu()] + [recon_grid_combined[r]], dim=0
            )
            recon_grid_combined_final.append(
                make_grid(
                    tensors_r_with_input, nrow=num_comb, padding=2, pad_value=0.0
                )
            )
        return outputss, recon_grid_combined_final

    if mode == CrossModalEvalForwardMode.POSTERIOR_CTRL:
        recon_triess = [
            [[] for i in range(num_modalities)] for j in range(num_modalities)
        ]
        outputss = [
            [[] for i in range(num_modalities)] for j in range(num_modalities)
        ]
        for i in range(N):
            dc = [d[num + i].unsqueeze(0).repeat(num, 1, 1, 1) for d in data]
            recons_mat = idmvae_self_and_cross_modal_generation_eval(
                model,
                [d[:num] for d in data],
                mode=CrossModalEvalForwardMode.POSTERIOR_CTRL,
                data_ctrl=dc,
                condition_type=condition_type,
                return_denoised=return_denoised,
            )
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.squeeze(0).cpu()
                    recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([data[r][:num].cpu()] + recon_triess[r][o]), nrow=num
                )
        outputss_extended = [
            [[] for _ in range(num_modalities)] for _ in range(num_modalities)
        ]
        for r in range(num_modalities):
            for o in range(num_modalities):
                placeholder = torch.full_like(data[r][0].cpu(), 0.5)
                data_row_imgs = data[r][:num].cpu()
                data_rolled_column_imgs = data[r][num : num + N].cpu()
                generated_imgs_list = recon_triess[r][o]
                all_images_for_grid = [placeholder] + list(data_row_imgs)
                for ii in range(N):
                    all_images_for_grid.append(data_rolled_column_imgs[ii])
                    all_images_for_grid.extend(list(generated_imgs_list[ii]))
                outputss_extended[r][o] = make_grid(
                    all_images_for_grid, nrow=num + 1, padding=2, pad_value=0.0
                )
        recon_grid_combined_final = []
        recon_grid_combined = []
        recon_grid = [[] for _ in range(num_modalities)]
        for n in range(N):
            for r in range(num_modalities):
                for o in range(num_modalities):
                    recon_triess[r][o][n] = recon_triess[r][o][n][:num_comb]
        for r in range(num_modalities):
            for o in range(num_modalities):
                tensors_r_o = torch.cat(recon_triess[r][o][:N_comb], dim=0)
                recon_grid[r].append(tensors_r_o)
            tensors_r = torch.cat(recon_grid[r], dim=0)
            recon_grid_combined.append(tensors_r)
            tensors_r_with_input = torch.cat(
                [data[r][:num_comb].cpu()] + [recon_grid_combined[r]], dim=0
            )
            recon_grid_combined_final.append(
                make_grid(
                    tensors_r_with_input, nrow=num_comb, padding=2, pad_value=0.0
                )
            )
        return outputss, outputss_extended

    if mode == CrossModalEvalForwardMode.POSTERIOR_NONSHUF:
        recon_triess = [
            [[] for i in range(num_modalities)] for j in range(num_modalities)
        ]
        outputss = [
            [[] for i in range(num_modalities)] for j in range(num_modalities)
        ]
        for i in range(N):
            recons_mat = idmvae_self_and_cross_modal_generation_eval(
                model,
                [d[:num] for d in data],
                mode=CrossModalEvalForwardMode.POSTERIOR_NONSHUF,
                condition_type=condition_type,
                return_denoised=return_denoised,
            )
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.squeeze(0).cpu()
                    recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(
                    torch.cat([data[r][:num].cpu()] + recon_triess[r][o]), nrow=num
                )
        return outputss

    raise ValueError(
        f"Unsupported mode {mode!r} for polymnist_self_and_cross_modal_generation_eval"
    )

