"""TCGA-specific evaluation entrypoints."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn import functional as F

def train_tcga_latent_classifiers(model, dl, device, args):
    model.eval()
    X_u, X_w, X_z, Y = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}
    with torch.no_grad():
        for (views, labels) in dl:
            labels = labels.view(-1)
            keep = labels >= 0
            if keep.sum() == 0:
                continue
            labels_np = labels[keep].cpu().numpy()
            for v, vae in enumerate(model.vaes):
                x_v = views[v][keep].to(device)
                mu, _ = vae.enc(x_v)
                w, z = torch.split(mu, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                X_u[v].append(mu.cpu().numpy())
                X_w[v].append(w.cpu().numpy())
                X_z[v].append(z.cpu().numpy())
                Y[v].append(labels_np)

    def _concat_np(dct):
        return {0: np.concatenate(dct[0], axis=0), 1: np.concatenate(dct[1], axis=0)}

    X_u, X_w, X_z = _concat_np(X_u), _concat_np(X_w), _concat_np(X_z)
    y_v = {0: np.concatenate(Y[0], axis=0), 1: np.concatenate(Y[1], axis=0)}

    def make_lr():
        return Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(random_state=args.seed, max_iter=1000))])

    clfs = {}
    for v in (0, 1):
        lr_u = make_lr(); lr_u.fit(X_u[v], y_v[v])
        lr_w = make_lr(); lr_w.fit(X_w[v], y_v[v])
        lr_z = make_lr(); lr_z.fit(X_z[v], y_v[v])
        clfs[f"m{v}_u"] = lr_u
        clfs[f"m{v}_w"] = lr_w
        clfs[f"m{v}_z"] = lr_z
    return clfs


def eval_tcga_latent_classifiers(model, dl, clfs, device, args):
    model.eval()
    probs_u, probs_w, probs_z, ys = [], [], [], []

    def get_probs(clf, X_tensor):
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X_tensor.cpu().numpy())[:, 1]
        clf_device = next(clf.parameters()).device
        logits = clf(X_tensor.to(clf_device))
        return F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    with torch.no_grad():
        for (views, labels) in dl:
            labels = labels.view(-1)
            keep = labels >= 0
            if keep.sum() == 0:
                continue
            y = labels[keep].cpu().numpy()
            mu_list, w_list, z_list = [], [], []
            for v, vae in enumerate(model.vaes):
                x_v = views[v][keep].to(device)
                mu, _ = vae.enc(x_v)
                w, z = torch.split(mu, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                mu_list.append(mu); w_list.append(w); z_list.append(z)

            p_u_v0 = get_probs(clfs["m0_u"], mu_list[0]); p_u_v1 = get_probs(clfs["m1_u"], mu_list[1])
            p_w_v0 = get_probs(clfs["m0_w"], w_list[0]); p_w_v1 = get_probs(clfs["m1_w"], w_list[1])
            p_z_v0 = get_probs(clfs["m0_z"], z_list[0]); p_z_v1 = get_probs(clfs["m1_z"], z_list[1])
            probs_u.append(0.5 * (p_u_v0 + p_u_v1))
            probs_w.append(0.5 * (p_w_v0 + p_w_v1))
            probs_z.append(0.5 * (p_z_v0 + p_z_v1))
            ys.append(y)

    y = np.concatenate(ys)
    p_u = np.concatenate(probs_u)
    p_w = np.concatenate(probs_w)
    p_z = np.concatenate(probs_z)
    return {
        "AUROC_u": roc_auc_score(y, p_u) if len(np.unique(y)) > 1 else float("nan"),
        "AUROC_w": roc_auc_score(y, p_w) if len(np.unique(y)) > 1 else float("nan"),
        "AUROC_z": roc_auc_score(y, p_z) if len(np.unique(y)) > 1 else float("nan"),
        "ACC_u": accuracy_score(y, (p_u >= 0.5).astype(int)),
        "ACC_w": accuracy_score(y, (p_w >= 0.5).astype(int)),
        "ACC_z": accuracy_score(y, (p_z >= 0.5).astype(int)),
    }

