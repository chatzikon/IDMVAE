# IDMVAE — Disentanglement of Variations with Multimodal Generative Modeling

Official PyTorch implementation of **IDMVAE** (Information-Disentangled Multimodal VAE) from the ICLR 2026 paper *Disentanglement of Variations with Multimodal Generative Modeling*.

| Resource | Link |
|----------|------|
| OpenReview (paper + BibTeX) | [ICLR 2026 forum page](https://openreview.net/forum?id=DcHGEcqdFf) |

## What this repository contains

- Training and evaluation code for IDMVAE on PolyMNIST, CUB-200-2011 (CUBcluster8 at 256px), CelebAMask-HQ, and TCGA (two complete views).
- Shell launchers under `src/commands/` for the main experiment workflows.
- `figures/` for assets used in the paper and repository.
- **`src/baseline/`** — third-party and legacy baseline code kept for reference. And a cleaned baseline bundle may be published later.

Local run directories such as `outputs/` and `src/wandb/` are git-ignored; they are not part of the release tree.

## Requirements

- **`requirements.txt`** is a **fully pinned** dependency lock (including transitive packages). It was produced with [`pip-tools`](https://pip-tools.readthedocs.io/) from **`requirements.in`** on **Python 3.12**, using PyTorch **2.5.1** with **CUDA 12** wheels from PyPI (`nvidia-*` packages in the lockfile).
- **Regenerate** after editing direct dependencies: `pip install pip-tools` then  
  `pip-compile --strip-extras -o requirements.txt requirements.in`  
  (optionally install your desired `torch` / `torchvision` / `torchaudio` build first, or add `--extra-index-url` for another CUDA major).
- Historical paper runs may have used an older stack; if you still need that environment, keep a separate conda env or a branch-specific `requirements-legacy.in` / lockfile rather than mixing pins.

```bash
cd /path/to/this/repo
pip install -r requirements.txt
```

## Data layout

**We do not currently redistribute zipped datasets or provide direct download mirrors from this repository.** Obtain the original corpora from their official releases (e.g. CUB-200-2011, CelebAMask-HQ, TCGA, and MNIST-derived assets for PolyMNIST-style setups), then prepare paths locally.

This codebase **does** ship **dataset loaders** under `src/` (`dataset_PolyMNIST_quadrant.py`, `dataset_CUBcluster8.py`, `dataset_CelebAMask_HQ.py`, `dataset_TCGA_2_complete_views.py`, …) and **helper scripts** to build or convert assets:

| Role | Location |
|------|----------|
| PolyMNIST quadrant generation | `src/commands/functions_helper/gen_polyMNIST_quadrant.sh` |
| PolyMNIST PNG → PT conversion | `src/commands/functions_helper/convert_polymnist_to_pt.sh` |
| PolyMNIST digit classifier pretraining | `src/commands/functions_helper/pretrain_classifier_polyMNIST.sh` |
| CelebAMask-HQ → PT conversion | `src/commands/functions_helper/convert_celebamask_hq_to_pt.sh` |
| Latent pre-generation (32×32 ×4, for diffusion / denoisers) | `src/commands/functions_post_eval/pregen_4x32x32_dataset_CUB256.sh`, `pregen_4x32x32_dataset_CelebAMask_HQ.sh` |

Read the comments at the top of each script for `OUT_ROOT` / `SRC_ROOT` and other env overrides. **TCGA** expects pre-built `.npz` tensors per split (see below); preparation of those arrays is outside the snippets above—use your own pipeline to match `train_IDMVAE_TCGA.py`.

| Dataset | Notes |
|---------|--------|
| PolyMNIST | Quadrant pipeline (`PolyMNISTDataset_pt` in `dataset_PolyMNIST_quadrant.py`) |
| CUB | **256px** CUBcluster8 split used in this repo (see dataloader + experiment scripts for directory layout) |
| CelebAMask-HQ | Images + masks; dataloaders in `dataset_CelebAMask_HQ.py` |
| TCGA | Two complete views; `complete_views_split{k}_{tr,val,te}.npz` under `DATADIR` |

Typical environment variables (defaults in some shell scripts are cluster placeholders—**override** for your machine):

```bash
export POLYMNIST_ROOT=/path/to/PolyMNIST
export CUB_ROOT=/path/to/CUB
export CELEBAMASK_ROOT=/path/to/CelebAMask-HQ_from_SBM
export DENOISER_ROOT=/path/to/pregen_4x32x32/_denoiser

# TCGA (`commands/run_TCGA_experiment.sh`)
export DATADIR=/path/to/TCGA/complete_splits
```

`DATADIR` for TCGA must contain `complete_views_split{k}_tr.npz`, `complete_views_split{k}_val.npz`, and `complete_views_split{k}_te.npz` for each split index `k` the script runs (default `k` in `0..4`).

Use each training script’s `DATADIR` / `--datadir` (or documented overrides) for the **exact** split directory used in your run.

## Running experiments

Working directory: **`src/`** (repository root is the parent of `src/`).

| Dataset | Command script (from `src/`) |
|---------|------------------------------|
| PolyMNIST | `python` / shell: `commands/run_PolyMNIST_experiment_train_and_checkpoint.sh` |
| CUB 256px | `commands/run_CUB_experiment_train_and_checkpoint_256.sh` |
| CelebAMask-HQ | `commands/run_CelebAMask_experiment_train_and_checkpoint.sh` |
| TCGA | `commands/run_TCGA_experiment.sh` |

Before running: set `MODE` (`train` / `resume` / `test`, etc.), dataset roots, checkpoint locations, and GPU ids inside the chosen script.

### Post-training / denoiser utilities

Scripts under `src/commands/functions_post_eval/` (for example denoiser train/eval) use environment variables such as `DATA_PATH`, `HIGH_RES_DATA_PATH`, `CKPT`, `RESNET_MODEL_ARGS`, and `RESNET_CHECKPOINT`. **Set these for your machine**; do not rely on any cluster-specific defaults.

## Weights & Biases

Logging with [Weights & Biases](https://wandb.ai/) is optional. If you use it, authenticate in your environment (do not commit API keys):

```bash
export WANDB_API_KEY=<your_key>
```

## License

This project is released under the **MIT License**; see [`LICENSE`](LICENSE). The implementation builds on [MMVAEplus](https://github.com/epalu/mmvaeplus) and other credited work (see below).

## Citation

```bibtex
@inproceedings{zhang2026idmvae,
  title     = {Disentanglement of Variations with Multimodal Generative Modeling},
  author    = {Yijie Zhang and Yiyang Shen and Weiran Wang},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=DcHGEcqdFf}
}
```

## Acknowledgements

This codebase extends ideas and code from the multimodal VAE community, in particular:

- [MMVAEplus](https://github.com/epalu/mmvaeplus)
- [MMVAE](https://github.com/iffsid/mmvae)
- [MoPoE](https://github.com/thomassutter/MoPoE)
- [DMVAE](https://github.com/seqam-lab/DMVAE)
- [DisentangledSSL](https://github.com/uhlerlab/DisentangledSSL)
- [SBM](https://github.com/DanielMitiku/score_based_multimodal_autoencoder)
