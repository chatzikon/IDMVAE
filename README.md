# IDMVAE — Disentanglement of Variations with Multimodal Generative Modeling

Official PyTorch implementation of **IDMVAE** (Information-Disentangled Multimodal VAE) from the ICLR 2026 paper <a href="https://openreview.net/forum?id=DcHGEcqdFf">*Disentanglement of Variations with Multimodal Generative Modeling*</a>.

## What this repository contains

- Training and evaluation code for IDMVAE on PolyMNIST, CUB-200-2011 (CUBcluster8 at 256px), and TCGA (two complete views).
- Shell launchers under `src/commands/` for the main experiment workflows.
- **`src/baseline/`** — third-party and legacy baseline code kept for reference. And a cleaned baseline bundle may be published later.

<!--
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
-->

<!-- | CelebAMask-HQ → PT conversion | `src/commands/functions_helper/convert_celebamask_hq_to_pt.sh` |-->
<!-- | CelebAMask-HQ | Images + masks; dataloaders in `dataset_CelebAMask_HQ.py` |-->
<!-- | **CelebAMask-HQ** | Face images + parsing masks (`dataset_CelebAMask_HQ.py`) | [CelebAMask-HQ (GitHub)](https://github.com/switchablenorms/CelebAMask-HQ) |-->
<!-- | CelebAMask-HQ | `commands/run_CelebAMask_experiment_train_and_checkpoint.sh` |-->



## Data layout

This repo ships **dataloaders and preparation scripts** only (no raw dataset archives in Git). Point them at data on disk using the environment variables below.

| Source | What you get |
|--------|----------------|
| [Dataset references](#dataset-references) | Official download links and BibTeX for each benchmark |
| [ICLR2026_IDMVAE (SharePoint)](https://iowa-my.sharepoint.com/:f:/g/personal/wwang157_uiowa_edu/IgCBk365vyJPQ55IlXTG0E3eAVobHl9UbX3AgLh25ZASjI4?e=IcY4zc) | Preprocessed splits, checkpoints, and preparation notes |

**Loaders** (`dataset_PolyMNIST_quadrant.py`, `dataset_CUBcluster8.py`, `dataset_TCGA_2_complete_views.py`, …) and **helper scripts**:

| Role | Location |
|------|----------|
| PolyMNIST quadrant generation | `src/commands/functions_helper/gen_polyMNIST_quadrant.sh` |
| PolyMNIST PNG → PT conversion | `src/commands/functions_helper/convert_polymnist_to_pt.sh` |
| PolyMNIST digit classifier pretraining | `src/commands/functions_helper/pretrain_classifier_polyMNIST.sh` |
| Latent pre-generation (32×32 ×4, for diffusion / denoisers) | `src/commands/functions_post_eval/pregen_4x32x32_dataset_CUB256.sh` |

Read the comments at the top of each script for `OUT_ROOT` / `SRC_ROOT` and other env overrides. **TCGA** expects pre-built `.npz` tensors per split (see below); preparation of those arrays is outside the snippets above—use your own pipeline to match `train_IDMVAE_TCGA.py`.

| Dataset | Notes |
|---------|--------|
| PolyMNIST | Quadrant pipeline (`PolyMNISTDataset_pt` in `dataset_PolyMNIST_quadrant.py`) |
| CUB | **256px** CUBcluster8 split used in this repo (see dataloader + experiment scripts for directory layout) |
| TCGA | Two complete views; `complete_views_split{k}_{tr,val,te}.npz` under `DATADIR` |

Typical environment variables (defaults in some shell scripts are cluster placeholders—**override** for your machine):

```bash
export POLYMNIST_ROOT=/path/to/PolyMNIST
export CUB_ROOT=/path/to/CUB
export DENOISER_ROOT=/path/to/pregen_4x32x32/_denoiser

# TCGA (`commands/run_TCGA_experiment.sh`)
export DATADIR=/path/to/TCGA/complete_splits
```

`DATADIR` for TCGA must contain `complete_views_split{k}_tr.npz`, `complete_views_split{k}_val.npz`, and `complete_views_split{k}_te.npz` for each split index `k` the script runs (default `k` in `0..4`).

Use each training script’s `DATADIR` / `--datadir` (or documented overrides) for the **exact** split directory used in your run.

## Dataset references

Please cite the **original dataset publications** when you use these benchmarks (in addition to [citing IDMVAE](#citation) if you use this code). We do not host the raw corpora; use the official sources below.

| Dataset | Role in this repo | Official source |
|---------|-------------------|-----------------|
| **MNIST** | Base digits for building PolyMNIST | **Citation:** [LeCun MNIST page](http://yann.lecun.com/exdb/mnist/) (directory often empty). **Downloads:** `torchvision.datasets.MNIST` or [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist) |
| **PolyMNIST** | 5-view colored MNIST (quadrant layout; see `dataset_PolyMNIST_quadrant.py`) | **Dataset introduced in** [Sutter et al., *Generalized Multimodal ELBO* (MoPoE)](https://arxiv.org/abs/2105.02470); prebuilt archives also via [MMVAE+](https://github.com/epalu/mmvaeplus#download-data) / [MoPoE](https://github.com/thomassutter/MoPoE) |
| **CUB-200-2011** | Bird images + attributes (base corpus) | [Caltech CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) |
| **CUBcluster8 (256px)** | Image + caption pairs, 8-species cluster split (`dataset_CUBcluster8.py`) | Built on CUB-200-2011; cluster grouping follows the **CUBICC** line of work in [CMVAE](https://github.com/epalu/CMVAE#cubicc) (256×256 preprocessed tensors: `images.pt`, `captions.pt`, `labels_cluster.pt`, …) |
| **TCGA** | Two complete omics views in `.npz` splits (`dataset_TCGA_2_complete_views.py`) | [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/tcga) via [GDC](https://portal.gdc.cancer.gov/) |

**Image–caption pairing for CUB:** widely attributed to Reed et al. (fine-grained captioning); include that citation if your work uses the caption modality.

**TCGA note:** this repository expects **preprocessed** `complete_views_split{k}_{tr,val,te}.npz` files (two views per sample). Cite TCGA for the underlying data; for the exact view selection and splits, follow the protocol described in the [IDMVAE paper](https://openreview.net/forum?id=DcHGEcqdFf).

**PolyMNIST:** cite **Sutter et al. (2021, MoPoE)** — that paper introduces the dataset ([arXiv:2105.02470](https://arxiv.org/abs/2105.02470)). Shi et al. (2019, MMVAE) is related multimodal work but not the PolyMNIST dataset definition.

### BibTeX (datasets)

```bibtex
@article{lecun2010mnist,
  title   = {{MNIST} Handwritten Digit Database},
  author  = {LeCun, Yann and Cortes, Corinna and Burges, Christopher J. C.},
  journal = {ATT Labs},
  year    = {2010},
  note    = {Historic host http://yann.lecun.com/exdb/mnist/ ; use torchvision or https://github.com/cvdfoundation/mnist for files}
}

@article{sutter2021mopoe,
  title   = {Generalized Multimodal {ELBO}},
  author  = {Sutter, Thomas M. and Daunhawer, Imant and Vogt, Julia E.},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  number  = {202},
  pages   = {1--60},
  year    = {2021},
  url     = {https://arxiv.org/abs/2105.02470}
}

@techreport{wah2011cub,
  title       = {The {Caltech-UCSD} Birds-200-2011 Dataset},
  author      = {Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge},
  institution = {California Institute of Technology},
  year        = {2011},
  number      = {CNS-TR-2011-001},
  url         = {https://www.vision.caltech.edu/datasets/cub_200_2011/}
}

@inproceedings{reed2016learning,
  title     = {Learning Deep Representations of Fine-Grained Visual Descriptions},
  author    = {Reed, Scott and Akata, Zeynep and Yan, Lajanugen and Wang, Lajan and Reed, Scott and Yu, Honglak and Darrell, Trevor},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2016}
}

@inproceedings{palumbo2024cmvae,
  title     = {Deep Generative Clustering with Multimodal Diffusion Variational Autoencoders},
  author    = {Palumbo, Emanuele and Manduchi, Laura and Laguna, Sonia and Chopard, Daphn{\'e} and Vogt, Julia E.},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=k5THrhXDV3}
}
```

## Running experiments

Working directory: **`src/`** (repository root is the parent of `src/`).

| Dataset | Command script (from `src/`) |
|---------|------------------------------|
| PolyMNIST | `python` / shell: `commands/run_PolyMNIST_experiment_train_and_checkpoint.sh` |
| CUB 256px | `commands/run_CUB_experiment_train_and_checkpoint_256.sh` |
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
