# Train IDMVAE on CUB Image-Captions dataset
import os
# Deterministic behavior: 
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Set before importing torch
# os.environ["WANDB_MODE"] = "disabled"

import glob
import re
import shutil
import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import optim
from torchvision.utils import make_grid
import models
from utils import CrossModalEvalForwardMode
from objectives import compute_idmvae_loss
from utils import Logger, save_model_light
from utils import unpack_data_CUBcluster8, get_test_CUBcluster8_samples
import wandb
import textwrap
import torchvision.transforms as transforms

from eval_functions_CUB import (
    linear_latent_classification_CUB_multi_labelTypes,
    save_images_with_labels,
    save_recon_sequences_with_labels,
    train_clf_lr_CUB_multi_labelTypes,
)
from eval_functions import visualize_latents_with_priors
from eval_functions_CUB import (
    calculate_fid_routine,
    cub_self_and_cross_modal_generation_eval,
    cub_generate_unconditional,
)
from dataset_CUBcluster8 import CUBcluster8Dataset, CUBImageViewDataset, CUBCaptionViewDataset

import utils

from diffusers.models import AutoencoderKL

parser = argparse.ArgumentParser(description='IDMVAE Image-Captions')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--img_size_original', type=int, default=256,
                    help='Original image resolution, mainly for plotting matching, such as text->tensor')
parser.add_argument('--text2img_ratio', type=float, default=2.0,
                    help='Height ratio of text image to original image when plotting text as image tensor')
parser.add_argument('--fontsize', type=int, default=18,
                    help='Font size when plotting text as image tensor(8->64*64)')

parser.add_argument('--img_size', type=int, default=32,
                    help='Image resolution.')
parser.add_argument('--img_channels', type=int, default=4,
                    help='Image number of channels.')
parser.add_argument('--use_pretrain_feats', action='store_true', default=False,
                    help='Whether to use pretrained VAE features.')
parser.add_argument('--use_DiT_arch', action='store_true', default=False,
                    help='Whether to use DiT architecture for image encoder/decoder.')
parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
parser.add_argument('--patch_size', type=int, default=2,
                    help='Patch size for DiT-based encoder/decoder, higher->faster.')
parser.add_argument('--hidden_size', type=int, default=1024,
                    help='Hidden size for DiT-based encoder/decoder, smaller->faster.[mlp_dim=128]')
parser.add_argument('--depth', type=int, default=22,
                    help='Depth (number of layers) for DiT-based encoder/decoder, smaller->faster.')
parser.add_argument('--num_heads', type=int, default=16,
                    help='Number of attention heads for DiT-based encoder/decoder, hidden_size must be perfectly divisible by num_heads.')
parser.add_argument('--mlp_ratio', type=float, default=4.0,
                    help='MLP ratio for DiT-based encoder/decoder.')
parser.add_argument('--mlp_dim', type=int, default=64,
                    help='MLP dimension for DiT-based encoder/decoder, smaller->faster.')
parser.add_argument('--denoiser_ckpt', type=str, default=None,
                    help='Path to a pretrained DiT denoiser checkpoint (.pt).')
parser.add_argument('--denoiser_model', type=str, default='DiT-XL/2',
                    help='DiT architecture to use for the pretrained denoiser (e.g., DiT-XL/2).')
parser.add_argument('--denoiser_num_sampling_steps', type=int, default=250,
                    help='Number of diffusion sampling steps when running the pretrained denoiser.')
parser.add_argument('--denoiser_num_classes', type=int, default=1000,
                    help='Number of class labels the pretrained denoiser was trained with.')
parser.add_argument('--denoiser_class_label', type=int, default=None,
                    help='Optional fixed class label for the denoiser; defaults to unconditional.')

parser.add_argument('--K', type=int, default=1,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size for data')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (paper Appendix C.2: 50)')
parser.add_argument('--latent-dim-w', type=int, default=32, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--latent-dim-z', type=int, default=64, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--print-freq', type=int, default=50, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--llik_scaling_sent', type=float, default=5.0,
                    help='likelihood scaling factor sentences')
parser.add_argument('--datadir', type=str, default='./data',
                    help=' Directory where data is stored and samples used for FID calculation are saved')
parser.add_argument('--outputdir', type=str, default='../outputs',
                    help='Output directory')
parser.add_argument('--inception_path', type=str, default='/data/backed_up/shared/Data/CUB/pt_inception-2015-12-05-6726825d.pth',
                    help='Path to inception module for FID calculation')
parser.add_argument('--priorposterior', type=str, default='Laplace', choices=['Normal', 'Laplace', 'Diffusion'],
                    help='distribution choice for prior and posterior')
parser.add_argument('--diffusion_loss_weight', type=float, default=0.0,
                    help='loss weight for diffusion')
parser.add_argument('--diffusion_stop_grad_on_input', action='store_true', default=False,
                    help='stop gradient on x_start of diffusion')
parser.add_argument('--likelihood', type=str, default='Laplace', choices=['Normal', 'Laplace'],
                    help='distribution choice for likelihood')
parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'CUBICC', 'CUBcluster8', 'CUBcluster8_256'],
                    help='dataset choice')
parser.add_argument('--tSNE_save_dir', type=str, default='./data/tSNE_results',
                    help='Directory to save tSNE results')
parser.add_argument('--use_mean_in_latent_visualization', action='store_true', default=False,
                    help='Use mean in latent visualization for tSNE/UMAP instead of sampling')
parser.add_argument('--lv_umap_n_neighbors', type=int, default=20,
                    help='Number of neighbors for UMAP on latent variables')
parser.add_argument('--lv_umap_min_dist', type=float, default=0.05,
                    help='Minimum distance for UMAP on latent variables')
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for data loading')

parser.add_argument('--cross_mi_loss_scale', type=float, default=0.0, help='Scale for cross-view MI loss')
parser.add_argument('--gen_aug_loss_scale', type=float, default=0.0, help='Scale for generative augmentation loss')
parser.add_argument('--gen_aug_sampling_scheme', type=str, default='posterior', choices=['posterior', 'prior', 'diffusion_prior'],
                    help='Sampling scheme for generative augmentation loss: posterior, prior, or diffusion_prior')
parser.add_argument('--gen_aug_loss_type', type=str, default='CL', choices=['CL', 'ML'],
                    help='Type of generative augmentation loss: CL (Contrastive Loss) or ML (Matching Loss)')

parser.add_argument('--debug_log', action='store_true', help='Enable debug logging')
parser.add_argument('--debug_pdb', action='store_true', help='Enable debug mode with pdb')
# add date, time and note for runId
parser.add_argument('--date', type=str, default='', help='Date for runId')
parser.add_argument('--time', type=str, default='', help='Time for runId')
parser.add_argument('--note', type=str, default='', help='Note for runId')
parser.add_argument('--wandb_allow_step_reset', action='store_true', default=False,
                    help='If set, drop the explicit step argument when logging to wandb to avoid monotonic step errors when reloading older checkpoints within the same run.')

# develop mode
parser.add_argument('--develop', action='store_true', default=False,
                    help='If set, run in develop mode with a fresh run ID and output directory for quick debugging iteration. This will not overwrite existing runs but will create a new one with the same parameters.')
# Checkpoint loading
# Mode arguments
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume training from the last checkpoint in the run directory.')
parser.add_argument('--resume_from_CPt_runId', action='store_true', default=False,
                    help='Resume training from an old runId, specified by --CPt_runId.')
parser.add_argument('--CPt_runId', type=str, default='',
                    help='Run ID to resume from if --resume_from_CPt_runId is set. This should match the runId used in the original training run.')
# Test-only arguments
parser.add_argument('--test-only', action='store_true', default=False,
                    help='Load a checkpoint and run evaluation only, without training.')
parser.add_argument('--checkpoint-path', type=str, default='',
                    help='Path to a model checkpoint .rar file (required with --test-only).')
parser.add_argument('--print-params-only', action='store_true', default=False,
                    help='Instantiate model (optionally load a checkpoint) and print parameter counts, then exit.')

# Test time state
parser.add_argument(
    '--test_time_dataset_state',
    type=str,
    default='eval',
    choices=['train', 'eval', 'test'],
    help="Dataset split used for evaluation: one of {'train','eval','test'}.",
)

# Evaluation metrics
parser.add_argument('--enable_test_epoch', action='store_true', default=False,
                    help='Enable test epoch for evaluation.')
parser.add_argument('--enable_unconditional_generation', action='store_true', default=False,
                    help='Enable unconditional generation during testing.')
parser.add_argument('--enable_latent_classification', action='store_true', default=False,
                    help='Enable latent classification during testing.')
parser.add_argument('--use_mean_for_latent_clf', action='store_true', default=False,
                    help='Use mean for latent classification instead of sampling.')
parser.add_argument('--enable_fid', action='store_true', default=False,
                    help='Enable FID calculation during testing.')
parser.add_argument('--enable_tSNE_UMAP', action='store_true', default=False,
                    help='Enable tSNE and UMAP visualization during testing.')
parser.add_argument('--save_eval_images_root', type=str, default='',
                    help='If set, save original/recon/denoised images during evaluation under this root directory.')

# Bird direction evaluation (see dataset generation utilities in project history)
parser.add_argument('--degree_away_center_threshold', type=float, default=0.0,
                   help="For labels_direction_xx.pt selection. [Generate/Verify Mode] Angle in degrees away from center (vertical) to be considered 'left' or 'right' based on head/body alignment.")

# args
args = parser.parse_args()

# Validate arguments
if not args.print_params_only:
    if args.resume and args.test_only:
        parser.error("--resume and --test-only cannot be used together.")
    if args.test_only and not args.checkpoint_path:
        parser.error("When using --test-only, you must provide --checkpoint-path.")

    if args.checkpoint_path and not args.test_only:
        parser.error("--checkpoint-path can only be used with --test-only or --print-params-only.")
else:
    if args.resume or args.test_only:
        parser.error("--print-params-only cannot be combined with --resume or --test-only.")

if args.checkpoint_path:
    args.checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
if args.test_only and not os.path.isfile(args.checkpoint_path):
    parser.error(
        f"Checkpoint not found: {args.checkpoint_path}\n"
        f"  cwd: {os.getcwd()}\n"
        f"  Training checkpoints: <outputdir>/<--experiment>/checkpoints/<runId>/model_*.rar"
    )
if args.print_params_only and args.checkpoint_path and not os.path.isfile(args.checkpoint_path):
    parser.error(f"Checkpoint not found: {args.checkpoint_path}\n  cwd: {os.getcwd()}")

args.latent_dim_u = args.latent_dim_w + args.latent_dim_z

# Update debug flags
utils.DEBUG_ENABLED = args.debug_log
utils.PDB_ENABLED = args.debug_pdb

# Random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# CUDA stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

modelC = getattr(models, 'IDMVAE_CUB_Image_Captions')
model = modelC(args).to(device)

def _log_param_counts(tag: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params/CUB/{tag}] total={total:,} trainable={trainable:,}")

_log_param_counts("init")

if args.print_params_only:
    if args.checkpoint_path:
        print(f"Loading checkpoint for parameter count: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
        _log_param_counts("print_only_checkpoint")
    sys.exit(0)

# Pretrained VAE for image feature extraction
pretrained_vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

# Set experiment name if not set
if not args.experiment:
    args.experiment = model.modelName

# Set up run path
if args.resume and args.resume_from_CPt_runId:
    runId = args.CPt_runId
elif args.develop:
    runId = (
        f"Dev_{args.note}_K{args.K}_B{args.batch_size}_{args.priorposterior}_{args.likelihood}_b{args.beta}_"
        f"{args.gen_aug_loss_scale}_{args.cross_mi_loss_scale}_"
        f"{args.latent_dim_w}_{args.latent_dim_z}_"
        f"s{args.seed}"
    )
else:
    runId = (
        f"{args.note}_K{args.K}_B{args.batch_size}_{args.priorposterior}_{args.likelihood}_b{args.beta}_"
        f"{args.gen_aug_loss_scale}_{args.cross_mi_loss_scale}_"
        f"{args.latent_dim_w}_{args.latent_dim_z}_"
        f"s{args.seed}"
    )

if args.test_only:
    experiment_dir = Path(os.path.join(args.outputdir, args.experiment, "checkpoints", "CP_test"))
elif args.develop:
    experiment_dir = Path(os.path.join(args.outputdir, args.experiment, "checkpoints", "Dev"))
else:
    experiment_dir = Path(os.path.join(args.outputdir, args.experiment, "checkpoints"))
# experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = os.path.join(str(experiment_dir), runId)

# Optimizer
print("Using Adam optimizer.")
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, amsgrad=True)

# Checkpoint and resuming logic
start_epoch = 1
experiment_dir.mkdir(parents=True, exist_ok=True)

if args.resume:
    if not os.path.exists(runPath):
        print(f"Error: Run path '{runPath}' does not exist for resuming. Exiting.")
        sys.exit(1)

    checkpoints = glob.glob(os.path.join(runPath, 'model_*.rar'))
    if not checkpoints:
        print(f"Error: No checkpoints found in '{runPath}'. Starting from scratch.")
        wandb_resume_status = None
    else:
        latest_checkpoint = max(checkpoints, key=lambda p: int(re.search(r'model_(\d+).rar', p).group(1)))
        checkpoint_epoch = int(re.search(r'model_(\d+).rar', latest_checkpoint).group(1))
        
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        _log_param_counts(f"resume_epoch_{checkpoint_epoch}")
        
        optimizer_path = os.path.join(runPath, f'optimizer_{checkpoint_epoch}.rar')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print(f"Loaded optimizer state from epoch {checkpoint_epoch}.")

        start_epoch = checkpoint_epoch + 1
        wandb_resume_status = "allow"
        print(f"Resuming from epoch {start_epoch}...")

elif args.develop:
    if os.path.isdir(runPath):
        print(f"Warning: Run path '{runPath}' already exists. Removing it for a fresh develop run.")
        shutil.rmtree(runPath)
    else:
        print(f"Creating new run path for develop mode: '{runPath}'")    
    os.makedirs(runPath, exist_ok=True)
    wandb_resume_status = None # "allow"

elif not args.test_only and not args.develop:
    if os.path.exists(runPath):
        print(f"Error: Run path '{runPath}' already exists. Use --resume to continue or change parameters to start a new run.")
        sys.exit(1)
    
    os.makedirs(runPath)
    wandb_resume_status = None
    
    with open('{}/args.json'.format(runPath), 'w') as fp:
        json.dump(args.__dict__, fp)
    torch.save(args, '{}/args.rar'.format(runPath))

else: # Test-only mode
    wandb_resume_status = "allow"
    os.makedirs(runPath, exist_ok=True)

# Setup logging
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)

# === Set Parameters from args ===
# Set model regularization coefficients from args
model.params.cross_mi_loss_scale = args.cross_mi_loss_scale
model.params.gen_aug_loss_scale = args.gen_aug_loss_scale
model.params.gen_aug_sampling_scheme = args.gen_aug_sampling_scheme
model.params.gen_aug_loss_type = args.gen_aug_loss_type

NUM_VAES = len(model.vaes)

# Creat path where to temporarily save images to compute FID scores
fid_path = os.path.join(args.datadir, 'FIDs/fids_CUB_Image_Captions' + (runPath.rsplit('/')[-1]))
datadirCUB = os.path.join(args.datadir, "CUB_Image_Captions")  # Retained for downstream utilities.

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# WandB

# NOTE: wandb_step_reset_enabled support remains partial; keep explicit run-id behavior unchanged.

wandb.login()

wandb_kwargs = dict(
    project=args.experiment,
    config=vars(args),
    name=runId,
    id=runId,
)

if wandb_resume_status is not None:
    wandb_kwargs["resume"] = wandb_resume_status

wandb.init(**wandb_kwargs)
# WandB

num_workers = args.num_workers if hasattr(args, 'num_workers') else 32

# Deterministic behavior
kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == 'cuda' else {}
g = torch.Generator()
g.manual_seed(0)
kwargs['generator'] = g

# Load CUB Image-Captions, currently focus on CUBcluster8
print('Loading CUB Image-Captions dataset: {}'.format(args.dataset))
if args.dataset == 'CUBcluster8' or args.dataset == 'CUBcluster8_256':
    
    base_dir = args.datadir #CUBcluster8

    augmentation_transform = transforms.Compose([
        # Randomly flip horizontally with a 50% probability
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    # Temporary. If this is true, loaded image tensors should be of size [4, 32, 32]. If false, [3, 64, 64].
    use_pretrain_feats=args.use_pretrain_feats
    if use_pretrain_feats:
        assert args.img_size == 32
        assert args.img_channels == 4
    else:
        assert args.img_size == 64
        assert args.img_channels == 3

    # Full train set (clusters + Other)
    train_dataset = CUBcluster8Dataset(
        datadir=base_dir,
        split='train',
        cluster_only=False,
        transform=augmentation_transform,  # None or augmentation_transform
        use_pretrain_feats=use_pretrain_feats,
        args=args  # Pass args for additional configurations
    )
    # Cluster-only train set (8 clusters without Other)
    train_cluster_dataset = CUBcluster8Dataset(
        datadir=base_dir,
        split='train',
        cluster_only=True,
        transform=None,
        use_pretrain_feats=use_pretrain_feats,
        args=args  # Pass args for additional configurations
    )
    # Validation set (clusters only)
    val_cluster_dataset = CUBcluster8Dataset(
        datadir=base_dir,
        split='val',
        transform=None,
        use_pretrain_feats=use_pretrain_feats,
        args=args  # Pass args for additional configurations
    )
    # Test set (clusters only)
    test_cluster_dataset = CUBcluster8Dataset(
        datadir=base_dir,
        split='test',
        transform=None,
        use_pretrain_feats=use_pretrain_feats,
        args=args  # Pass args for additional configurations
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    train_cluster_loader = torch.utils.data.DataLoader(
        train_cluster_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    val_cluster_loader = torch.utils.data.DataLoader(
        val_cluster_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
    test_cluster_loader = torch.utils.data.DataLoader(
        test_cluster_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    # Select dataset/dataloader for evaluation.
    if args.test_time_dataset_state == "train":  # reconstruction verification: better->dataset is small, no change->architecture is weak
        test_time_dataset = train_cluster_dataset
        test_time_loader = train_cluster_loader
    if args.test_time_dataset_state == "eval":
        test_time_dataset = val_cluster_dataset
        test_time_loader = val_cluster_loader
    if args.test_time_dataset_state == "test":
        test_time_dataset = test_cluster_dataset
        test_time_loader = test_cluster_loader

    # Create wrapper datasets for validation and testing
    # For Image View (View 0)
    latent_visualization_dataset = test_time_dataset  # , train_cluster_dataset
    test_time_image_dataset = CUBImageViewDataset(latent_visualization_dataset)
    test_time_image_loader = torch.utils.data.DataLoader(
        test_time_image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    # For Caption View (View 1)
    test_time_caption_dataset = CUBCaptionViewDataset(latent_visualization_dataset)
    test_time_caption_loader = torch.utils.data.DataLoader(
        test_time_caption_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

else:
    raise ValueError(f"Unknown dataset '{args.dataset}'")


def train(epoch):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        # CUBICC:
        data, label = unpack_data_CUBcluster8(dataT, device=device)
        optimizer.zero_grad()

        bs = data[0].size(0)

        loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(model, data, K=args.K)

        wandb.log({"Loss/train_loss": loss}, step=epoch)
        wandb.log({"Loss/train_recon_kl_sum": recon_kl_sum_loss}, step=epoch)
        wandb.log({"Loss/train_likelihood": llik_recon_loss}, step=epoch)
        wandb.log({"Loss/train_kl": kl_div_loss}, step=epoch)
        wandb.log({"Loss/train_cross_mi": cross_mi_loss}, step=epoch)
        wandb.log({"Loss/train_gen_aug": gen_aug_loss}, step=epoch)
        wandb.log({"Loss/train_diffusion_loss": diffusion_loss.item()}, step=epoch)

        loss.backward()
        optimizer.step()
        b_loss += loss.item() * bs
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item())) # / args.batch_size))
    # Epoch loss
    epoch_loss = b_loss / len(train_loader.dataset)
    wandb.log({"Loss/train": epoch_loss}, step=epoch)
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))


def test(epoch):
    """Test-time loss on the test loader only."""
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for _, dataT in enumerate(test_time_loader):
            data, _ = unpack_data_CUBcluster8(dataT, device=device)
            bs = data[0].size(0)
            loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(
                model, data, K=args.K, test=True
            )
            wandb.log({"Loss/test_loss": loss}, step=epoch)
            wandb.log({"Loss/test_recon_kl_sum": recon_kl_sum_loss}, step=epoch)
            wandb.log({"Loss/test_likelihood": llik_recon_loss}, step=epoch)
            wandb.log({"Loss/test_kl": kl_div_loss}, step=epoch)
            wandb.log({"Loss/test_cross_mi": cross_mi_loss}, step=epoch)
            wandb.log({"Loss/test_gen_aug": gen_aug_loss}, step=epoch)
            wandb.log({"Loss/test_diffusion_loss": diffusion_loss.item()}, step=epoch)
            b_loss += loss.item() * bs
    epoch_loss = b_loss / len(test_time_loader.dataset)
    wandb.log({"Loss/test": epoch_loss}, step=epoch)
    print('====>             Test loss: {:.4f}'.format(epoch_loss))


def _cub_test_epoch_qualitative_visuals(epoch):
    """Qualitative test samples and cross-modal wandb grids (after test loss)."""
    model.eval()
    with torch.no_grad():
        # randomly select samples for qualitative examples
        test_selected_samples, labels_list_of_tuples, \
            raw_selected_captions, raw_all_captions = \
                get_test_CUBcluster8_samples(test_time_loader.dataset,
                                                     num_testing_images=test_time_loader.dataset.__len__(), 
                                                     device=device, args=args, pretrained_vae=pretrained_vae)
        # Or use fixed samples by index or image ID for reproducibility
        # test_selected_samples, labels_list_of_tuples, \
        #     raw_selected_captions, raw_all_captions = \
        #         get_and_log_CUBcluster8_samples_by_Idx_or_ID(
        #             test_time_loader.dataset,
        #             num_testing_images=test_time_loader.dataset.__len__(),
        #             device=device, args=args)

        print("Selected qualitative test samples: {}".format(len(test_selected_samples[0])))

        # === START: Plot selected test samples to WandB ===
        images_to_log = []
        direction_map = {0: "Left", 1: "Right", 88: "Center", 98: "Close-L", 99: "Close-R", 404: "N/A"}
        i2w = test_time_loader.dataset.i2w

        for i in range(len(test_selected_samples[0])):
            image = test_selected_samples[0][i]
            cap_tensor = test_selected_samples[1][i]
            labels = labels_list_of_tuples[i]

            # Convert caption tensor back to text
            indices = torch.argmax(cap_tensor, dim=-1).cpu().numpy()
            words = [i2w.get(str(idx), '<unk>') for idx in indices]
            raw_caption_i2w = ' '.join(word for word in words if word not in ['<pad>', '<eos>'])
            raw_caption_pt = raw_selected_captions[i]

            # Wrap the raw caption text for better display
            wrapped_caption_i2w = '\n'.join(textwrap.wrap(raw_caption_i2w, width=72))
            wrapped_caption_pt = '\n'.join(textwrap.wrap(raw_caption_pt, width=72))

            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index = labels

            dir_label_str = direction_map.get(int(lbl_dir), "Unk")

            caption_log = (f"Sample: {i+1} | Index: {dataset_index} | ImgID: {img_id}\n"
                           f"Category: {lbl_cat} | Cluster: {lbl_cluster} | Direction: {dir_label_str}({lbl_dir})\n"
                           f"Caption_i2w: {wrapped_caption_i2w}\n"
                           f"Caption_.pt: {wrapped_caption_pt}"
                           )

            images_to_log.append(wandb.Image(image, caption=caption_log))

        if images_to_log:
            wandb.log({"Samples_Display/Selected_Test_Samples": images_to_log}, step=epoch)
        # --- END: Plot samples---
        # Grid configuration parameters:
        num_cols = len(test_selected_samples[0])  # 8 # Number of columns in the grid
        num_rows = 10 # len(test_selected_samples[0])  # make square, or 10 # Number of generated_samples of each selected_sample. Number of rows in the grid (generations part)
        num_gen_samples = min(num_rows, 3) # single column to fit the paper space
        group_grid_nrow_with_input = 1  # vertical: 1 (default), horizontal: num_gen_samples + 1
        group_grid_nrow_without_input = 1  # vertical: 1 (default), horizontal: num_gen_samples
        # rsample for priors
        # Shared: cluster
        cg_imgs_cluster, input_data_cluster, recon_triess_to_table_cluster = \
            cub_self_and_cross_modal_generation_eval(model,
                test_selected_samples, num_cols, num_rows,
                mode=CrossModalEvalForwardMode.PRIOR_CTRL,
                condition_type='shared')
        cg_imgs_cluster_denoised = getattr(model, 'last_denoised_prior_grids', None)
        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                wandb.log({'IMG_Self&Cros_Gen_Cluster_prior/qz_m{}+pw_m{}'.format(i, j): wandb.Image(cg_imgs_cluster[i][j])}, step=epoch)
                if cg_imgs_cluster_denoised and cg_imgs_cluster_denoised[i][j] is not None:
                    wandb.log({'IMG_Self&Cros_Gen_Cluster_prior/qz_m{}+pw_m{}_denoised'.format(i, j): wandb.Image(cg_imgs_cluster_denoised[i][j])}, step=epoch)

        # plot cluster generations separately
        """
        "Img->Img", "Img->Cap", "Cap->Img", "Cap->Cap"
        separately instead of table so far, 
        and in each case, the sample image, caption, and corresponding labels should be displayed individually. 
        case "Img->Img":
            just display the generated image and their corresponding labels.
        case "Img->Cap":
            display the input image and its generated caption, and their corresponding labels.
        case "Cap->Img":
            display its generated image and the input caption, and their corresponding labels.
        case "Cap->Cap":
            display the corresponding image and the input caption, and the generated caption and their corresponding labels.
        all the input or generated captions and the labels can log in the wandb.caption
        """
        input_images_latent = test_selected_samples[0]
        if args.use_pretrain_feats:
            """cannot do sth like:
            input_images[n] = pretrained_vae.decode(
            RuntimeError: The expanded size of the tensor (32) must match the existing size (256) at non-singleton dimension 2.  Target sizes: [4, 32, 32].  Tensor sizes: [3, 256, 256]
            """
            vae_device = next(pretrained_vae.parameters()).device
            decoded_images = []
            for n in range(num_cols):
                # input_images_latent[n]: [4, 32, 32]
                # convert to [3, 256, 256] for better visualization
                latent = (input_images_latent[n].unsqueeze(0) / 0.18215).to(vae_device)
                decoded = pretrained_vae.decode(latent).sample.squeeze(0)
                decoded = decoded.add(1).div(2).clamp(0, 1)
                decoded_images.append(decoded.cpu())
            input_images = torch.stack(decoded_images, dim=0)
        else:
            input_images = input_images_latent.cpu()

        input_caption_tensor = test_selected_samples[1]
        if args.save_eval_images_root:
            labels_tensor = (
                torch.tensor([lbl[0] for lbl in labels_list_of_tuples]),
                torch.tensor([lbl[1] if lbl[1] is not None else -1 for lbl in labels_list_of_tuples]),
                torch.tensor([lbl[2] for lbl in labels_list_of_tuples]),
                torch.tensor([lbl[3] for lbl in labels_list_of_tuples]),
                torch.tensor([lbl[4] for lbl in labels_list_of_tuples]),
            )
            save_root = os.path.join(args.save_eval_images_root, f"epoch_{epoch}", "prior_shared")
            save_images_with_labels(input_images, labels_tensor, os.path.join(save_root, "orig"), prefix="orig_shared")
            save_recon_sequences_with_labels(recon_triess_to_table_cluster[0][0], labels_tensor, os.path.join(save_root, "noisy"), prefix="noisy_shared")
            den_entries = getattr(model, 'last_denoised_prior_entries', None)
            if den_entries and den_entries[0][0]:
                save_recon_sequences_with_labels(den_entries[0][0], labels_tensor, os.path.join(save_root, "denoised"), prefix="denoised_shared")

        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                if i == 0 and j == 0: # Img->Img
                    # Log Img->Img generations
                    images_to_log_00_left = []
                    images_to_log_00_right = []
                    img_gen_samples_groups = [[] for i in range(num_cols)]
                    img_gen_samples_groups_grids_with_input = []
                    img_gen_samples_groups_grids_without_input = []
                    for m in range(num_rows): # N @ idmvae_CUB.py
                        for n in range(num_cols): # num (B) @ idmvae_CUB.py
                            input_img = input_images[n]
                            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                                = labels_list_of_tuples[n]
                            dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                            generated_img = recon_triess_to_table_cluster[i][j][m][n] # recon_: [M][M][N][B,C,H,W]
                            caption_log = (f"[Img->Img]: generated_img + labels info\n"
                                        f"Sample: ({m+1},{n+1}) | Index: {dataset_index} | ImgID: {img_id}\n"
                                        f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                        f"Direction: {dir_label_str}({lbl_dir})\n"
                                        #    f"Input Image: {input_img}\n" # NG: will log the pixel values in number
                                        )

                            if lbl_dir == 0:
                                images_to_log_00_left.append(wandb.Image(generated_img, caption=caption_log))
                            else:
                                images_to_log_00_right.append(wandb.Image(generated_img, caption=caption_log))
                            img_gen_samples_groups[n].append(generated_img)

                    # make grid for the samples group                                
                    for n in range(num_cols):
                        lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                            = labels_list_of_tuples[n]
                        dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                        caption_log_common = (f"Sample: {n+1} | Index: {dataset_index} | ImgID: {img_id}\n"
                                            f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                            f"Direction: {dir_label_str}({lbl_dir})\n"
                                            )
                        caption_log_with_input = (f"[Img->Img]: Input Image #{n+1} + {num_gen_samples} generations for it\n"
                                                    f"{caption_log_common}"
                                                )
                        caption_log_without_input = (f"[Img->Img]: {num_gen_samples} generations for Input Image #{n+1}\n"
                                                    f"{caption_log_common}"
                                                )
                        img_gen_samples_groups_grids_with_input.append(
                            wandb.Image(
                                make_grid(
                                    [input_images[n].cpu()] + img_gen_samples_groups[n][:num_gen_samples],
                                    nrow=group_grid_nrow_with_input,
                                    normalize=True,
                                    scale_each=True
                                ),
                                caption=caption_log_with_input
                            )
                        )
                        img_gen_samples_groups_grids_without_input.append(
                            wandb.Image(
                                make_grid(
                                    img_gen_samples_groups[n][:num_gen_samples], 
                                    nrow=group_grid_nrow_without_input, 
                                    normalize=True, 
                                    scale_each=True
                                ),
                                caption=caption_log_without_input
                            )
                        )

                    # log m x n samples
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Left/Img->Img|qz_m{}+pw_m{}".format(i, j): 
                            images_to_log_00_left}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Right/Img->Img|qz_m{}+pw_m{}".format(i, j): 
                            images_to_log_00_right}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Group_w_Input/Img->Img|qz_m{}+pw_m{}".format(i, j): 
                            img_gen_samples_groups_grids_with_input}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Group_wo_Input/Img->Img|qz_m{}+pw_m{}".format(i, j): 
                            img_gen_samples_groups_grids_without_input}, step=epoch)

                if i == 0 and j == 1:  # Img->Cap
                    # Log Img->Cap generations
                    captions_to_log_01_left = []
                    captions_to_log_01_right = []
                    for m in range(num_rows): # 3 (make_grid) or num_rows
                        for n in range(num_cols):
                            input_img = input_images[n]
                            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                                = labels_list_of_tuples[n]
                            dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                            generated_caption = recon_triess_to_table_cluster[i][j][m][n]
                            # Convert caption tensor back to text
                            indices = torch.argmax(generated_caption, dim=-1).cpu().numpy()
                            words = [i2w.get(str(idx), '<unk>') for idx in indices]
                            gen_caption_i2w = ' '.join(word for word in words if word not in ['<pad>', '<eos>'])
                            # Wrap the raw caption text for better display
                            wrapped_gen_caption_i2w = '\n'.join(textwrap.wrap(gen_caption_i2w, width=66))
                            caption_log = (f"[Img->Cap]: input image + generated_caption + labels info\n"
                                        f"Sample: ({m+1},{n+1}) | Index: {dataset_index} | ImgID: {img_id}\n"
                                        f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                        f"Direction: {dir_label_str}({lbl_dir})\n"
                                        f"Generated Caption: {wrapped_gen_caption_i2w}\n"
                                        #    f"Input Image: {input_img}\n"
                                        )

                            if lbl_dir == 0:
                                captions_to_log_01_left.append(wandb.Image(input_img, caption=caption_log))
                            else:
                                captions_to_log_01_right.append(wandb.Image(input_img, caption=caption_log))

                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Left/Img->Cap|qz_m{}+pw_m{}".format(i, j):
                            captions_to_log_01_left}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Right/Img->Cap|qz_m{}+pw_m{}".format(i, j):
                            captions_to_log_01_right}, step=epoch)

                if i == 1 and j == 0: # Cap->Img
                    # Log Cap->Img generations
                    images_to_log_10_left = []
                    images_to_log_10_right = []
                    img_gen_samples_groups = [[] for i in range(num_cols)]
                    img_gen_samples_groups_grids_with_input = []
                    img_gen_samples_groups_grids_without_input = []
                    for m in range(num_rows):
                        for n in range(num_cols):
                            input_caption = input_caption_tensor[n]
                            indices = torch.argmax(input_caption, dim=-1).cpu().numpy()
                            words = [i2w.get(str(idx), '<unk>') for idx in indices]
                            input_caption_i2w = ' '.join(word for word in words if word not in ['<pad>', '<eos>'])
                            input_caption_pt = raw_selected_captions[n]
                            wrapped_caption_i2w = '\n'.join(textwrap.wrap(input_caption_i2w, width=66))
                            wrapped_caption_pt = '\n'.join(textwrap.wrap(input_caption_pt, width=66))

                            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                                = labels_list_of_tuples[n]
                            dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                            generated_image = recon_triess_to_table_cluster[i][j][m][n]
                            # # Convert image tensor back to PIL Image
                            # gen_image = transforms.ToPILImage()(generated_image.cpu().detach())
                            caption_log = (f"[Cap->Img]: generated_image + input_caption + labels info\n"
                                        f"Sample: ({m+1},{n+1}) | Index: {dataset_index} | ImgID: {img_id}\n"
                                        f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                        f"Direction: {dir_label_str}({lbl_dir})\n"
                                        f"Input Caption i2w: {wrapped_caption_i2w}\n"
                                        f"Input Caption .pt: {wrapped_caption_pt}\n"
                                        #    f"Generated Image: {generated_image}\n"
                                        )

                            if lbl_dir == 0:
                                images_to_log_10_left.append(wandb.Image(generated_image, caption=caption_log))
                            else:
                                images_to_log_10_right.append(wandb.Image(generated_image, caption=caption_log))
                            img_gen_samples_groups[n].append(generated_image)

                    for n in range(num_cols):
                        input_caption = input_caption_tensor[n]
                        indices = torch.argmax(input_caption, dim=-1).cpu().numpy()
                        words = [i2w.get(str(idx), '<unk>') for idx in indices]
                        input_caption_i2w = ' '.join(word for word in words if word not in ['<pad>', '<eos>'])
                        input_caption_pt = raw_selected_captions[n]
                        wrapped_caption_i2w = '\n'.join(textwrap.wrap(input_caption_i2w, width=66))
                        wrapped_caption_pt = '\n'.join(textwrap.wrap(input_caption_pt, width=66))
                        lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                            = labels_list_of_tuples[n]
                        dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                        caption_log_common = (f"Sample: {n+1} | Index: {dataset_index} | ImgID: {img_id}\n"
                                            f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                            f"Direction: {dir_label_str}({lbl_dir})\n"
                                            f"Input Caption i2w: {wrapped_caption_i2w}\n"
                                            f"Input Caption .pt: {wrapped_caption_pt}\n"
                                            )
                        caption_log_with_input = (f"[Cap->Img]: Input Caption #{n+1} + {num_gen_samples} generations for it\n"
                                                    f"{caption_log_common}"
                                                )
                        caption_log_without_input = (f"[Cap->Img]: {num_gen_samples} generations for Input Caption {n+1}\n"
                                                    f"{caption_log_common}"
                                                )
                        img_gen_samples_groups_grids_with_input.append(
                            wandb.Image(
                                make_grid(
                                    [input_images[n].cpu()] + img_gen_samples_groups[n][:num_gen_samples],
                                    nrow=group_grid_nrow_with_input,
                                    normalize=True,
                                    scale_each=True
                                ),
                                caption=caption_log_with_input
                            )
                        )
                        img_gen_samples_groups_grids_without_input.append(
                            wandb.Image(
                                make_grid(
                                    img_gen_samples_groups[n][:num_gen_samples], 
                                    nrow=group_grid_nrow_without_input, 
                                    normalize=True, 
                                    scale_each=True
                                ),
                                caption=caption_log_without_input
                            )
                        )

                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Left/Cap->Img|qz_m{}+pw_m{}".format(i, j):
                            images_to_log_10_left}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Right/Cap->Img|qz_m{}+pw_m{}".format(i, j):
                            images_to_log_10_right}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Group_w_Input/Cap->Img|qz_m{}+pw_m{}".format(i, j):
                            img_gen_samples_groups_grids_with_input}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Group_wo_Input/Cap->Img|qz_m{}+pw_m{}".format(i, j):
                            img_gen_samples_groups_grids_without_input}, step=epoch)

                if i == 1 and j == 1: # Cap->Cap
                    # Log Cap->Cap generations
                    captions_to_log_11_left = []
                    captions_to_log_11_right = []
                    for m in range(num_rows): # 3 (make_grid) or num_rows
                        for n in range(num_cols): # 3 (make_grid) or num_cols
                            input_img = input_images[n]
                            input_caption = input_caption_tensor[n]
                            indices = torch.argmax(input_caption, dim=-1).cpu().numpy()
                            words = [i2w.get(str(idx), '<unk>') for idx in indices]
                            input_caption_i2w = ' '.join(word for word in words if word not in ['<pad>', '<eos>'])
                            input_caption_pt = raw_selected_captions[n]
                            wrapped_caption_i2w = '\n'.join(textwrap.wrap(input_caption_i2w, width=66))
                            wrapped_caption_pt = '\n'.join(textwrap.wrap(input_caption_pt, width=66))

                            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                                = labels_list_of_tuples[n]
                            dir_label_str = direction_map.get(int(lbl_dir), "Unk")

                            generated_caption = recon_triess_to_table_cluster[i][j][m][n]
                            gen_indices = torch.argmax(generated_caption, dim=-1).cpu().numpy()
                            gen_words = [i2w.get(str(idx), '<unk>') for idx in gen_indices]
                            gen_caption_i2w = ' '.join(word for word in gen_words if word not in ['<pad>', '<eos>'])
                            wrapped_gen_caption_i2w = '\n'.join(textwrap.wrap(gen_caption_i2w, width=66))
                            caption_log = (f"[Cap->Cap]: corresp_image + input&gen_captions + labels info\n"
                                        f"Sample: ({m+1},{n+1}) | Index: {dataset_index} | ImgID: {img_id}\n"
                                        f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                        f"Direction: {dir_label_str}({lbl_dir})\n"
                                        f"Input Caption i2w: {wrapped_caption_i2w}\n"
                                        f"Input Caption .pt: {wrapped_caption_pt}\n"
                                        f"Generated Caption: {wrapped_gen_caption_i2w}\n"
                                        #    f"Generated Image: {generated_image}\n"
                                        )

                            if lbl_dir == 0:
                                captions_to_log_11_left.append(wandb.Image(input_img, caption=caption_log))
                            else:
                                captions_to_log_11_right.append(wandb.Image(input_img, caption=caption_log))
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Left/Cap->Cap|qz_m{}+pw_m{}".format(i, j):
                            captions_to_log_11_left}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Cluster_prior_seperated/Single_Right/Cap->Cap|qz_m{}+pw_m{}".format(i, j):
                            captions_to_log_11_right}, step=epoch)

        # Private: direction
        cg_imgs_direction, input_data_direction, recon_triess_to_table_direction = \
            cub_self_and_cross_modal_generation_eval(model,
                test_selected_samples, num_cols, num_rows,
                mode=CrossModalEvalForwardMode.PRIOR_CTRL,
                condition_type='private')
        cg_imgs_direction_denoised = getattr(model, 'last_denoised_prior_grids', None)
        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                if i == j:
                    wandb.log({'IMG_Self_Gen_Direction_prior/qw_m{}+pz_m{}'.format(i, j): wandb.Image(cg_imgs_direction[i][j])}, step=epoch)
                    if cg_imgs_direction_denoised and cg_imgs_direction_denoised[i][j] is not None:
                        wandb.log({'IMG_Self_Gen_Direction_prior/qw_m{}+pz_m{}_denoised'.format(i, j): wandb.Image(cg_imgs_direction_denoised[i][j])}, step=epoch)

        # plot direction generations separately
        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                if i == 0 and j == 0: # Img->Img
                    # Log Img->Img generations
                    images_to_log_00_left = []
                    images_to_log_00_right = []
                    img_gen_samples_groups = [[] for i in range(num_cols)]
                    img_gen_samples_groups_grids_with_input = []
                    img_gen_samples_groups_grids_without_input = []
                    for m in range(num_rows):
                        for n in range(num_cols):
                            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                                = labels_list_of_tuples[n]
                            dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                            generated_img = recon_triess_to_table_direction[i][j][m][n]
                            caption_log = (f"[Img->Img]: generated_img + labels info\n"
                                        f"Sample: ({m+1},{n+1}) | Index: {dataset_index} | ImgID: {img_id}\n"
                                        f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                        f"Direction: {dir_label_str}({lbl_dir})\n"
                                        #    f"Generated Image: {gen_img}\n"
                                        )

                            if lbl_dir == 0:
                                images_to_log_00_left.append(wandb.Image(generated_img, caption=caption_log))
                            else:
                                images_to_log_00_right.append(wandb.Image(generated_img, caption=caption_log))
                            img_gen_samples_groups[n].append(generated_img)

                    for n in range(num_cols):
                        lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                            = labels_list_of_tuples[n]
                        dir_label_str = direction_map.get(int(lbl_dir), "Unk")
                        caption_log_common = (f"Sample: {n+1} | Index: {dataset_index} | ImgID: {img_id}\n"
                                            f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                            f"Direction: {dir_label_str}({lbl_dir})\n"
                                            )
                        caption_log_with_input = (f"[Img->Img]: Input Image #{n+1} + {num_gen_samples} generations for it\n"
                                                    f"{caption_log_common}"
                                                )
                        caption_log_without_input = (f"[Img->Img]: {num_gen_samples} generations for Input Image #{n+1}\n"
                                                    f"{caption_log_common}"
                                                )
                        img_gen_samples_groups_grids_with_input.append(
                            wandb.Image(
                                make_grid(
                                    [input_images[n].cpu()] + img_gen_samples_groups[n][:num_gen_samples],
                                    nrow=group_grid_nrow_with_input,
                                    normalize=True,
                                    scale_each=True
                                ),
                                caption=caption_log_with_input
                            )
                        )
                        img_gen_samples_groups_grids_without_input.append(
                            wandb.Image(
                                make_grid(
                                    img_gen_samples_groups[n][:num_gen_samples], 
                                    nrow=group_grid_nrow_without_input, 
                                    normalize=True, 
                                    scale_each=True
                                ),
                                caption=caption_log_without_input
                            )
                        )

                    wandb.log(
                        {"IMG_Gen_Direction_prior_seperated/Single_Left/Img->Img|qw_m{}+pz_m{}".format(i, j): 
                            images_to_log_00_left}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Direction_prior_seperated/Single_Right/Img->Img|qw_m{}+pz_m{}".format(i, j): 
                            images_to_log_00_right}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Direction_prior_seperated/Group_w_Input/Img->Img|qw_m{}+pz_m{}".format(i, j): 
                            img_gen_samples_groups_grids_with_input}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Direction_prior_seperated/Group_wo_Input/Img->Img|qw_m{}+pz_m{}".format(i, j): 
                            img_gen_samples_groups_grids_without_input}, step=epoch)

                if i == 1 and j == 1: # Cap->Cap
                    # Log Cap->Cap generations
                    captions_to_log_11_left = []
                    captions_to_log_11_right = []
                    for m in range(num_rows): # 3 (make_grid) or num_rows
                        for n in range(num_cols):
                            input_img = input_images[n]
                            input_caption = input_caption_tensor[n]
                            indices = torch.argmax(input_caption, dim=-1).cpu().numpy()
                            words = [i2w.get(str(idx), '<unk>') for idx in indices]
                            input_caption_i2w = ' '.join(word for word in words if word not in ['<pad>', '<eos>'])
                            input_caption_pt = raw_selected_captions[n]
                            wrapped_caption_i2w = '\n'.join(textwrap.wrap(input_caption_i2w, width=66))
                            wrapped_caption_pt = '\n'.join(textwrap.wrap(input_caption_pt, width=66))

                            lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index \
                                = labels_list_of_tuples[n]
                            dir_label_str = direction_map.get(int(lbl_dir), "Unk")

                            generated_caption = recon_triess_to_table_direction[i][j][m][n]
                            gen_indices = torch.argmax(generated_caption, dim=-1).cpu().numpy()
                            gen_words = [i2w.get(str(idx), '<unk>') for idx in gen_indices]
                            gen_caption_i2w = ' '.join(word for word in gen_words if word not in ['<pad>', '<eos>'])
                            wrapped_gen_caption_i2w = '\n'.join(textwrap.wrap(gen_caption_i2w, width=66))
                            caption_log = (f"[Cap->Cap]: corresp_image + input&gen_captions + labels info\n"
                                        f"Sample: ({m+1},{n+1}) | Index: {dataset_index} | ImgID: {img_id}\n"
                                        f"Category: {lbl_cat} | Cluster: {lbl_cluster} | "
                                        f"Direction: {dir_label_str}({lbl_dir})\n"
                                        f"Input Caption i2w: {wrapped_caption_i2w}\n"
                                        f"Input Caption .pt: {wrapped_caption_pt}\n"
                                        f"Generated Caption: {wrapped_gen_caption_i2w}\n"
                                        #    f"Generated Image: {generated_image}\n"
                                        )

                            if lbl_dir == 0:
                                captions_to_log_11_left.append(wandb.Image(input_img, caption=caption_log))
                            else:
                                captions_to_log_11_right.append(wandb.Image(input_img, caption=caption_log))
                    wandb.log(
                        {"IMG_Gen_Direction_prior_seperated/Single_Left/Cap->Cap|qw_m{}+pz_m{}".format(i, j):
                            captions_to_log_11_left}, step=epoch)
                    wandb.log(
                        {"IMG_Gen_Direction_prior_seperated/Single_Right/Cap->Cap|qw_m{}+pz_m{}".format(i, j):
                            captions_to_log_11_right}, step=epoch)
        if args.save_eval_images_root:
            save_root_private = os.path.join(args.save_eval_images_root, f"epoch_{epoch}", "prior_private")
            save_images_with_labels(input_images, labels_tensor, os.path.join(save_root_private, "orig"), prefix="orig_private")
            save_recon_sequences_with_labels(recon_triess_to_table_direction[0][0], labels_tensor, os.path.join(save_root_private, "noisy"), prefix="noisy_private")
            den_entries_priv = getattr(model, 'last_denoised_prior_entries', None)
            if den_entries_priv and den_entries_priv[0][0]:
                save_recon_sequences_with_labels(den_entries_priv[0][0], labels_tensor, os.path.join(save_root_private, "denoised"), prefix="denoised_private")

        # rsample for posteriors, shuffled
        # Current call uses num=8 and N=8, bounded by selected test samples.
        cg_imgs_Shared_post, cg_imgs_Shared_post_ext, cg_imgs_Shared_post_ext_denoised = cub_self_and_cross_modal_generation_eval(model,
            test_selected_samples, 8, 8, mode=CrossModalEvalForwardMode.POSTERIOR_CTRL, condition_type='shared')
        cg_imgs_Shared_post_denoised = getattr(model, 'last_denoised_posterior_grids', None)
        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                wandb.log({'IMG_Self&Cros_Gen_Cluster_post/qz_m{}+qw_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Shared_post[i][j])}, step=epoch)
                if cg_imgs_Shared_post_denoised and cg_imgs_Shared_post_denoised[i][j] is not None:
                    wandb.log({'IMG_Self&Cros_Gen_Cluster_post/qz_m{}+qw_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Shared_post_denoised[i][j])}, step=epoch)
                if cg_imgs_Shared_post_ext and cg_imgs_Shared_post_ext[i][j] is not None:
                    wandb.log({'IMG_Self&Cros_Gen_Cluster_post_extended/qz_m{}+qw_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Shared_post_ext[i][j])}, step=epoch)
                if cg_imgs_Shared_post_ext_denoised and cg_imgs_Shared_post_ext_denoised[i][j] is not None:
                    wandb.log({'IMG_Self&Cros_Gen_Cluster_post_extended/qz_m{}+qw_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Shared_post_ext_denoised[i][j])}, step=epoch)

        cg_imgs_Private_post, cg_imgs_Private_post_ext, cg_imgs_Private_post_ext_denoised = cub_self_and_cross_modal_generation_eval(model,
            test_selected_samples, 8, 8, mode=CrossModalEvalForwardMode.POSTERIOR_CTRL, condition_type='private')
        cg_imgs_Private_post_denoised = getattr(model, 'last_denoised_posterior_grids', None)
        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                if i == j:
                    wandb.log({'IMG_Self_Gen_Direction_post/qw_m{}+qz_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Private_post[i][j])}, step=epoch)
                    if cg_imgs_Private_post_denoised and cg_imgs_Private_post_denoised[i][j] is not None:
                        wandb.log({'IMG_Self_Gen_Direction_post/qw_m{}+qz_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Private_post_denoised[i][j])}, step=epoch)
                    if cg_imgs_Private_post_ext and cg_imgs_Private_post_ext[i][j] is not None:
                        wandb.log({'IMG_Self_Gen_Direction_post_extended/qw_m{}+qz_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Private_post_ext[i][j])}, step=epoch)
                    if cg_imgs_Private_post_ext_denoised and cg_imgs_Private_post_ext_denoised[i][j] is not None:
                        wandb.log({'IMG_Self_Gen_Direction_post_extended/qw_m{}+qz_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Private_post_ext_denoised[i][j])}, step=epoch)


        # rsample for posteriors, non-shuffled
        cg_imgs_Shared_post_nonshuf = cub_self_and_cross_modal_generation_eval(model,
            test_selected_samples, 8, 10, mode=CrossModalEvalForwardMode.POSTERIOR_NONSHUF, condition_type='shared')
        cg_imgs_Shared_post_nonshuf_denoised = getattr(model, 'last_denoised_posterior_nonshuf_grids', None)
        for i in range(NUM_VAES):
            for j in range(NUM_VAES):
                wandb.log({'IMG_Self&Cros_Gen_Shared_post_nonshuf/qz_m{}+qw_m{}'.format(i, j): wandb.Image(cg_imgs_Shared_post_nonshuf[i][j])}, step=epoch)
                if cg_imgs_Shared_post_nonshuf_denoised and cg_imgs_Shared_post_nonshuf_denoised[i][j] is not None:
                    wandb.log({'IMG_Self&Cros_Gen_Shared_post_nonshuf/qz_m{}+qw_m{}_denoised'.format(i, j): wandb.Image(cg_imgs_Shared_post_nonshuf_denoised[i][j])}, step=epoch)


def run_evaluation(epoch):
    """
    Runs the full evaluation suite for CUB.
    """
    print(f"--- Running Full Evaluation for Epoch {epoch} ---")
    if args.enable_test_epoch:
        test(epoch)
        _cub_test_epoch_qualitative_visuals(epoch)

    if args.enable_unconditional_generation:
        gen_samples = cub_generate_unconditional(model, N=100, coherence_calculation=False, fid_calculation=False)
        for j in range(NUM_VAES):
            wandb.log({'Unconditional_Generations/m{}'.format(j): wandb.Image(gen_samples[j])}, step=epoch)

    # ======== START: Latent Classifiers and Linear Classification ========
    if args.enable_latent_classification:
        # --- Cluster (Shared Attribute) Classification ---
        print("Evaluating Cluster Classification...")

        # MODIFIED: Create more descriptive captions for wandb logs
        direction_map = {0: "Left", 1: "Right", 99: "Center", 404: "N/A"}

        clf_lr_cluster = train_clf_lr_CUB_multi_labelTypes(
            model, train_cluster_loader, device, args, condition_type='shared')
        accuracies_lc_cluster = linear_latent_classification_CUB_multi_labelTypes(
            model, test_time_loader, clf_lr_cluster, device, args, condition_type='shared')
                
        for key, val in accuracies_lc_cluster.items():
            if key != 'prediction_data' and key != 'top_confidence_samples': # Don't log the data dictionary itself
                tag = f"LatentClassAcc_Cluster_clf/{key}"
                # log to wandb (will silently fail if offline)
                try:
                    wandb.log({tag: val}, step=epoch)
                except Exception:
                    pass
                # always print to stdout
                print(f"Epoch {epoch:03d} | {tag} = {val:.4f}")

        # NEW: Log top 5 confident cluster predictions
        # NOTE: prediction_data keys have plural 's', top_confidence_samples keys do not have plural 's'.
        top_cluster_samples = accuracies_lc_cluster.get('top_confidence_samples', {}).get('cluster', {})
        if top_cluster_samples:
            for cluster_id, samples in top_cluster_samples.items():
                images_to_log = []
                for sample_data in samples:
                    dir_label_str = direction_map.get(int(sample_data['direction_label']), "Unk")
                    caption = (f"GT_cluster: {sample_data['gt']} | "
                            f"Img(Z,+): {sample_data['pred_img_z']} ({sample_data['conf_img_z']:.2f}) | "
                            f"Img(W,-): {sample_data['pred_img_w']} ({sample_data['conf_img_w']:.2f}) | "
                            f"Txt(Z,+): {sample_data['pred_txt_z']} ({sample_data['conf_txt_z']:.2f}) | "
                            f"Txt(W,-): {sample_data['pred_txt_w']} ({sample_data['conf_txt_w']:.2f})\n"
                            f"[DataIdx: {int(sample_data['dataset_index'])}, "
                            f"ImgID: {int(sample_data['img_id'])}, "
                            f"Cat: {int(sample_data['category_label'])}, "
                            f"Clu: {int(sample_data['cluster_label'])}, "
                            f"Dir: {dir_label_str}({int(sample_data['direction_label'])})]")
                    images_to_log.append(wandb.Image(sample_data['image'], caption=caption))
                if images_to_log:
                    wandb.log({f"Predictions/Top-{len(images_to_log)}_Confidence_Cluster_{cluster_id}": images_to_log}, step=epoch)


        # Log all validation/test samples, grouped by cluster
        if 'prediction_data' in accuracies_lc_cluster:
            """
            In each batch, the length of cluster_data is 10 times the batch size due to duplicated 10 images for its 10 captions.
            """
            cluster_data = accuracies_lc_cluster['prediction_data']

            # ADDED: Debug prints to check data shape
            print("--- DEBUG: Cluster Table Data ---")
            print(f"Number of images: {len(cluster_data['images'])}")
            print(f"Number of ground truths: {len(cluster_data['gts'])}")
            print(f"Image tensor shape: {cluster_data['images'].shape}")
            print(f"Ground truth array shape: {cluster_data['gts'].shape}")
            for key, value in cluster_data['preds'].items():
                print(f"Prediction '{key}' shape: {value.shape}")
            print("---------------------------------")

            # Group samples by their ground truth cluster label and ambiguity status
            grouped_samples = defaultdict(list)
            grouped_ambiguous_samples = defaultdict(list)
            for i in range(int(len(cluster_data['gts'])/10)): # Loop through all samples
                gt_cluster = int(cluster_data['cluster_labels'][i*10])  # Get the ground truth cluster label for the first image of the 10 duplicates
                # grouped_samples[gt_cluster].append(i*10)  # Store the index of the first image of the 10 duplicates
                if cluster_data['is_ambiguous'][i*10]:  # Check if the sample is ambiguous
                    grouped_ambiguous_samples[gt_cluster].append(i*10)  # Store the index of the first image of the 10 duplicates
                else:
                    grouped_samples[gt_cluster].append(i*10)  # Store the index of the first image of the 10 duplicates

            for cluster_id, indices in grouped_samples.items():
                # 1. Create a list to hold the wandb.Image objects
                images_to_log = []
                # Log up to xx samples per direction
                for i in indices: # [:100]:
                    # 2. Get the image and labels for one sample
                    image = cluster_data['images'][i]  # *10 Get the first image of the 10 duplicates

                    # 3. Format the labels into a caption string
                    gt = cluster_data['gts'][i]
                    dir_label_str = direction_map.get(int(cluster_data['direction_labels'][i]), "Unk")
                    # Get predictions for the first image of the 10 duplicates
                    pred_img_z = cluster_data['preds']['m0_z'][i]
                    pred_img_w = cluster_data['preds']['m0_w'][i]
                    pred_txt_z = cluster_data['preds']['m1_z'][i]
                    pred_txt_w = cluster_data['preds']['m1_w'][i]
                    # Get confidence scores for the predictions
                    pred_img_z_conf = cluster_data['preds']['m0_z_confidence'][i]
                    pred_img_w_conf = cluster_data['preds']['m0_w_confidence'][i]
                    pred_txt_z_conf = cluster_data['preds']['m1_z_confidence'][i]
                    pred_txt_w_conf = cluster_data['preds']['m1_w_confidence'][i]
                    
                    caption = (f"GT_cluster: {gt} | "
                            f"Img(Z,+): {pred_img_z} ({pred_img_z_conf:.2f}) | "
                            f"Img(W,-): {pred_img_w} ({pred_img_w_conf:.2f}) | "
                            f"Txt(Z,+): {pred_txt_z} ({pred_txt_z_conf:.2f}) | "
                            f"Txt(W,-): {pred_txt_w} ({pred_txt_w_conf:.2f})\n"
                            f"[DataIdx: {int(cluster_data['dataset_indices'][i])}, "
                            f"ImgID: {int(cluster_data['img_ids'][i])}, "
                            f"Cat: {int(cluster_data['category_labels'][i])}, "
                            f"Clu: {int(cluster_data['cluster_labels'][i])}, "
                            f"Dir: {dir_label_str}({int(cluster_data['direction_labels'][i])})]")

                    # 4. Create the wandb.Image with its caption and add to the list
                    images_to_log.append(wandb.Image(image, caption=caption))
                    
                # 5. Log the entire list of captioned images under a single key
                if images_to_log:
                    wandb.log({f"Predictions/All-{len(images_to_log)}_valid_in_order_Cluster_{cluster_id}": images_to_log}, step=epoch)

            # Log AMBIGUOUS samples, grouped by cluster
            for cluster_id, indices in grouped_ambiguous_samples.items():
                images_to_log = []
                for i in indices: # Log all ambiguous samples
                    image = cluster_data['images'][i]  # *10 Get the first image of the 10 duplicates

                    # 3. Format the labels into a caption string
                    gt = cluster_data['gts'][i]
                    dir_label_str = direction_map.get(int(cluster_data['direction_labels'][i]), "Unk")
                    # Get predictions for the first image of the 10 duplicates
                    pred_img_z = cluster_data['preds']['m0_z'][i]
                    pred_img_w = cluster_data['preds']['m0_w'][i]
                    pred_txt_z = cluster_data['preds']['m1_z'][i]
                    pred_txt_w = cluster_data['preds']['m1_w'][i]
                    # Get confidence scores for the predictions
                    pred_img_z_conf = cluster_data['preds']['m0_z_confidence'][i]
                    pred_img_w_conf = cluster_data['preds']['m0_w_confidence'][i]
                    pred_txt_z_conf = cluster_data['preds']['m1_z_confidence'][i]
                    pred_txt_w_conf = cluster_data['preds']['m1_w_confidence'][i]

                    caption = (f"GT_cluster: {gt} | "
                            f"Img(Z,+): {pred_img_z} ({pred_img_z_conf:.2f}) | "
                            f"Img(W,-): {pred_img_w} ({pred_img_w_conf:.2f}) | "
                            f"Txt(Z,+): {pred_txt_z} ({pred_txt_z_conf:.2f}) | "
                            f"Txt(W,-): {pred_txt_w} ({pred_txt_w_conf:.2f})\n"
                            f"[DataIdx: {int(cluster_data['dataset_indices'][i])}, "
                            f"ImgID: {int(cluster_data['img_ids'][i])}, "
                            f"Cat: {int(cluster_data['category_labels'][i])}, "
                            f"Clu: {int(cluster_data['cluster_labels'][i])}, "
                            f"Dir: {dir_label_str}({int(cluster_data['direction_labels'][i])})]")

                    # 4. Create the wandb.Image with its caption and add to the list
                    images_to_log.append(wandb.Image(image, caption=caption))

                # 5. Log the entire list of captioned images under a single key
                if images_to_log:
                    wandb.log({f"Predictions/Ambiguous-{len(images_to_log)}_Cluster_{cluster_id}": images_to_log}, step=epoch)

            # <<< Log confidence histograms >>>
            wandb.log({
                "Confidence/Hist_Cluster_Img_Z": wandb.Histogram(cluster_data['preds']['m0_z_confidence']),
                "Confidence/Hist_Cluster_Img_W": wandb.Histogram(cluster_data['preds']['m0_w_confidence']),
                "Confidence/Hist_Cluster_Txt_Z": wandb.Histogram(cluster_data['preds']['m1_z_confidence']),
                "Confidence/Hist_Cluster_Txt_W": wandb.Histogram(cluster_data['preds']['m1_w_confidence']),
            }, step=epoch)

        
        # --- Direction (Private Attribute) Classification ---
        print("\nEvaluating Direction Classification...")
        clf_lr_direction = train_clf_lr_CUB_multi_labelTypes(
            model, train_cluster_loader, device, args, condition_type='private')
        accuracies_lc_direction = linear_latent_classification_CUB_multi_labelTypes(
            model, test_time_loader, clf_lr_direction, device, args, condition_type='private')

        for key, val in accuracies_lc_direction.items():
            if key != 'prediction_data' and key != 'top_confidence_samples':  # Don't log the data dictionary itself
                tag = f"LatentClassAcc_Direction_clf/{key}"
                # log to wandb (will silently fail if offline)
                try:
                    wandb.log({tag: val}, step=epoch)
                except Exception:
                    pass
                # always print to stdout
                print(f"Epoch {epoch:03d} | {tag} = {val:.4f}")

        # NEW: Log top 5 confident direction predictions
        top_direction_samples = accuracies_lc_direction.get('top_confidence_samples', {}).get('direction', {})
        if top_direction_samples:
            for direction_id, samples in top_direction_samples.items():
                direction_label = direction_map.get(direction_id, "Unk")
                images_to_log = []
                for sample_data in samples:
                    gt_label = direction_map.get(sample_data['gt'], "N/A")
                    pred_w_label = direction_map.get(sample_data['pred_img_w'], "N/A")
                    pred_z_label = direction_map.get(sample_data['pred_img_z'], "N/A")

                    caption = (f"GT_direction: {gt_label} | "
                            f"Img(W,+): {pred_w_label} ({sample_data['conf_img_w']:.2f}) | "
                            f"Img(Z,-): {pred_z_label} ({sample_data['conf_img_z']:.2f})\n"
                            f"[DataIdx: {int(sample_data['dataset_index'])}, "
                            f"ImgID: {int(sample_data['img_id'])}, "
                            f"Cat: {int(sample_data['category_label'])}, "
                            f"Clu: {int(sample_data['cluster_label'])}, "
                            f"Dir: {direction_label}({direction_id})]")
                    images_to_log.append(wandb.Image(sample_data['image'], caption=caption))
                if images_to_log:
                    wandb.log({f"Predictions/Top-{len(images_to_log)}_Confidence_Direction_{direction_label}({direction_id})": images_to_log}, step=epoch)


        # Log all validation/test samples, grouped by direction
        if 'prediction_data' in accuracies_lc_direction:
            """ 
            In each batch, the length of direction_data is 10 times the batch size due to duplicated 10 images for its 10 captions.
            """
            direction_data = accuracies_lc_direction['prediction_data']

            # ADDED: Debug prints to check data shape
            print("--- DEBUG: Direction Table Data ---")
            print(f"Number of images: {len(direction_data['images'])}")
            print(f"Number of GT directions: {len(direction_data['gts'])}")
            print(f"Number of predicted directions (m0_w): {len(direction_data['preds']['m0_w'])}")
            print(f"Number of predicted directions (m0_z): {len(direction_data['preds']['m0_z'])}")
            for key, value in direction_data['preds'].items():
                print(f"Prediction '{key}' shape: {value.shape}")

            # Group samples by their ground truth cluster label and ambiguity status
            grouped_samples = defaultdict(list)
            grouped_ambiguous_samples = defaultdict(list)

            for i in range(int(len(direction_data['gts'])/10)):  # Loop through all samples
                gt_direction = int(direction_data['gts'][i*10])  # Get the ground truth direction for the first image of the 10 duplicates
                # grouped_samples[gt_direction].append(i*10)  # Store the index of the first image of the 10 duplicates
                if direction_data['is_ambiguous'][i*10]:  # Check if the sample is ambiguous
                    grouped_ambiguous_samples[gt_direction].append(i*10)  # Store the index of the first image of the 10 duplicates
                else:
                    grouped_samples[gt_direction].append(i*10)

            for direction_id, indices in grouped_samples.items():

                # 1. Create a list to hold the wandb.Image objects
                images_to_log = []
                direction_label_str = direction_map.get(direction_id, "Unk")
                for i in indices:  # [:100]:
                    # 2. Get the image and labels for one sample
                    image = direction_data['images'][i]  # *10 Get the first image of the 10 duplicates

                    # 3. Format the labels into a caption string
                    gt = direction_data['gts'][i]
                    # Get predictions for the first image of the 10 duplicates
                    pred_img_w = direction_data['preds']['m0_w'][i]
                    pred_img_z = direction_data['preds']['m0_z'][i]
                    # Get confidence scores for the predictions
                    pred_img_w_conf = direction_data['preds']['m0_w_confidence'][i]
                    pred_img_z_conf = direction_data['preds']['m0_z_confidence'][i]

                    
                    caption = (f"GT_direction: {gt} | "
                            f"Img(W,+): {pred_img_w} ({pred_img_w_conf:.2f}) | "
                            f"Img(Z,-): {pred_img_z} ({pred_img_z_conf:.2f})\n"
                            f"[DataIdx: {int(direction_data['dataset_indices'][i])}, "
                            f"ImgID: {int(direction_data['img_ids'][i])}, "
                            f"Cat: {int(direction_data['category_labels'][i])}, "
                            f"Clu: {int(direction_data['cluster_labels'][i])}, "
                            f"Dir: {direction_label_str}({direction_id})]")
                    
                    # 4. Create the wandb.Image with its caption and add to the list
                    images_to_log.append(wandb.Image(image, caption=caption))

                # Log the entire list of captioned images under a single key
                if images_to_log:
                    wandb.log({f"Predictions/All-{len(images_to_log)}_valid_in_order_Direction_{direction_label_str}({direction_id})": images_to_log}, step=epoch)

            # Log AMBIGUOUS samples, grouped by direction
            for direction_id, indices in grouped_ambiguous_samples.items():
                direction_label_str = direction_map.get(direction_id, "Unk")
                images_to_log = []
                for i in indices: # Log all ambiguous samples
                    image = direction_data['images'][i]  # *10 Get the first image of the 10 duplicates
                    caption = (f"GT_direction: {direction_data['gts'][i]} | "
                            f"Img(W,+): {direction_data['preds']['m0_w'][i]} ({direction_data['preds']['m0_w_confidence'][i]:.2f}) | "
                            f"Img(Z,-): {direction_data['preds']['m0_z'][i]} ({direction_data['preds']['m0_z_confidence'][i]:.2f})\n"
                            f"[DataIdx: {int(direction_data['dataset_indices'][i])}, "
                            f"ImgID: {int(direction_data['img_ids'][i])}, "
                            f"Cat: {int(direction_data['category_labels'][i])}, "
                            f"Clu: {int(direction_data['cluster_labels'][i])}, "
                            f"Dir: {direction_label_str}({direction_id})]")
                    images_to_log.append(wandb.Image(image, caption=caption))

                if images_to_log:
                    wandb.log({f"Predictions/Ambiguous-{len(images_to_log)}_Direction_{direction_label_str}({direction_id})": images_to_log}, step=epoch)

            # <<< Log confidence histograms >>>
            wandb.log({
                "Confidence/Hist_Direction_Img_W": wandb.Histogram(direction_data['preds']['m0_w_confidence']),
                "Confidence/Hist_Direction_Img_Z": wandb.Histogram(direction_data['preds']['m0_z_confidence']),
            }, step=epoch)

    # ======== END: Latent Classifiers and Linear Classification ========

    # === FID Calculation ===
    if args.enable_fid:
        print("\nCalculating FID...")
        calculate_fid_routine(
            datadirCUB,
            fid_path,
            10000,
            epoch,
            model,
            test_time_loader,
            device,
            args.inception_path,
        )

    # Direction-label analysis hooks
    # # # Latent space visualization
    # # if epoch % visualize_epoch == 0:

    if args.enable_tSNE_UMAP:
        visualize_ratio = 1.0  # Use full evaluation set by default.

        view0_z_cluster, view0_w_cluster_mis = visualize_latents_with_priors(
                                    model, model.vaes[0], model.encoders[0], test_time_image_loader, test_time_image_loader, 
                                    "m0_img_z_cluster", "m0_img_w_cluster_mis", device=device, figure=1, condition_type='shared', 
                                    view_index=0, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                    save_file=os.path.join(args.tSNE_save_dir, 
                                                        'view0_shared_image.png')) # device=device
        
        view1_shared_caption, _ = visualize_latents_with_priors(
                                    model, model.vaes[1], model.encoders[1], test_time_caption_loader, test_time_caption_loader,
                                    "view1_shared_caption", "N/A?", device=device, figure=2, condition_type='shared', 
                                    view_index=1, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                    save_file=os.path.join(args.tSNE_save_dir, 
                                                        'view1_shared_caption.png'))

        view0_w_direction, view0_z_direction_mis = visualize_latents_with_priors(
                                    model, model.vaes[0], model.encoders[0], test_time_image_loader, test_time_image_loader, 
                                    "m0_img_w_direction", "m0_img_z_direction_mis", device=device, figure=3, condition_type='private', 
                                    view_index=0, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                    save_file=os.path.join(args.tSNE_save_dir, 
                                                        'view0_private_image.png'))
            
        # log tSNE images to wandb
        wandb.log({'tSNE_or_UMAP/m0_z_cluster': wandb.Image(view0_z_cluster)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m0_w_cluster_mis': wandb.Image(view0_w_cluster_mis)}, step=epoch)
        
        wandb.log({'tSNE_or_UMAP/m1_shared_caption': wandb.Image(view1_shared_caption)}, step=epoch)
        
        wandb.log({'tSNE_or_UMAP/m0_w_direction': wandb.Image(view0_w_direction)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m0_z_direction_mis': wandb.Image(view0_z_direction_mis)}, step=epoch)



if __name__ == '__main__':
    if args.test_only:
        print("--- Running in Test-Only Mode ---")
        checkpoint_to_load = args.checkpoint_path
        print(f"Loading checkpoint: {checkpoint_to_load}")
        match = re.search(r'model_(\d+).rar', checkpoint_to_load)
        if match:
            epoch_to_test = int(match.group(1))
        else:
            epoch_to_test = 999
            print(
                "Warning: Could not infer epoch from filename (expected model_<epoch>.rar). "
                f"Using epoch {epoch_to_test} for logging."
            )

        print(f"Loading model weights from {checkpoint_to_load}...")
        model.load_state_dict(torch.load(checkpoint_to_load, map_location=device), strict=False)
        _log_param_counts(f"test_only_epoch_{epoch_to_test}")

        run_evaluation(epoch_to_test)
        
        print("--- Test-Only Mode Finished ---")

    else:
        # --- Training Mode ---
        test_epoch_freq = 1

        def _collect_checkpoints(prefix):
            pattern = os.path.join(runPath, f'{prefix}_*.rar')
            checkpoints = []
            for file_path in glob.glob(pattern):
                match = re.search(rf'{prefix}_(\d+)\.rar$', os.path.basename(file_path))
                if match:
                    checkpoints.append((int(match.group(1)), file_path))
            checkpoints.sort(key=lambda item: item[0])
            return checkpoints

        def _prune_checkpoints(prefix, keep_recent, special_epochs_fn=None):
            checkpoints = _collect_checkpoints(prefix)
            if not checkpoints:
                return
            keep_epochs = set(epoch for epoch, _ in checkpoints[-keep_recent:])
            if special_epochs_fn:
                keep_epochs.update(special_epochs_fn([epoch for epoch, _ in checkpoints]))
            for epoch, path in checkpoints:
                if epoch not in keep_epochs:
                    os.remove(path)

        def _model_special_epochs(epochs):
            return {epoch for epoch in epochs if epoch % test_epoch_freq == 0}

        def _optimizer_special_epochs(epochs):
            eval_epochs = [epoch for epoch in epochs if epoch % test_epoch_freq == 0]
            return {max(eval_epochs)} if eval_epochs else set()

        for epoch in range(start_epoch, args.epochs + 1):
            train(epoch)

            model_checkpoint_path = os.path.join(runPath, f'model_{epoch}.rar')
            optimizer_checkpoint_path = os.path.join(runPath, f'optimizer_{epoch}.rar')
            save_model_light(model, model_checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_checkpoint_path)

            _prune_checkpoints('model', keep_recent=2, special_epochs_fn=_model_special_epochs)
            _prune_checkpoints('optimizer', keep_recent=2, special_epochs_fn=_optimizer_special_epochs)

            if epoch % test_epoch_freq == 0:
                run_evaluation(epoch)
