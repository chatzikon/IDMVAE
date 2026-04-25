# Train IDMVAE model on PolyMNIST dataset
import os
# Deterministic behavior: 
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Set before importing torch
# os.environ["WANDB_MODE"] = "disabled"

import argparse
import json
import numpy as np
import sys
from pathlib import Path
import re
import glob

import torch
from torch import optim
from torchvision import transforms

import models
from objectives import compute_idmvae_loss
from dataset_PolyMNIST_quadrant import PolyMNISTDataset_pt as PolyMNISTDataset_quadrant_pt

from utils import (
    CrossModalEvalForwardMode,
    DigitClassifier,
    Logger,
    get_and_save_structured_polymnist_samples,
    load_structured_samples,
    save_model_light,
    unpack_data_PM_quadrant as unpack_data,
)

from eval_functions_polyMNIST import (
    calculate_fid_routine,
    linear_latent_classification_multi_labelType,
    self_and_cross_coherence_calculation,
    train_clf_lr_multi_labelType,
    unconditional_coherence,
)
from eval_functions import visualize_latents_with_priors
from eval_functions_polyMNIST import (
    polymnist_generate_unconditional_plot,
    polymnist_self_and_cross_modal_generation_eval,
)

import wandb

parser = argparse.ArgumentParser(description='IDMVAE PolyMNIST Experiment')
parser.add_argument('--experiment', type=str, default='IDMVAE_PolyMNIST', metavar='E',
                    help='experiment name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')

parser.add_argument('--priorposterior', type=str, default='Normal', choices=['Normal', 'Laplace', 'Diffusion'],
                    help='distribution choice for prior and posterior')
parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                    help='loss weight for diffusion')
parser.add_argument('--diffusion_stop_grad_on_input', action='store_true', default=False,
                    help='stop gradient on x_start of diffusion')
parser.add_argument('--likelihood', type=str, default='Laplace', choices=['Normal', 'Laplace'],
                    help='distribution choice for likelihood')
parser.add_argument('--beta', type=float, default=2.5,
                    help='beta hyperparameter in VAE objective')
parser.add_argument('--K', type=int, default=1,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100,
                    help='Training epochs (paper Appendix B.1 uses 100)')
parser.add_argument('--latent-dim-w', type=int, default=128,
                    help='latent modality-specific dimensionality')
parser.add_argument('--latent-dim-z', type=int, default=32,
                    help='latent shared dimensionality')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

parser.add_argument('--cross_mi_loss_scale', type=float, default=80.0,
                    help='Weight λ1 on L_CrossMI (paper Appendix B.1)')
parser.add_argument('--gen_aug_loss_scale', type=float, default=20.0,
                    help='Weight λ2_contrast on L_GenAug when --gen_aug_loss_type CL (paper Appendix B.1)')
parser.add_argument('--gen_aug_sampling_scheme', type=str, default='posterior',
                    choices=['posterior', 'prior', 'diffusion_prior'],
                    help='Sampling scheme for generative augmentation loss')
parser.add_argument('--gen_aug_loss_type', type=str, default='CL', choices=['CL', 'ML'],
                    help='Generative augmentation loss: CL (contrastive) or ML (matching)')

parser.add_argument('--dataset', type=str, default='PolyMNIST', choices=['PolyMNIST', 'PolyMNIST_original'],
                    help='dataset choice')
parser.add_argument('--datadir', type=str, default='./data',
                    help='Directory where data is stored')
parser.add_argument('--outputdir', type=str, default='../outputs',
                    help='Output directory')
parser.add_argument('--datadir_fid', type=str, default='/data/backed_up/shared/Data/PolyMNIST/quadrants_4x_64x64_scl1/Mar19_2025_t0',
                    help='Directory where data is stored')
parser.add_argument('--print-freq', type=int, default=50,
                    help='Frequenty for printing')
parser.add_argument('--inception_path', type=str, default='/data/backed_up/shared/Data/PolyMNIST/pt_inception-2015-12-05-6726825d.pth',
                    help='Path to inception module for FID calculation')
parser.add_argument('--pretrained-clfs-dir-path', type=str, default='./data/trained_clfs_polyMNIST',
                    help="Path to directory containing pre-trained digit classifiers for each modality")
parser.add_argument('--pretrained_clfs_digit_dir_path', type=str, default='./data/trained_clfs_polyMNIST',
                    help="Path to directory containing pre-trained digit classifiers for each modality")
parser.add_argument('--pretrained_clfs_quadrant_dir_path', type=str, default='./data/trained_clfs_polyMNIST',
                    help="Path to directory containing pre-trained quadrant classifiers for each modality")
parser.add_argument('--debug_mini_data', action='store_true', default=False,
                    help='Use mini dataset for debugging')
parser.add_argument('--tSNE_save_dir', type=str, default='./data/tSNE_results',
                    help='Directory to save tSNE results')

parser.add_argument('--note', type=str, default='', help='Note for runId')

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
parser.add_argument('--enable_generation_coherence', action='store_true', default=False,
                    help='Enable generation coherence during testing.')
parser.add_argument('--enable_fid', action='store_true', default=False,
                    help='Enable FID calculation during testing.')
parser.add_argument('--enable_tSNE_UMAP', action='store_true', default=False,
                    help='Enable tSNE and UMAP visualization during testing.')
parser.add_argument('--use_mean_in_latent_visualization', action='store_true', default=False,
                    help='Use mean of the latent distribution for tSNE/UMAP visualization instead of sampling')
parser.add_argument('--lv_umap_n_neighbors', type=int, default=50,
                    help='Number of neighbors for UMAP on latent variables')
parser.add_argument('--lv_umap_min_dist', type=float, default=0.1,
                    help='Minimum distance for UMAP on latent variables')


# checkpoint
## Resuming
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume training from the last checkpoint in the run directory.')
## Test only
parser.add_argument('--test-only', action='store_true', default=False,
                    help='Load a checkpoint and run evaluation only, without training.')
parser.add_argument('--checkpoint-path', type=str, default='',
                    help='Path to a model checkpoint .rar file (required with --test-only).')
parser.add_argument('--print-params-only', action='store_true', default=False,
                    help='Instantiate model (optionally load a checkpoint) and print parameter counts, then exit.')

# Args
args = parser.parse_args()


def build_polymnist_datasets(datadir: str):
    """Build PolyMNIST datasets (quadrant_pt only)."""
    tx = transforms.ToTensor()
    train_datapath = datadir + "/train/"
    valid_datapath = datadir + "/val/"
    test_datapath = datadir + "/test/"
    subtrain_datapath = datadir + "/Subsample/train/"
    debug_mini_datapath = datadir + "/Subsample/debug_mini/"

    train_dataset = PolyMNISTDataset_quadrant_pt(train_datapath, transform=tx)
    valid_dataset = PolyMNISTDataset_quadrant_pt(valid_datapath, transform=tx)
    test_dataset = PolyMNISTDataset_quadrant_pt(test_datapath, transform=tx)
    subtrain_dataset = PolyMNISTDataset_quadrant_pt(subtrain_datapath, transform=tx)
    debug_mini_dataset = PolyMNISTDataset_quadrant_pt(debug_mini_datapath, transform=tx)

    return train_dataset, valid_dataset, test_dataset, subtrain_dataset, debug_mini_dataset

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


flags_clf_lr = {'latdimz': args.latent_dim_z,
                'latdimw': args.latent_dim_w}
args.latent_dim_u = args.latent_dim_w + args.latent_dim_z

# Random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# CUDA stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Get model class and initialize model
modelC = getattr(models, 'IDMVAE_PolyMNIST_5modalities')
model = modelC(args).to(device) # Original

def _log_param_counts(tag: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params/PolyMNIST/{tag}] total={total:,} trainable={trainable:,}")

_log_param_counts("init")

if args.print_params_only:
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'. Exiting.")
            sys.exit(1)
        print(f"Loading checkpoint for parameter count: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
        _log_param_counts("print_only_checkpoint")
    sys.exit(0)

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

# Set up run path
runId = (
    f"{args.note}_K{args.K}_B{args.batch_size}_{args.priorposterior}_{args.likelihood}_b{args.beta}_"
    f"{args.gen_aug_loss_scale}_{args.cross_mi_loss_scale}_"
    f"{args.latent_dim_w}_{args.latent_dim_z}_"
    f"s{args.seed}"
)

experiment_dir = Path(os.path.join(args.outputdir, args.experiment, "checkpoints"))
experiment_dir.mkdir(parents=True, exist_ok=True)

runPath = os.path.join(str(experiment_dir), runId)
# Checkpoint and resuming logic
# Resume behavior: require runPath to exist and load latest available checkpoint.
start_epoch = 1
# Ensure parent experiment directory exists
experiment_dir.mkdir(parents=True, exist_ok=True)
if args.resume:
    if not os.path.exists(runPath):
        print(f"Error: Run path '{runPath}' does not exist for resuming. Exiting.")
        sys.exit(1)

    # Find the latest checkpoint
    checkpoints = glob.glob(os.path.join(runPath, 'model_*.rar'))
    if not checkpoints:
        print(f"Error: No checkpoints found in '{runPath}'. Starting from scratch.")
        wandb_resume_status = "never"
    else:
        latest_checkpoint = max(checkpoints, key=lambda p: int(re.search(r'model_(\d+).rar', p).group(1)))
        checkpoint_epoch = int(re.search(r'model_(\d+).rar', latest_checkpoint).group(1))
        
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        _log_param_counts(f"resume_epoch_{checkpoint_epoch}")
        
        # Optional: Load optimizer state if saved
        optimizer_path = os.path.join(runPath, f'optimizer_{checkpoint_epoch}.rar')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print(f"Loaded optimizer state from epoch {checkpoint_epoch}.")

        start_epoch = checkpoint_epoch + 1
        wandb_resume_status = "allow"
        print(f"Resuming from epoch {start_epoch}...")

elif not args.test_only:
    if os.path.exists(runPath):
        print(f"Error: Run path '{runPath}' already exists. Use --resume to continue or change parameters to start a new run.")
        sys.exit(1)
    
    # experiment_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(runPath)
    wandb_resume_status = "never"
    
    # Save args to run directory for new runs
    with open('{}/args.json'.format(runPath), 'w') as fp:
        json.dump(args.__dict__, fp)
    torch.save(args, '{}/args.rar'.format(runPath))
else: # Test-only mode
    wandb_resume_status = "allow" # Allow wandb to resume to the same run for logging test metrics
        # Create the directory for the test run's log file
    os.makedirs(runPath, exist_ok=True)

sys.stdout = Logger('{}/run.log'.format(runPath))

print('Expt:', runPath)
print('RunID:', runId)


# === Set Parameters from args ===
# core parameters from args
model.params.cross_mi_loss_scale = args.cross_mi_loss_scale
model.params.gen_aug_loss_scale = args.gen_aug_loss_scale
model.params.gen_aug_sampling_scheme = args.gen_aug_sampling_scheme
model.params.gen_aug_loss_type = args.gen_aug_loss_type

NUM_VAES = len(model.vaes)

# Create path where to temporarily save images to compute FID scores
fid_path = os.path.join(args.datadir_fid, 'FID/fids_PM_' + (runPath.rsplit('/')[-1]))
datadirPM = os.path.join(args.datadir_fid, "PolyMNIST")
# WandB
wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project=args.experiment,
    # Track hyperparameters and run metadata
    config=args,
    # Run name
    name=runId,
    id=runId, # Use runId to ensure resuming logs to the same run
    resume=wandb_resume_status
)

num_workers = args.num_workers if hasattr(args, 'num_workers') else 32

# Data loaders
train_dataset, valid_dataset, test_dataset, subtrain_dataset, debug_mini_dataset = build_polymnist_datasets(args.datadir)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if device == 'cuda' else {}

# Deterministic behavior for DataLoader
g = torch.Generator()
g.manual_seed(0)
kwargs['generator'] = g
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
subtrain_loader = torch.utils.data.DataLoader(subtrain_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
debug_mini_loader = torch.utils.data.DataLoader(debug_mini_dataset, batch_size=args.batch_size, shuffle=True, **kwargs) 

if args.debug_mini_data:
    training_loader = debug_mini_loader
    validating_loader = debug_mini_loader
    testing_loader = debug_mini_loader
    subtraining_loader = debug_mini_loader
else:
    training_loader = train_loader
    validating_loader = val_loader
    testing_loader = test_loader
    subtraining_loader = subtrain_loader

 # select test time dataset and dataloader
if args.test_time_dataset_state == "eval":
    test_time_loader = validating_loader
if args.test_time_dataset_state == "test":
    test_time_loader = testing_loader


IMG_SIZE = 64
RESNET_S0 = 8

# Loading pre-trained digit / quadrant classifiers (DigitClassifier architecture in utils).
clfs_digit = [
    DigitClassifier(input_size=IMG_SIZE, res_s0=RESNET_S0) for _ in model.vaes
]
clfs_quadrant = [
    DigitClassifier(input_size=IMG_SIZE, res_s0=RESNET_S0) for _ in model.vaes
]
print("Loading pre-trained image classifiers: DigitClassifier")


needs_conversion = not args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
for idx, vae in enumerate(model.vaes):
    clfs_digit[idx].load_state_dict(
        torch.load(os.path.join(args.pretrained_clfs_digit_dir_path, "pretrained_img_to_digit_clf_m" + str(idx)),
                   **conversion_kwargs), strict=True)
    clfs_quadrant[idx].load_state_dict(
        torch.load(os.path.join(args.pretrained_clfs_quadrant_dir_path, "pretrained_img_to_quadrant_clf_m" + str(idx)),
                   **conversion_kwargs), strict=True)    
    clfs_digit[idx].eval()
    clfs_quadrant[idx].eval()
    if args.cuda:
        clfs_digit[idx].cuda()
        clfs_quadrant[idx].cuda()


def train(epoch):
    """
    Training function
    """
    model.train()
    b_loss = 0
    # Iterate over the data
    for i, dataT in enumerate(training_loader):
        # Unpack data
        data, labels_batch = unpack_data(dataT, device=device)
        optimizer.zero_grad()

        bs = data[0].size(0)

        # Compute loss and backprop
        loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(model, data, K=args.K)

        wandb.log({"Loss/train_loss": loss.item()}, step=epoch)
        wandb.log({"Loss/train_recon_kl_sum": recon_kl_sum_loss.item()}, step=epoch)
        wandb.log({"Loss/train_likelihood": llik_recon_loss.item()}, step=epoch)
        wandb.log({"Loss/train_kl": kl_div_loss.item()}, step=epoch)
        wandb.log({"Loss/train_cross_mi": cross_mi_loss.item()}, step=epoch)
        wandb.log({"Loss/train_gen_aug": gen_aug_loss.item()}, step=epoch)
        wandb.log({"Loss/train_diffusion_loss": diffusion_loss.item()}, step=epoch)

        loss.backward()

        # Optimizer step
        optimizer.step()
        # Get batch loss
        b_loss += loss.item() * bs
        # Printing
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item()))
    # Epoch loss
    epoch_loss = b_loss / len(training_loader.dataset)
    wandb.log({"Loss/train": epoch_loss}, step=epoch) # Loss/*** could automatically categorize into "Loss" in wandb
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))

def test(epoch):
    """
    Test-time loss on the test-time loader only.
    """
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for _, dataT in enumerate(test_time_loader):
            data, _ = unpack_data(dataT, device=device)
            bs = data[0].size(0)
            loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(
                model, data, K=args.K, test=True
            )

            wandb.log({"Loss/test_loss": loss.item()}, step=epoch)
            wandb.log({"Loss/test_recon_kl_sum": recon_kl_sum_loss.item()}, step=epoch)
            wandb.log({"Loss/test_likelihood": llik_recon_loss.item()}, step=epoch)
            wandb.log({"Loss/test_kl": kl_div_loss.item()}, step=epoch)
            wandb.log({"Loss/test_cross_mi": cross_mi_loss.item()}, step=epoch)
            wandb.log({"Loss/test_gen_aug": gen_aug_loss.item()}, step=epoch)
            wandb.log({"Loss/test_diffusion_loss": diffusion_loss.item()}, step=epoch)

            b_loss += loss.item() * bs

    # Epoch test loss
    epoch_loss = b_loss / len(test_time_loader.dataset)
    wandb.log({"Loss/test": epoch_loss}, step=epoch)
    print('Test Time Dataset Size:', len(test_time_loader.dataset))
    print('====>             Test loss: {:.4f}'.format(epoch_loss))


def run_evaluation(epoch):
    """
    Runs the full evaluation suite, including testing, coherence, and visualization.
    """
    print(f"--- Running Full Evaluation for Epoch {epoch} ---")

    if args.enable_test_epoch:
        test(epoch)
        # Structured qualitative samples + cross-modal generation grids (wandb).
        qual_sample_dir = os.path.join(args.datadir, "qualitative_samples")
        qual_sample_pt_path = os.path.join(qual_sample_dir, "qualitative_samples.pt")
        if os.path.exists(qual_sample_pt_path):
            print("Loading pre-saved qualitative samples...")
            test_selected_samples_combined, verification_grid = load_structured_samples(qual_sample_dir, device)
        else:
            print("Searching for and saving new qualitative samples...")
            test_selected_samples_combined, verification_grid = get_and_save_structured_polymnist_samples(
                dataset=test_time_loader.dataset,
                num_testing_images=test_time_loader.dataset.__len__(),
                device=device,
                save_dir=qual_sample_dir,
            )
        wandb.log(
            {"Visualization_Samples/Qualitative_Samples_Verification_Grid": wandb.Image(verification_grid)},
            step=epoch,
        )
        model.eval()
        with torch.no_grad():
            # Prior-sampled shared / private (PRIOR_CTRL)
            cg_imgs_digit, recon_grid_combined_digit = polymnist_self_and_cross_modal_generation_eval(
                model,
                test_selected_samples_combined,
                num=10,
                N=10,
                mode=CrossModalEvalForwardMode.PRIOR_CTRL,
                condition_type="shared",
                num_comb=5,
                N_comb=1,
            )
            cg_imgs_quadrant, recon_grid_combined_quadrant = polymnist_self_and_cross_modal_generation_eval(
                model,
                test_selected_samples_combined,
                num=10,
                N=10,
                mode=CrossModalEvalForwardMode.PRIOR_CTRL,
                condition_type="private",
                num_comb=5,
                N_comb=1,
            )
            for mi in range(NUM_VAES):
                wandb.log(
                    {"Visualization_Samples/Prior_qz_m{}+pw_all".format(mi): wandb.Image(recon_grid_combined_digit[mi])},
                    step=epoch,
                )
                wandb.log(
                    {"Visualization_Samples/Prior_qw_m{}+pz_all".format(mi): wandb.Image(recon_grid_combined_quadrant[mi])},
                    step=epoch,
                )
                wandb.log(
                    {"IMG_Self&Cros_Gen_Digit_prior/qz_m{}+pw_all".format(mi): wandb.Image(recon_grid_combined_digit[mi])},
                    step=epoch,
                )
                wandb.log(
                    {"IMG_Cros_Gen_Quad_prior/qw_m{}+pz_all".format(mi): wandb.Image(recon_grid_combined_quadrant[mi])},
                    step=epoch,
                )
                for mj in range(NUM_VAES):
                    wandb.log(
                        {"IMG_Self&Cros_Gen_Digit_prior/qz_m{}+pw_m{}".format(mi, mj): wandb.Image(cg_imgs_digit[mi][mj])},
                        step=epoch,
                    )
                    if mi == mj:
                        wandb.log(
                            {"IMG_Self_Gen_Quad_prior/qw_m{}+pz_m{}".format(mi, mj): wandb.Image(cg_imgs_quadrant[mi][mj])},
                            step=epoch,
                        )
                    if mi != mj:
                        wandb.log(
                            {"IMG_Cros_Gen_Quad_prior/qw_m{}+pz_m{}".format(mi, mj): wandb.Image(cg_imgs_quadrant[mi][mj])},
                            step=epoch,
                        )
            # Posterior-sampled shared / private (POSTERIOR_CTRL)
            cg_imgs_digit_post, cg_imgs_digit_post_extended = polymnist_self_and_cross_modal_generation_eval(
                model,
                test_selected_samples_combined,
                num=10,
                N=10,
                mode=CrossModalEvalForwardMode.POSTERIOR_CTRL,
                condition_type="shared",
                num_comb=5,
                N_comb=1,
            )
            cg_imgs_quadrant_post, cg_imgs_quadrant_post_extended = polymnist_self_and_cross_modal_generation_eval(
                model,
                test_selected_samples_combined,
                num=10,
                N=10,
                mode=CrossModalEvalForwardMode.POSTERIOR_CTRL,
                condition_type="private",
                num_comb=5,
                N_comb=1,
            )
            for mi in range(NUM_VAES):
                for mj in range(NUM_VAES):
                    wandb.log(
                        {"IMG_Self&Cros_Gen_Digit_post/qz_m{}+qw_m{}".format(mi, mj): wandb.Image(cg_imgs_digit_post[mi][mj])},
                        step=epoch,
                    )
                    if mi == mj:
                        wandb.log(
                            {"Visualization_Samples/Post_qz_m{}+qw_m{}".format(mi, mj): wandb.Image(cg_imgs_digit_post[mi][mj])},
                            step=epoch,
                        )
                        wandb.log(
                            {"Visualization_Samples/Post_extended_qz_m{}+qw_m{}".format(mi, mj): wandb.Image(cg_imgs_digit_post_extended[mi][mj])},
                            step=epoch,
                        )
                        wandb.log(
                            {"Visualization_Samples/Post_qw_m{}+qz_m{}".format(mi, mj): wandb.Image(cg_imgs_quadrant_post[mi][mj])},
                            step=epoch,
                        )
                        wandb.log(
                            {"Visualization_Samples/Post_extended_qw_m{}+qz_m{}".format(mi, mj): wandb.Image(cg_imgs_quadrant_post_extended[mi][mj])},
                            step=epoch,
                        )
                        wandb.log(
                            {"IMG_Self_Gen_Quad_post/qw_m{}+qz_m{}".format(mi, mj): wandb.Image(cg_imgs_quadrant_post[mi][mj])},
                            step=epoch,
                        )
                    if mi != mj:
                        wandb.log(
                            {"IMG_Cros_Gen_Quad_post/qw_m{}+qz_m{}".format(mi, mj): wandb.Image(cg_imgs_quadrant_post[mi][mj])},
                            step=epoch,
                        )

    if args.enable_generation_coherence:
        coherence_digit_main, coherence_digit_cross, means_selfcoh_digit_main, means_selfcoh_digit_cross, \
            means_cctarget_digit_main, means_cctarget_digit_cross, cctarget_all_digit_main, cctarget_all_digit_cross, \
                meanall_digit_main, meanall_digit_cross = self_and_cross_coherence_calculation(
                    model, test_time_loader, clfs_digit, clfs_quadrant, device, condition_type='shared', rsample_type='prior'
                    )
        coherence_quad_main, coherence_quad_cross, means_selfcoh_quad_main, means_selfcoh_quad_cross, \
            means_cctarget_quad_main, means_cctarget_quad_cross, cctarget_all_quad_main, cctarget_all_quad_cross, \
                meanall_quad_main, meanall_quad_cross = self_and_cross_coherence_calculation(
                    model, test_time_loader, clfs_quadrant, clfs_digit, device, condition_type='private', rsample_type='prior'
                )

        # --- Bad Name of Digit: Should be Shared Post + Private Prior ---
        wandb.log({"Condition_Cohere_Digit_prior/self_mean(N/A_MMVAE+)": means_selfcoh_digit_main}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_prior/target_meanall": cctarget_all_digit_main}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_prior/mean_All(N/A_MMVAE+)": meanall_digit_main}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_prior/self_mean_crossClf(N/A_MMVAE+)": means_selfcoh_digit_cross}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_prior/target_meanall_crossClf": cctarget_all_digit_cross}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_prior/mean_All_crossClf(N/A_MMVAE+)": meanall_digit_cross}, step=epoch)

        # --- Bad Name of Quad: Should be Private Post + Shared Prior ---
        wandb.log({"Condition_Self_Cohere_Quad_prior/self_mean(N/A_MMVAE+)": means_selfcoh_quad_main}, step=epoch) 
        wandb.log({"Condition_Cros_Cohere_Quad_prior/target_meanall": cctarget_all_quad_main}, step=epoch)                
        wandb.log({"Condition_Cros_Cohere_Quad_prior/mean_All(N/A_MMVAE+)": meanall_quad_main}, step=epoch)
        wandb.log({"Condition_Self_Cohere_Quad_prior/self_mean_crossClf(N/A_MMVAE+)": means_selfcoh_quad_cross}, step=epoch)
        wandb.log({"Condition_Cros_Cohere_Quad_prior/target_meanall_crossClf": cctarget_all_quad_cross}, step=epoch)
        wandb.log({"Condition_Cros_Cohere_Quad_prior/mean_All_crossClf(N/A_MMVAE+)": meanall_quad_cross}, step=epoch)

        for i in range(NUM_VAES):
            
            wandb.log({"Condition_Cohere_Digit_prior/target_pw_m{}".format(i): means_cctarget_digit_main[i]}, step=epoch)
            wandb.log({"Condition_Cohere_Digit_prior/target_crossClf_pw_m{}".format(i): means_cctarget_digit_cross[i]}, step=epoch)

            wandb.log({"Condition_Cros_Cohere_Quad_prior/target_pz_m{}".format(i): means_cctarget_quad_main[i]}, step=epoch)  
            wandb.log({"Condition_Cros_Cohere_Quad_prior/target_crossClf_pz_m{}".format(i): means_cctarget_quad_cross[i]}, step=epoch)                  

            for j in range(NUM_VAES):
                wandb.log({"Condition_Cohere_Digit_prior/qz_m{}+pw_m{}".format(i, j): coherence_digit_main[i][j]}, step=epoch)
                wandb.log({"Condition_Cohere_Digit_prior/crossClf_qz_m{}+pw_m{}".format(i, j): coherence_digit_cross[i][j]}, step=epoch)
                if i == j:
                    wandb.log({"Condition_Self_Cohere_Quad_prior/qw_m{}+pz_m{}".format(i, j): coherence_quad_main[i][j]}, step=epoch)
                    wandb.log({"Condition_Self_Cohere_Quad_prior/crossClf_qw_m{}+pz_m{}".format(i, j): coherence_quad_cross[i][j]}, step=epoch)
                else:
                    wandb.log({"Condition_Cros_Cohere_Quad_prior/qw_m{}+pz_m{}".format(i, j): coherence_quad_main[i][j]}, step=epoch)
                    wandb.log({"Condition_Cros_Cohere_Quad_prior/crossClf_qw_m{}+pz_m{}".format(i, j): coherence_quad_cross[i][j]}, step=epoch)

        coherence_digit_main_post, coherence_digit_cross_post, means_selfcoh_digit_main_post, means_selfcoh_digit_cross_post, \
            means_cctarget_digit_main_post, means_cctarget_digit_cross_post, cctarget_all_digit_main_post, cctarget_all_digit_cross_post, \
                meanall_digit_main_post, meanall_digit_cross_post = self_and_cross_coherence_calculation(
                    model, test_time_loader, clfs_digit, clfs_quadrant, device, condition_type='shared', rsample_type='posterior'
                    )
        coherence_quad_main_post, coherence_quad_cross_post, means_selfcoh_quad_main_post, means_selfcoh_quad_cross_post, \
            means_cctarget_quad_main_post, means_cctarget_quad_cross_post, cctarget_all_quad_main_post, cctarget_all_quad_cross_post, \
                meanall_quad_main_post, meanall_quad_cross_post = self_and_cross_coherence_calculation(
                    model, test_time_loader, clfs_quadrant, clfs_digit, device, condition_type='private', rsample_type='posterior'
                )
        wandb.log({"Condition_Cohere_Digit_post/self_mean(N/A_MMVAE+)": means_selfcoh_digit_main_post}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_post/target_meanall": cctarget_all_digit_main_post}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_post/mean_All(N/A_MMVAE+)": meanall_digit_main_post}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_post/self_mean_crossClf(N/A_MMVAE+)": means_selfcoh_digit_cross_post}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_post/target_meanall_crossClf": cctarget_all_digit_cross_post}, step=epoch)
        wandb.log({"Condition_Cohere_Digit_post/mean_All_crossClf(N/A_MMVAE+)": meanall_digit_cross_post}, step=epoch)

        wandb.log({"Condition_Self_Cohere_Quad_post/self_mean(N/A_MMVAE+)": means_selfcoh_quad_main_post}, step=epoch) 
        wandb.log({"Condition_Cros_Cohere_Quad_post/target_meanall": cctarget_all_quad_main_post}, step=epoch)                
        wandb.log({"Condition_Cros_Cohere_Quad_post/mean_All(N/A_MMVAE+)": meanall_quad_main_post}, step=epoch)
        wandb.log({"Condition_Self_Cohere_Quad_post/self_mean_crossClf(N/A_MMVAE+)": means_selfcoh_quad_cross_post}, step=epoch)
        wandb.log({"Condition_Cros_Cohere_Quad_post/target_meanall_crossClf": cctarget_all_quad_cross_post}, step=epoch)
        wandb.log({"Condition_Cros_Cohere_Quad_post/mean_All_crossClf(N/A_MMVAE+)": meanall_quad_cross_post}, step=epoch)

        for i in range(NUM_VAES):
            wandb.log({"Condition_Cohere_Digit_post/target_qw_m{}".format(i): means_cctarget_digit_main_post[i]}, step=epoch)
            wandb.log({"Condition_Cohere_Digit_post/target_crossClf_qw_m{}".format(i): means_cctarget_digit_cross_post[i]}, step=epoch)


            wandb.log({"Condition_Cros_Cohere_Quad_post/target_qz_m{}".format(i): means_cctarget_quad_main_post[i]}, step=epoch)
            wandb.log({"Condition_Cros_Cohere_Quad_post/target_crossClf_qz_m{}".format(i): means_cctarget_quad_cross_post[i]}, step=epoch)

            for j in range(NUM_VAES):
                wandb.log({"Condition_Cohere_Digit_post/qz_m{}+qw_m{}".format(i, j): coherence_digit_main_post[i][j]}, step=epoch)
                wandb.log({"Condition_Cohere_Digit_post/crossClf_qz_m{}+qw_m{}".format(i, j): coherence_digit_cross_post[i][j]}, step=epoch)
                if i == j:
                    wandb.log({"Condition_Self_Cohere_Quad_post/qw_m{}+qz_m{}".format(i, j): coherence_quad_main_post[i][j]}, step=epoch)
                    wandb.log({"Condition_Self_Cohere_Quad_post/crossClf_qw_m{}+qz_m{}".format(i, j): coherence_quad_cross_post[i][j]}, step=epoch)
                else:
                    wandb.log({"Condition_Cros_Cohere_Quad_post/qw_m{}+qz_m{}".format(i, j): coherence_quad_main_post[i][j]}, step=epoch)
                    wandb.log({"Condition_Cros_Cohere_Quad_post/crossClf_qw_m{}+qz_m{}".format(i, j): coherence_quad_cross_post[i][j]}, step=epoch)

    # Latent Classification
    if args.enable_latent_classification:
        # digit (shared, Z)
        clf_lr_digit = train_clf_lr_multi_labelType(model, subtraining_loader, device, args, condition_type='shared')
        # quadrant (modality-specific/private, W)
        clf_lr_quadrant = train_clf_lr_multi_labelType(model, subtraining_loader, device, args, condition_type='private')
        accuracies_lc_digit = linear_latent_classification_multi_labelType(model, test_time_loader, clf_lr_digit, device, args, condition_type='shared')
        accuracies_lc_quadrant = linear_latent_classification_multi_labelType(model, test_time_loader, clf_lr_quadrant, device, args, condition_type='private')
        for key in accuracies_lc_digit:
            wandb.log({"LatentClassAcc_Digit_clf/" + key: accuracies_lc_digit[key]}, step=epoch)
        for key in accuracies_lc_quadrant:
            if key == 'confidence_data':
                continue # skip confidence_data
            elif key == 'confidence_figs':
                continue # skip confidence_figs
            else:
                wandb.log({"LatentClassAcc_Quad_clf/" + key: accuracies_lc_quadrant[key]}, step=epoch)


    if args.enable_unconditional_generation:
        # # Generate unconditional samples
        gen_samples, combined_grid = polymnist_generate_unconditional_plot(model, num_rows=1, num_cols=10)
        wandb.log({"Visualization_Samples/Unconditional_Generations_Grid": wandb.Image(combined_grid)}, step=epoch)
        wandb.log({'IMG_Unconditional_Generations/Combined_Grid': wandb.Image(combined_grid)}, step=epoch)
        for j in range(NUM_VAES):
            wandb.log({'IMG_Unconditional_Generations/m{}'.format(j) :  wandb.Image(gen_samples[j])}, step=epoch)
        # -------
        # Calculate unconditional coherence
        uncond_coher = unconditional_coherence(model, test_time_loader, clfs_digit, device)
        wandb.log({"Unconditional_Coherence/prior": uncond_coher}, step=epoch)

    # Calculate FID scores
    if args.enable_fid:
        calculate_fid_routine(datadirPM, fid_path, 10000, epoch, model, test_time_loader, device, args)

    if args.enable_tSNE_UMAP:
        visualize_ratio = 0.25 # 0.25 for 25% of the data, 1.0 for all data

        view0_z_digit, view0_w_digit_mis = visualize_latents_with_priors(
                                            model, model.vaes[0], model.encoders[0], validating_loader, testing_loader, 
                                            "z_digit_0", "w_digit_mis_0", device=device, figure=1, condition_type='shared', 
                                            view_index=0, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                            save_file=os.path.join(args.tSNE_save_dir, 
                                                                'view0_shared_digit.png')) # device=device
        view1_z_digit, view1_w_digit_mis = visualize_latents_with_priors(
                                                model, model.vaes[1], model.encoders[1], validating_loader, testing_loader,
                                                "z_digit_1", "w_digit_mis_1", device=device, figure=2, condition_type='shared',
                                                view_index=1, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view1_shared_digit.png'))
        view2_z_digit, view2_w_digit_mis = visualize_latents_with_priors(
                                                model, model.vaes[2], model.encoders[2], validating_loader, testing_loader,
                                                "z_digit_2", "w_digit_mis_2", device=device, figure=3, condition_type='shared', 
                                                view_index=2, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view2_shared_digit.png'))
        view3_z_digit, view3_w_digit_mis = visualize_latents_with_priors(
                                                model, model.vaes[3], model.encoders[3], validating_loader, testing_loader,
                                                "z_digit_3", "w_digit_mis_3", device=device, figure=4, condition_type='shared', 
                                                view_index=3, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view3_shared_digit.png'))
        view4_z_digit, view4_w_digit_mis = visualize_latents_with_priors(
                                                model, model.vaes[4], model.encoders[4], validating_loader, testing_loader,
                                                "z_digit_4", "w_digit_mis_4", device=device, figure=5, condition_type='shared', 
                                                view_index=4, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view4_shared_digit.png'))

        view0_w_quadrant, view0_z_quadrant_mis = visualize_latents_with_priors(
                                                model, model.vaes[0], model.encoders[0], validating_loader, testing_loader, 
                                                "w_quadrant_0", "z_quadrant_mis_0", device=device, figure=6, condition_type='private', 
                                                view_index=0, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view0_private_quadrant.png'))
        view1_w_quadrant, view1_z_quadrant_mis = visualize_latents_with_priors(
                                                model, model.vaes[1], model.encoders[1], validating_loader, testing_loader, 
                                                "w_quadrant_1", "z_quadrant_mis_1", device=device, figure=7, condition_type='private', 
                                                view_index=1, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view1_private_quadrant.png'))
        view2_w_quadrant, view2_z_quadrant_mis = visualize_latents_with_priors(
                                                model, model.vaes[2], model.encoders[2], validating_loader, testing_loader, 
                                                "w_quadrant_2", "z_quadrant_mis_2", device=device, figure=8, condition_type='private', 
                                                view_index=2, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view2_private_quadrant.png'))
        view3_w_quadrant, view3_z_quadrant_mis = visualize_latents_with_priors(
                                                model, model.vaes[3], model.encoders[3], validating_loader, testing_loader, 
                                                "w_quadrant_3", "z_quadrant_mis_3", device=device, figure=9, condition_type='private', 
                                                view_index=3, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view3_private_quadrant.png'))
        view4_w_quadrant, view4_z_quadrant_mis = visualize_latents_with_priors(
                                                model, model.vaes[4], model.encoders[4], validating_loader, testing_loader, 
                                                "w_quadrant_4", "z_quadrant_mis_4", device=device, figure=10, condition_type='private', 
                                                view_index=4, batch_size=args.batch_size, visualize_ratio=visualize_ratio,
                                                save_file=os.path.join(args.tSNE_save_dir, 
                                                                    'view4_private_quadrant.png'))

            
        # log tSNE images to wandb
        wandb.log({'tSNE_or_UMAP/m0_z_digit': wandb.Image(view0_z_digit)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m1_z_digit': wandb.Image(view1_z_digit)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m2_z_digit': wandb.Image(view2_z_digit)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m3_z_digit': wandb.Image(view3_z_digit)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m4_z_digit': wandb.Image(view4_z_digit)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m0_w_digit_mis': wandb.Image(view0_w_digit_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m1_w_digit_mis': wandb.Image(view1_w_digit_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m2_w_digit_mis': wandb.Image(view2_w_digit_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m3_w_digit_mis': wandb.Image(view3_w_digit_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m4_w_digit_mis': wandb.Image(view4_w_digit_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m0_w_quadrant': wandb.Image(view0_w_quadrant)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m1_w_quadrant': wandb.Image(view1_w_quadrant)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m2_w_quadrant': wandb.Image(view2_w_quadrant)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m3_w_quadrant': wandb.Image(view3_w_quadrant)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m4_w_quadrant': wandb.Image(view4_w_quadrant)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m0_z_quadrant_mis': wandb.Image(view0_z_quadrant_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m1_z_quadrant_mis': wandb.Image(view1_z_quadrant_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m2_z_quadrant_mis': wandb.Image(view2_z_quadrant_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m3_z_quadrant_mis': wandb.Image(view3_z_quadrant_mis)}, step=epoch)
        wandb.log({'tSNE_or_UMAP/m4_z_quadrant_mis': wandb.Image(view4_z_quadrant_mis)}, step=epoch)

if __name__ == '__main__':
    if args.test_only:
        print("--- Running in Test-Only Mode ---")
        checkpoint_to_load = args.checkpoint_path
        print(f"Loading checkpoint: {checkpoint_to_load}")
        if not os.path.exists(checkpoint_to_load):
            print(f"Error: Checkpoint file not found at '{checkpoint_to_load}'. Exiting.")
            sys.exit(1)
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
        # Optimizer state loading is intentionally skipped in test-only mode.

        run_evaluation(epoch_to_test)
        
        print("--- Test-Only Mode Finished ---")

    else:
        # --- Training Mode ---
        for epoch in range(start_epoch, args.epochs + 1):
            train(epoch)

            test_epoch_freq = 1
            if epoch % test_epoch_freq == 0:
                run_evaluation(epoch)

                # Save checkpoint (light version)
                save_model_light(model, runPath + '/model_' + str(epoch) + '.rar')
                # Optional: Save optimizer state for better resuming
                torch.save(optimizer.state_dict(), os.path.join(runPath, f'optimizer_{epoch}.rar'))
