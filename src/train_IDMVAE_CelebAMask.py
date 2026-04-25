
import os
import argparse
import sys
import json
import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image
from objectives import compute_idmvae_loss
from utils import Logger, save_model_light, get_mean
import wandb
from dataset_CelebAMask_HQ import CelebAHQMaskDS
from models.idmvae_CelebAMask import CelebA_IDMVAE
from utils import CrossModalEvalForwardMode
from eval_functions import idmvae_generate_unconditional, idmvae_self_and_cross_modal_generation_eval
from eval_functions_CelebAMask import celeba_self_and_cross_modal_generation_eval
from sklearn.metrics import f1_score
import shutil
import glob
import re
from fid.fid_score import calculate_fid_given_paths

# Deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parser = argparse.ArgumentParser(description='IDMVAE CelebAMask')
parser.add_argument('--experiment', type=str, default='CelebA_IDMVAE', metavar='E',
                    help='experiment name')
parser.add_argument('--datadir', type=str, default='/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM/',
                    help='path to dataset')
parser.add_argument('--K', type=int, default=1,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size for data')
parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='number of epochs to train')
parser.add_argument('--latent-dim-w', type=int, default=128, metavar='L',
                    help='latent dimensionality w')
parser.add_argument('--latent-dim-z', type=int, default=128, metavar='L',
                    help='latent dimensionality z')
parser.add_argument('--print-freq', type=int, default=50, metavar='f',
                    help='frequency with which to print stats')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed')
parser.add_argument('--beta', type=float, default=5.0)
parser.add_argument('--priorposterior', type=str, default='Laplace', choices=['Normal', 'Laplace'],
                    help='distribution choice for prior and posterior')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--outputdir', type=str, default='../outputs', help='Output directory')
parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
parser.add_argument('--diffusion_loss_weight', type=float, default=0.0, help='loss weight for diffusion')
parser.add_argument('--diffusion_stop_grad_on_input', action='store_true', default=False, help='Stop gradient on input for diffusion prior')
parser.add_argument('--skip_train', action='store_true', default=False, help='Disable training loop')
parser.add_argument('--test_freq', type=int, default=1, help='Frequency of testing')
parser.add_argument('--eval_freq', type=int, default=5, help='Frequency of evaluation')
parser.add_argument('--qualitative_freq', type=int, default=1, help='Frequency of qualitative evaluation (0 to disable)')
parser.add_argument('--f1_freq', type=int, default=1, help='Frequency of calculating F1 score (0 to disable)')
parser.add_argument('--fid_freq', type=int, default=1, help='Frequency of calculating FID score (0 to disable)')
parser.add_argument('--ckpt_freq', type=int, default=25, help='Frequency of saving checkpoints')
parser.add_argument('--run_note', type=str, default='', help='Run note')
parser.add_argument('--debug_loader', action='store_true', default=False, help='Use validation dataset for training to debug')

# Develop and Test modes
parser.add_argument('--develop', action='store_true', default=False,
                    help='If set, run in develop mode with a fresh run ID and output directory for quick debugging iteration.')
parser.add_argument('--resume_from_CPt_runId', action='store_true', default=False,
                    help='Resume training from an old runId, specified by --CPt_runId.')
parser.add_argument('--CPt_runId', type=str, default='',
                    help='Run ID to resume from if --resume_from_CPt_runId is set.')
parser.add_argument('--test-only', action='store_true', default=False,
                    help='Load checkpoint(s) and run evaluation only, without training.')
parser.add_argument('--checkpoint-path', action='append', default=[],
                    help='Path to a model checkpoint .rar; repeat flag for multiple checkpoints (--test-only).')
parser.add_argument('--val-test-dataset', type=str, default='val', choices=['val', 'test'],
                    help='Dataset to use for testing/evaluation (val or test)')
parser.add_argument('--eval-shuffle', action='store_true', default=False,
                    help='Shuffle the evaluation dataset.')
parser.add_argument('--max-fid-images', type=int, default=None,
                    help='Maximum number of images for FID calculation. If None or larger than dataset, uses full dataset.')
parser.add_argument('--print-params-only', action='store_true', default=False,
                    help='Instantiate model (optionally load a checkpoint) and print parameter counts, then exit.')
parser.add_argument('--eval-fusion-random', action='store_true', default=False,
                    help='Enable random selection for multi-modal fusion evaluation')
parser.add_argument('--eval-fusion-average', action='store_true', default=False,
                    help='Enable latent averaging for multi-modal fusion evaluation')

# IDMVAE regularization args
parser.add_argument('--cross_mi_loss_scale', type=float, default=0.0, help='Scale for cross-view MI loss')
parser.add_argument('--gen_aug_loss_scale', type=float, default=0.0, help='Scale for generative augmentation loss')
parser.add_argument('--gen_aug_sampling_scheme', type=str, default='posterior', choices=['posterior', 'prior', 'diffusion_prior'])
parser.add_argument('--gen_aug_loss_type', type=str, default='CL', choices=['CL', 'ML'])
parser.add_argument('--denoiser_ckpt', type=str, default=None,
                    help='Path to a pretrained DiT denoiser checkpoint (.pt) for denoised generation plotting.')
parser.add_argument('--denoiser_num_sampling_steps', type=int, default=250,
                    help='Number of diffusion sampling steps when running the pretrained denoiser.')
parser.add_argument('--save_eval_images_root', type=str, default='',
                    help='If set, save original/recon/denoised images during evaluation under this root directory.')

args = parser.parse_args()

# Validate arguments
if not args.print_params_only:
    if args.resume and args.test_only:
        parser.error("--resume and --test-only cannot be used together.")
    if args.test_only and not args.checkpoint_path:
        parser.error("When using --test-only, provide at least one --checkpoint-path.")

    if args.checkpoint_path and not args.test_only:
        parser.error("--checkpoint-path is only valid with --test-only.")
else:
    if args.resume or args.test_only:
        parser.error("--print-params-only cannot be combined with --resume or --test-only.")

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Logging
base_id = f"K{args.K}_B{args.batch_size}_w{args.latent_dim_w}_z{args.latent_dim_z}_{args.priorposterior}_b{args.beta}_lw{args.diffusion_loss_weight}_{args.gen_aug_loss_scale}_{args.cross_mi_loss_scale}_s{args.seed}"

if args.resume and args.resume_from_CPt_runId:
    runId = args.CPt_runId
elif args.develop:
    if args.run_note:
        runId = f"Dev_{args.run_note}_{base_id}"
    else:
        runId = f"Dev_{base_id}"
else:
    if args.run_note:
        runId = f"{args.run_note}_{base_id}"
    else:
        runId = base_id

if args.test_only:
    experiment_dir = os.path.join(args.outputdir, args.experiment, "checkpoints", "CP_test")
elif args.develop:
    experiment_dir = os.path.join(args.outputdir, args.experiment, "checkpoints", "Dev")
else:
    experiment_dir = os.path.join(args.outputdir, args.experiment, "checkpoints")

os.makedirs(experiment_dir, exist_ok=True)
runPath = os.path.join(experiment_dir, runId)

if args.develop:
    if os.path.isdir(runPath):
        print(f"Warning: Run path '{runPath}' already exists. Removing it for a fresh develop run.")
        shutil.rmtree(runPath)
    else:
        print(f"Creating new run path for develop mode: '{runPath}'")    
    os.makedirs(runPath, exist_ok=True)
    wandb_resume_status = None
elif args.test_only:
    os.makedirs(runPath, exist_ok=True)
    wandb_resume_status = "allow"
elif args.resume:
    if not os.path.exists(runPath):
        print(f"Error: Run path '{runPath}' does not exist for resuming. Exiting.")
        sys.exit(1)
    wandb_resume_status = "allow"
else:
    if os.path.exists(runPath):
        print(f"Error: Run path '{runPath}' already exists. Use --resume to continue or change parameters to start a new run.")
        sys.exit(1)
    os.makedirs(runPath)
    wandb_resume_status = None

sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
print('RunID:', runId)

with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)

wandb.init(project=args.experiment, config=vars(args), name=runId, id=runId, resume=wandb_resume_status)

print(f'Skip Train: {args.skip_train}')
print(f'Test Freq: {args.test_freq}')
print(f'Eval Freq: {args.eval_freq}, but dominated by test_only if > 0')
print(f'Qualitative Freq: {args.qualitative_freq}, but dominated by Eval Freq')
print(f'F1 Freq: {args.f1_freq}, but dominated by Eval Freq')
print(f'FID Freq: {args.fid_freq}, but dominated by Eval Freq')
print(f'Ckpt Freq: {args.ckpt_freq}')

# Model
model = CelebA_IDMVAE(args).to(device)

# Denoiser status (for denoised generation plotting)
if getattr(args, 'denoiser_ckpt', None):
    print(f"Denoiser for plotting: {'enabled' if model._has_denoiser() else 'disabled (checkpoint not found or load failed)'}")
else:
    print("Denoiser for plotting: disabled (no --denoiser_ckpt provided)")

def _log_param_counts(tag: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params/CelebA/{tag}] total={total:,} trainable={trainable:,}")

_log_param_counts("init")

if args.print_params_only:
    if args.checkpoint_path:
        ckpt = args.checkpoint_path[0]
        if not os.path.exists(ckpt):
            print(f"Error: Checkpoint file not found at '{ckpt}'. Exiting.")
            sys.exit(1)
        print(f"Loading checkpoint for parameter count: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        _log_param_counts("print_only_checkpoint")
    sys.exit(0)

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)

# Data Loaders
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
g = torch.Generator()
g.manual_seed(0)
kwargs['generator'] = g

if args.debug_loader:
    print("WARNING: Using validation dataset for training (debug_loader=True)")
    train_dataset = CelebAHQMaskDS(size=128, datapath=args.datadir, ds_type='val')
else:
    train_dataset = CelebAHQMaskDS(size=128, datapath=args.datadir, ds_type='train')

if args.val_test_dataset == 'val':
    print("Using Validation dataset for evaluation.")
    eval_dataset = CelebAHQMaskDS(size=128, datapath=args.datadir, ds_type='val')
else:
    print("Using Test dataset for evaluation.")
    eval_dataset = CelebAHQMaskDS(size=128, datapath=args.datadir, ds_type='test')

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=args.eval_shuffle, **kwargs)

# Attributes to select (18 out of 40)
attr_visible = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]

new_id_to_attr = ['Bald',
        'Bangs',
        'Black_Hair',
        'Blond_Hair',
        'Brown_Hair',
        'Bushy_Eyebrows',
        'Eyeglasses',
        'Gray_Hair',
        'Heavy_Makeup',
        'Male',
        'Mouth_Slightly_Open',
        'Mustache',
        'Pale_Skin',
        'Receding_Hairline',
        'Smiling',
        'Straight_Hair',
        'Wavy_Hair',
        'Wearing_Hat',
]

def generate_from_multiple(model, data, present_indices, target_index, method='random'):
    """
    Generate target modality from multiple present modalities.
    Args:
        model: IDMVAE model
        data: list of input tensors for all modalities
        present_indices: list of indices of present modalities
        target_index: index of target modality
        method: 'random' or 'average'
    Returns:
        generated tensor for target modality (mean of distribution)
    """
    with torch.no_grad():
        # Get latents for all modalities
        # self_and_cross_modal_generation_forward returns qu_xs, px_us, uss
        # uss[m] is the latent for modality m
        qu_xs, px_us, uss = model.self_and_cross_modal_generation_forward(data, K=args.K)
        
        if method == 'random':
            # Randomly select one present modality
            idx = np.random.choice(present_indices)
            # Return the cross-reconstruction from idx to target_index
            # px_us[idx][target_index] is the distribution
            return get_mean(px_us[idx][target_index])
            
        elif method == 'average':
            # Average the shared latents (z) from present modalities
            z_list = []
            for idx in present_indices:
                us = uss[idx]
                # Split us into w and z
                _, z = torch.split(us, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
                z_list.append(z)
            
            # Average z
            z_avg = torch.mean(torch.stack(z_list), dim=0)
            
            # Sample w for target modality from prior
            if model.diffusion_loss_weight > 0.0:
                pw = model.pws_diffusion[target_index]
            else:
                pw = model.get_simple_prior_w(view=target_index, aux=False)
            
            # Sample w
            # z_avg shape: [K, B, latent_dim_z]
            # We need to sample w with same shape
            w_new = pw.rsample(torch.Size([z_avg.size(0), z_avg.size(1)])).squeeze(2)
            
            # Combine w and z
            u_new = torch.cat((w_new, z_avg), dim=-1)
            
            # Decode
            target_vae = model.vaes[target_index]
            px_u = target_vae.px_u(*target_vae.dec(u_new))
            
            return get_mean(px_u)
        
        else:
            raise ValueError(f"Unknown method: {method}")

def unpack_data(data, device='cuda'):
    data = [d.to(device) for d in data]
    # Select specific attributes
    data[2] = data[2][:, attr_visible].float()
    return data

def train(epoch):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        data = unpack_data(dataT, device=device)
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
        
        if i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item()))

    epoch_loss = b_loss / len(train_loader.dataset)
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))
    wandb.log({"Loss/train": epoch_loss}, step=epoch)

def test(epoch):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(eval_loader):
            data = unpack_data(dataT, device=device)
            bs = data[0].size(0)

            loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(model, data, K=args.K, test=True)

            wandb.log({"Loss/test_loss": loss}, step=epoch)
            wandb.log({"Loss/test_recon_kl_sum": recon_kl_sum_loss}, step=epoch)
            wandb.log({"Loss/test_likelihood": llik_recon_loss}, step=epoch)
            wandb.log({"Loss/test_kl": kl_div_loss}, step=epoch)
            wandb.log({"Loss/test_cross_mi": cross_mi_loss}, step=epoch)
            wandb.log({"Loss/test_gen_aug": gen_aug_loss}, step=epoch)
            wandb.log({"Loss/test_diffusion_loss": diffusion_loss.item()}, step=epoch)
            
            b_loss += loss.item() * bs

    epoch_loss = b_loss / len(eval_loader.dataset)
    print('====>             Test loss: {:.4f}'.format(epoch_loss))
    wandb.log({"Loss/test": epoch_loss}, step=epoch)

def run_evaluation(epoch):
    model.eval()
    # Qualitative evaluation
    # Qualitative block is optional and controlled by qualitative_freq.
    if (args.qualitative_freq > 0) and (args.test_only or epoch % args.qualitative_freq == 0):
        with torch.no_grad():
            # Get a batch
            dataT = next(iter(eval_loader))
            data = unpack_data(dataT, device=device)
                        
            # Posterior rsample generation (shuffled shared)
            # Similar to train_IDMVAE_CUB.py
            cg_imgs_Shared_post, cg_imgs_Shared_post_ext, cg_imgs_Shared_post_ext_pruned, _ = celeba_self_and_cross_modal_generation_eval(
                model, data, 8, 8, mode=CrossModalEvalForwardMode.POSTERIOR_CTRL, condition_type='shared', prune_cols=[0, 1, 2, 4, 7], prune_rows=None)
            cg_imgs_Shared_post_denoised = getattr(model, 'last_denoised_posterior_grids', None)
            cg_imgs_Shared_post_ext_denoised = getattr(model, 'last_denoised_posterior_extended_grids', None)
            cg_imgs_Shared_post_ext_pruned_denoised = getattr(model, 'last_denoised_posterior_extended_pruned_grids', None)

            # Log to wandb
            # We have 3 modalities: Image(0), Mask(1), Attr(2).
            # We only visualize 0 and 1.
            for i in [0, 1]:
                for j in [0, 1]:
                    if cg_imgs_Shared_post[i][j] is not None:
                        wandb.log({'IMG_Self&Cros_Gen_Cluster_post/qz_m{}+qw_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Shared_post[i][j])}, step=epoch)
                    if cg_imgs_Shared_post_denoised and cg_imgs_Shared_post_denoised[i][j] is not None:
                        wandb.log({'IMG_Self&Cros_Gen_Cluster_post/qz_m{}+qw_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Shared_post_denoised[i][j])}, step=epoch)
                    if cg_imgs_Shared_post_ext[i][j] is not None:
                        wandb.log({'IMG_Self&Cros_Gen_Cluster_post_extended/qz_m{}+qw_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Shared_post_ext[i][j])}, step=epoch)
                    if cg_imgs_Shared_post_ext_denoised and cg_imgs_Shared_post_ext_denoised[i][j] is not None:
                        wandb.log({'IMG_Self&Cros_Gen_Cluster_post_extended/qz_m{}+qw_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Shared_post_ext_denoised[i][j])}, step=epoch)
                    if cg_imgs_Shared_post_ext_pruned[i][j] is not None:
                        wandb.log({'IMG_Self&Cros_Gen_Cluster_post_extended_pruned/qz_m{}+qw_m{}_shuf'.format(i, j): wandb.Image(cg_imgs_Shared_post_ext_pruned[i][j])}, step=epoch)
                    if cg_imgs_Shared_post_ext_pruned_denoised and cg_imgs_Shared_post_ext_pruned_denoised[i][j] is not None:
                        wandb.log({'IMG_Self&Cros_Gen_Cluster_post_extended_pruned/qz_m{}+qw_m{}_shuf_denoised'.format(i, j): wandb.Image(cg_imgs_Shared_post_ext_pruned_denoised[i][j])}, step=epoch)

    # Quantitative evaluation (F1 and FID)
    do_f1 = (args.f1_freq > 0) and (args.test_only or epoch % args.f1_freq == 0)
    do_fid = (args.fid_freq > 0) and (args.test_only or epoch % args.fid_freq == 0)

    if do_f1 or do_fid:
        print(f"Starting quantitative evaluation (F1: {do_f1}, FID: {do_fid})...")
        
        # Initialize F1 containers
        if do_f1:
            # Shared Ground Truth
            true_attr_list = []
            true_mask_list = []
            
            # Unconditional Generated
            uncond_gen_attr_list = []
            uncond_gen_mask_list = []
            
            # Conditional Generated
            cond_gen_attr_given_img_list = []
            cond_gen_attr_given_mask_list = []
            cond_gen_attr_given_attr_list = []
            cond_gen_mask_given_img_list = []
            cond_gen_mask_given_attr_list = []
            cond_gen_mask_given_mask_list = []

            # Multi-modal Conditional Generated
            if args.eval_fusion_random:
                cond_gen_attr_given_img_mask_rand_list = []
                cond_gen_mask_given_img_attr_rand_list = []
            if args.eval_fusion_average:
                cond_gen_attr_given_img_mask_avg_list = []
                cond_gen_mask_given_img_attr_avg_list = []

        # Initialize FID directories
        if do_fid:
            fid_real_dir = os.path.join(runPath, f"fid_real_epoch{epoch}")
            fid_uncond_gen_dir = os.path.join(runPath, f"fid_uncond_gen_epoch{epoch}")
            fid_mask2img_dir = os.path.join(runPath, f"fid_mask2img_epoch{epoch}")
            fid_attr2img_dir = os.path.join(runPath, f"fid_attr2img_epoch{epoch}")
            fid_img2img_dir = os.path.join(runPath, f"fid_img2img_epoch{epoch}")
            
            fid_dirs_to_create = [fid_real_dir, fid_uncond_gen_dir, fid_mask2img_dir, fid_attr2img_dir, fid_img2img_dir]
            
            if args.eval_fusion_random:
                fid_mask_attr2img_rand_dir = os.path.join(runPath, f"fid_mask_attr2img_rand_epoch{epoch}")
                fid_dirs_to_create.append(fid_mask_attr2img_rand_dir)
            
            if args.eval_fusion_average:
                fid_mask_attr2img_avg_dir = os.path.join(runPath, f"fid_mask_attr2img_avg_epoch{epoch}")
                fid_dirs_to_create.append(fid_mask_attr2img_avg_dir)
            
            for d in fid_dirs_to_create:
                os.makedirs(d, exist_ok=True)
                
            fid_count = 0
            
            if args.max_fid_images is None or args.max_fid_images > len(eval_loader.dataset):
                max_fid_images = len(eval_loader.dataset)
            else:
                max_fid_images = args.max_fid_images
            
            print(f"FID: Generating max {max_fid_images} images (Dataset size: {len(eval_loader.dataset)})")

        with torch.no_grad():
            for i, dataT in enumerate(eval_loader):
                data = unpack_data(dataT, device=device)
                bs = data[0].size(0)
                
                # Pre-process Ground Truth for F1 (shared)
                if do_f1:
                    # Attributes
                    gt_attr = (data[2].cpu() > 0.5).int()
                    # Masks
                    gt_mask = (data[1].cpu() > 0.5).int()
                    gt_mask_flat = gt_mask.view(bs, -1)
                    
                    # Append to shared lists
                    true_attr_list.append(gt_attr)
                    true_mask_list.append(gt_mask_flat)
                
                # --- Unconditional Generation ---
                # We need unconditional samples if F1 is on OR if FID is on and we haven't reached max images
                if do_f1 or (do_fid and fid_count < max_fid_images):
                    uncond_samples = idmvae_generate_unconditional(model, bs)
                    
                    if do_f1:
                        # Attributes
                        uncond_gen_attr = (uncond_samples[2].cpu() > 0.5).int()
                        uncond_gen_attr_list.append(uncond_gen_attr)
                        
                        # Masks
                        uncond_gen_mask = (uncond_samples[1].cpu() > 0.5).int()
                        uncond_gen_mask_list.append(uncond_gen_mask.view(bs, -1))
                    
                    if do_fid and fid_count < max_fid_images:
                        real_imgs = data[0]
                        uncond_gen_imgs = uncond_samples[0]

                # --- Conditional Generation ---
                if do_f1 or (do_fid and fid_count < max_fid_images):
                    recons = idmvae_self_and_cross_modal_generation_eval(model, data)
                    # NOTE: the diagonal elements are self-reconstructions
                    
                    if do_f1:
                        # Attributes
                        cond_gen_attr_given_img = (recons[0][2].mean(0).cpu() > 0.5).int()
                        cond_gen_attr_given_img_list.append(cond_gen_attr_given_img)
                        
                        cond_gen_attr_given_mask = (recons[1][2].mean(0).cpu() > 0.5).int()
                        cond_gen_attr_given_mask_list.append(cond_gen_attr_given_mask)

                        cond_gen_attr_given_attr = (recons[2][2].mean(0).cpu() > 0.5).int()
                        cond_gen_attr_given_attr_list.append(cond_gen_attr_given_attr)
                        
                        # Masks
                        cond_gen_mask_given_img = (recons[0][1].mean(0).cpu() > 0.5).int()
                        cond_gen_mask_given_img_list.append(cond_gen_mask_given_img.view(cond_gen_mask_given_img.size(0), -1))
                        
                        cond_gen_mask_given_attr = (recons[2][1].mean(0).cpu() > 0.5).int()
                        cond_gen_mask_given_attr_list.append(cond_gen_mask_given_attr.view(cond_gen_mask_given_attr.size(0), -1))

                        cond_gen_mask_given_mask = (recons[1][1].mean(0).cpu() > 0.5).int()
                        cond_gen_mask_given_mask_list.append(cond_gen_mask_given_mask.view(cond_gen_mask_given_mask.size(0), -1))

                    # --- Multi-modal Generation ---
                    if args.eval_fusion_random:
                        # (Img, Mask) -> Attr (0, 1 -> 2)
                        gen_attr_from_img_mask_rand = generate_from_multiple(model, data, [0, 1], 2, method='random')
                        # (Img, Attr) -> Mask (0, 2 -> 1)
                        gen_mask_from_img_attr_rand = generate_from_multiple(model, data, [0, 2], 1, method='random')
                        # (Mask, Attr) -> Img (1, 2 -> 0)
                        gen_img_from_mask_attr_rand = generate_from_multiple(model, data, [1, 2], 0, method='random')

                        if do_f1:
                            cond_gen_attr_given_img_mask_rand = (gen_attr_from_img_mask_rand.mean(0).cpu() > 0.5).int()
                            cond_gen_attr_given_img_mask_rand_list.append(cond_gen_attr_given_img_mask_rand)
                            
                            cond_gen_mask_given_img_attr_rand = (gen_mask_from_img_attr_rand.mean(0).cpu() > 0.5).int()
                            cond_gen_mask_given_img_attr_rand_list.append(cond_gen_mask_given_img_attr_rand.view(bs, -1))

                    if args.eval_fusion_average:
                        # (Img, Mask) -> Attr (0, 1 -> 2)
                        gen_attr_from_img_mask_avg = generate_from_multiple(model, data, [0, 1], 2, method='average')
                        # (Img, Attr) -> Mask (0, 2 -> 1)
                        gen_mask_from_img_attr_avg = generate_from_multiple(model, data, [0, 2], 1, method='average')
                        # (Mask, Attr) -> Img (1, 2 -> 0)
                        gen_img_from_mask_attr_avg = generate_from_multiple(model, data, [1, 2], 0, method='average')

                        if do_f1:
                            cond_gen_attr_given_img_mask_avg = (gen_attr_from_img_mask_avg.mean(0).cpu() > 0.5).int()
                            cond_gen_attr_given_img_mask_avg_list.append(cond_gen_attr_given_img_mask_avg)
                            
                            cond_gen_mask_given_img_attr_avg = (gen_mask_from_img_attr_avg.mean(0).cpu() > 0.5).int()
                            cond_gen_mask_given_img_attr_avg_list.append(cond_gen_mask_given_img_attr_avg.view(bs, -1))

                    if do_fid and fid_count < max_fid_images:
                        mask2img = recons[1][0].mean(0)
                        attr2img = recons[2][0].mean(0)
                        img2img = recons[0][0].mean(0)
                        
                        # Save images
                        for j in range(bs):
                            if fid_count >= max_fid_images:
                                break
                            
                            # Save for Unconditional FID & Conditional FID (Real)
                            save_image(real_imgs[j], os.path.join(fid_real_dir, f"{fid_count}.png"))
                            save_image(uncond_gen_imgs[j], os.path.join(fid_uncond_gen_dir, f"{fid_count}.png"))
                            
                            # Save for Conditional FID
                            save_image(mask2img[j], os.path.join(fid_mask2img_dir, f"{fid_count}.png"))
                            save_image(attr2img[j], os.path.join(fid_attr2img_dir, f"{fid_count}.png"))
                            save_image(img2img[j], os.path.join(fid_img2img_dir, f"{fid_count}.png"))
                            
                            if args.eval_fusion_random:
                                mask_attr2img_rand = gen_img_from_mask_attr_rand.mean(0)
                                save_image(mask_attr2img_rand[j], os.path.join(fid_mask_attr2img_rand_dir, f"{fid_count}.png"))
                            
                            if args.eval_fusion_average:
                                mask_attr2img_avg = gen_img_from_mask_attr_avg.mean(0)
                                save_image(mask_attr2img_avg[j], os.path.join(fid_mask_attr2img_avg_dir, f"{fid_count}.png"))
                            
                            fid_count += 1

        # --- Calculate and Log F1 ---
        if do_f1:
            print("Calculating F1 scores...")
            # Shared Ground Truth
            true_attr_list = torch.cat(true_attr_list, dim=0).numpy()
            true_mask_list = torch.cat(true_mask_list, dim=0).numpy()

            # Unconditional
            uncond_gen_attr_list = torch.cat(uncond_gen_attr_list, dim=0).numpy()
            uncond_gen_mask_list = torch.cat(uncond_gen_mask_list, dim=0).numpy()
            
            f1_uncond_gen_attr = f1_score(true_attr_list, uncond_gen_attr_list, average='samples')
            f1_uncond_gen_mask = f1_score(true_mask_list, uncond_gen_mask_list, average='samples')
            
            print(f"Uncond_Gen F1 Score (Attr): {f1_uncond_gen_attr:.4f}")
            print(f"Uncond_Gen F1 Score (Mask): {f1_uncond_gen_mask:.4f}")
            
            wandb.log({
                "F1/Uncond_Attr": f1_uncond_gen_attr, 
                "F1/Uncond_Mask": f1_uncond_gen_mask
            }, step=epoch)

            # Conditional
            cond_gen_attr_given_img = torch.cat(cond_gen_attr_given_img_list, dim=0).numpy()
            cond_gen_attr_given_mask = torch.cat(cond_gen_attr_given_mask_list, dim=0).numpy()
            cond_gen_attr_given_attr = torch.cat(cond_gen_attr_given_attr_list, dim=0).numpy()
            
            cond_gen_mask_given_img = torch.cat(cond_gen_mask_given_img_list, dim=0).numpy()
            cond_gen_mask_given_attr = torch.cat(cond_gen_mask_given_attr_list, dim=0).numpy()
            cond_gen_mask_given_mask = torch.cat(cond_gen_mask_given_mask_list, dim=0).numpy()
            
            f1_attr_given_img = f1_score(true_attr_list, cond_gen_attr_given_img, average='samples')
            f1_attr_given_mask = f1_score(true_attr_list, cond_gen_attr_given_mask, average='samples')
            f1_attr_given_attr = f1_score(true_attr_list, cond_gen_attr_given_attr, average='samples')

            f1_mask_given_img = f1_score(true_mask_list, cond_gen_mask_given_img, average='samples')
            f1_mask_given_attr = f1_score(true_mask_list, cond_gen_mask_given_attr, average='samples')
            f1_mask_given_mask = f1_score(true_mask_list, cond_gen_mask_given_mask, average='samples')
            
            print(f"Cond_Gen F1 Score (Img->Attr): {f1_attr_given_img:.4f}")
            print(f"Cond_Gen F1 Score (Mask->Attr): {f1_attr_given_mask:.4f}")
            print(f"Cond_Gen F1 Score (Attr->Attr, recon): {f1_attr_given_attr:.4f}")
            print(f"Cond_Gen F1 Score (Img->Mask): {f1_mask_given_img:.4f}")
            print(f"Cond_Gen F1 Score (Attr->Mask): {f1_mask_given_attr:.4f}")
            print(f"Cond_Gen F1 Score (Mask->Mask, recon): {f1_mask_given_mask:.4f}")
            
            wandb_dict = {
                "F1/Img_to_Attr": f1_attr_given_img, 
                "F1/Mask_to_Attr": f1_attr_given_mask,
                "F1/Attr_to_Attr_recon": f1_attr_given_attr,
                "F1/Img_to_Mask": f1_mask_given_img,
                "F1/Attr_to_Mask": f1_mask_given_attr,
                "F1/Mask_to_Mask_recon": f1_mask_given_mask,
            }

            if args.eval_fusion_random:
                cond_gen_attr_given_img_mask_rand = torch.cat(cond_gen_attr_given_img_mask_rand_list, dim=0).numpy()
                cond_gen_mask_given_img_attr_rand = torch.cat(cond_gen_mask_given_img_attr_rand_list, dim=0).numpy()
                
                f1_attr_given_img_mask_rand = f1_score(true_attr_list, cond_gen_attr_given_img_mask_rand, average='samples')
                f1_mask_given_img_attr_rand = f1_score(true_mask_list, cond_gen_mask_given_img_attr_rand, average='samples')
                
                print(f"Cond_Gen F1 Score (Img+Mask->Attr) [Random]: {f1_attr_given_img_mask_rand:.4f}")
                print(f"Cond_Gen F1 Score (Img+Attr->Mask) [Random]: {f1_mask_given_img_attr_rand:.4f}")
                
                wandb_dict["F1/Img_Mask_to_Attr_rand"] = f1_attr_given_img_mask_rand
                wandb_dict["F1/Img_Attr_to_Mask_rand"] = f1_mask_given_img_attr_rand

            if args.eval_fusion_average:
                cond_gen_attr_given_img_mask_avg = torch.cat(cond_gen_attr_given_img_mask_avg_list, dim=0).numpy()
                cond_gen_mask_given_img_attr_avg = torch.cat(cond_gen_mask_given_img_attr_avg_list, dim=0).numpy()
                
                f1_attr_given_img_mask_avg = f1_score(true_attr_list, cond_gen_attr_given_img_mask_avg, average='samples')
                f1_mask_given_img_attr_avg = f1_score(true_mask_list, cond_gen_mask_given_img_attr_avg, average='samples')
                
                print(f"Cond_Gen F1 Score (Img+Mask->Attr) [Average]: {f1_attr_given_img_mask_avg:.4f}")
                print(f"Cond_Gen F1 Score (Img+Attr->Mask) [Average]: {f1_mask_given_img_attr_avg:.4f}")
                
                wandb_dict["F1/Img_Mask_to_Attr_avg"] = f1_attr_given_img_mask_avg
                wandb_dict["F1/Img_Attr_to_Mask_avg"] = f1_mask_given_img_attr_avg
            
            wandb.log(wandb_dict, step=epoch)

        # --- Calculate and Log FID ---
        if do_fid:
            print(f"Saved {fid_count} images for FID calculation.")
            try:
                # Unconditional
                fid_value = calculate_fid_given_paths([fid_real_dir, fid_uncond_gen_dir], 
                                                    batch_size=256, 
                                                    device=device, 
                                                    dims=2048)
                # Keep both console prints and wandb logs for debugging + tracking.
                print(f"FID (Uncond_Img): {fid_value}")
                wandb.log({"FID/Uncond_Img": fid_value}, step=epoch)
                
                # Conditional
                fid_mask2img = calculate_fid_given_paths([fid_real_dir, fid_mask2img_dir], batch_size=256, device=device, dims=2048)
                print(f"FID (Mask->Img): {fid_mask2img}")
                wandb.log({"FID/Mask_to_Img": fid_mask2img}, step=epoch)
                
                fid_attr2img = calculate_fid_given_paths([fid_real_dir, fid_attr2img_dir], batch_size=256, device=device, dims=2048)
                print(f"FID (Attr->Img): {fid_attr2img}")
                wandb.log({"FID/Attr_to_Img": fid_attr2img}, step=epoch)
                
                fid_img2img = calculate_fid_given_paths([fid_real_dir, fid_img2img_dir], batch_size=256, device=device, dims=2048)
                print(f"FID (Img->Img, recon): {fid_img2img}")
                wandb.log({"FID/Img_to_Img_recon": fid_img2img}, step=epoch)

                if args.eval_fusion_random:
                    fid_mask_attr2img_rand = calculate_fid_given_paths([fid_real_dir, fid_mask_attr2img_rand_dir], batch_size=256, device=device, dims=2048)
                    print(f"FID (Mask+Attr->Img) [Random]: {fid_mask_attr2img_rand}")
                    wandb.log({"FID/Mask_Attr_to_Img_rand": fid_mask_attr2img_rand}, step=epoch)

                if args.eval_fusion_average:
                    fid_mask_attr2img_avg = calculate_fid_given_paths([fid_real_dir, fid_mask_attr2img_avg_dir], batch_size=256, device=device, dims=2048)
                    print(f"FID (Mask+Attr->Img) [Average]: {fid_mask_attr2img_avg}")
                    wandb.log({"FID/Mask_Attr_to_Img_avg": fid_mask_attr2img_avg}, step=epoch)
                
            except Exception as e:
                print(f"FID calculation failed: {e}")
            
            # Cleanup
            for d in fid_dirs_to_create:
                shutil.rmtree(d)

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
    return {epoch for epoch in epochs if args.ckpt_freq > 0 and epoch % args.ckpt_freq == 0}

def _optimizer_special_epochs(epochs):
    saved_epochs = [epoch for epoch in epochs if args.ckpt_freq > 0 and epoch % args.ckpt_freq == 0]
    return {max(saved_epochs)} if saved_epochs else set()

if args.test_only:
    print("--- Running in Test-Only Mode ---")

    checkpoints_to_test = []
    for ckpt_path in args.checkpoint_path:
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint file not found at '{ckpt_path}'. Exiting.")
            sys.exit(1)
        match = re.search(r'model_(\d+).rar', ckpt_path)
        epoch = int(match.group(1)) if match else 999
        checkpoints_to_test.append((epoch, ckpt_path))

    for epoch_to_test, checkpoint_to_load in checkpoints_to_test:
        print(f"Loading model from {checkpoint_to_load}...")
        if not os.path.exists(checkpoint_to_load):
            print(f"Error: Checkpoint file not found at '{checkpoint_to_load}'. Skipping.")
            continue

        model.load_state_dict(torch.load(checkpoint_to_load, map_location=device), strict=False)
        _log_param_counts(f"test_only_epoch_{epoch_to_test}")

        test(epoch_to_test)
        run_evaluation(epoch_to_test)
    
    print("--- Test-Only Mode Finished ---")

else:
    start_epoch = 1
    if args.resume:
        model_checkpoints = _collect_checkpoints('model')
        if model_checkpoints:
            last_epoch, last_ckpt_path = model_checkpoints[-1]
            print(f"Resuming from epoch {last_epoch} (checkpoint: {last_ckpt_path})")
            
            # Load weights into the existing model (load_model_light() builds a new model; use state_dict here)
            model.load_state_dict(torch.load(last_ckpt_path, map_location=device), strict=False)
            
            # Load optimizer
            optimizer_ckpt_path = os.path.join(runPath, f'optimizer_{last_epoch}.rar')
            if os.path.exists(optimizer_ckpt_path):
                print(f"Loading optimizer state from {optimizer_ckpt_path}")
                optimizer.load_state_dict(torch.load(optimizer_ckpt_path))
            else:
                print(f"Warning: Optimizer checkpoint not found at {optimizer_ckpt_path}")
                
            start_epoch = last_epoch + 1
        else:
            print("Resume requested but no checkpoints found. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        if not args.skip_train:
            train(epoch)
        if args.test_freq > 0 and epoch % args.test_freq == 0:
            test(epoch)
        
        # Check if any evaluation is needed, but eval_freq will dominate
        is_eval_epoch = (args.eval_freq > 0 and epoch % args.eval_freq == 0) and ( \
                        (args.qualitative_freq > 0 and epoch % args.qualitative_freq == 0) or \
                        (args.f1_freq > 0 and epoch % args.f1_freq == 0) or \
                        (args.fid_freq > 0 and epoch % args.fid_freq == 0))
                        
        if is_eval_epoch:
            run_evaluation(epoch)
        
        model_checkpoint_path = os.path.join(runPath, f'model_{epoch}.rar')
        optimizer_checkpoint_path = os.path.join(runPath, f'optimizer_{epoch}.rar')
        save_model_light(model, model_checkpoint_path)
        torch.save(optimizer.state_dict(), optimizer_checkpoint_path)

        _prune_checkpoints('model', keep_recent=2, special_epochs_fn=_model_special_epochs)
        _prune_checkpoints('optimizer', keep_recent=2, special_epochs_fn=_optimizer_special_epochs)

