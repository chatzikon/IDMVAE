# Train IDMVAE model on TCGA dataset
import os
# Deterministic behavior:
# https://pytorch.org/docs/stable/notes/randomness.html
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Set before importing torch

import argparse
import numpy as np
import torch
from torch import optim

from models.idmvae_TCGA import IDMVAE_TCGA
from objectives import compute_idmvae_loss

from utils import unpack_data_PM as unpack_data
from dataset_TCGA_2_complete_views import TCGA2CompleteViews
from eval_functions_TCGA import eval_tcga_latent_classifiers, train_tcga_latent_classifiers

import wandb
import sys


parser = argparse.ArgumentParser(description='IDMVAE TCGA Experiment')
parser.add_argument('--experiment', type=str, default='IDMVAE_TCGA', metavar='E',
                    help='experiment name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--dataset', type=str, default='TCGA', help='dataset name')

parser.add_argument('--priorposterior', type=str, default='Laplace', choices=['Normal', 'Laplace', 'Diffusion'],
                    help='distribution choice for prior and posterior')
parser.add_argument('--diffusion_loss_weight', type=float, default=0.1,
                    help='weight for latent diffusion prior loss (0 disables)')
parser.add_argument('--diffusion_stop_grad_on_input', action='store_true', default=False,
                    help='stop gradient on x_start of diffusion')
parser.add_argument('--likelihood', type=str, default='Laplace', choices=['Normal', 'Laplace'],
                    help='distribution choice for likelihood')
parser.add_argument('--beta', type=float, default=2.5,
                    help='beta hyperparameter in VAE objective')
parser.add_argument('--K', type=int, default=1,
                    help='number of samples when resampling in the latent space')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--input_dim', type=int, default=100,
                    help='Dimensionality of the input data')
parser.add_argument('--latent_dim_w', type=int, default=32,
                    help='latent modality-specific dimensionality')
parser.add_argument('--latent_dim_z', type=int, default=16,
                    help='latent shared dimensionality')

parser.add_argument('--cross_mi_loss_scale', type=float, default=10.0,
                    help='Scale for cross-view MI loss')
parser.add_argument('--gen_aug_loss_scale', type=float, default=0.001,
                    help='Scale for generative augmentation loss')
parser.add_argument('--gen_aug_sampling_scheme', type=str, default='posterior', choices=['posterior', 'prior', 'diffusion_prior'],
                    help='Sampling scheme for generative augmentation loss: posterior, prior, or diffusion_prior')
parser.add_argument('--gen_aug_loss_type', type=str, default='CL', choices=['CL', 'ML'],
                    help='Type of generative augmentation loss: CL (Contrastive Loss) or ML (Matching Loss)')

parser.add_argument('--datadir', type=str, default='./data',
                    help='Directory where data is stored')
parser.add_argument('--split', type=str, choices=['0', '1', '2', '3', '4'], help='Data split to use')
parser.add_argument('--runId', type=str, default='', help='Run ID for this experiment')
parser.add_argument('--checkpoint_dir', type=str, default='/data/backed_up/shared/Data/TCGA/ckpts/',
                    help='Directory to save/load checkpoints')


args = parser.parse_args()

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

model = IDMVAE_TCGA(args).to(device)

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

# Set parameters from args
model.params.cross_mi_loss_scale = args.cross_mi_loss_scale
model.params.gen_aug_loss_scale = args.gen_aug_loss_scale
model.params.gen_aug_sampling_scheme = args.gen_aug_sampling_scheme
model.params.gen_aug_loss_type = args.gen_aug_loss_type

# WandB
wandb.login()
wandb.init(
    project=args.experiment,
    config=args,
    name=args.runId,
)

# Data loaders
train_dataset = TCGA2CompleteViews(args.datadir + f'/complete_views_split{args.split}_tr.npz')
valid_dataset = TCGA2CompleteViews(args.datadir + f'/complete_views_split{args.split}_val.npz')
test_dataset = TCGA2CompleteViews(args.datadir + f'/complete_views_split{args.split}_te.npz')

num_workers = args.num_workers if hasattr(args, 'num_workers') else 16

kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == 'cuda' else {}

# Deterministic behavior for DataLoader
g = torch.Generator()
g.manual_seed(0)
kwargs['generator'] = g
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

# Checkpointing utilities
os.makedirs(args.checkpoint_dir, exist_ok=True)
ckpt_name = f"idmvae_split_{args.split}_am_{args.gen_aug_loss_scale}_crossmi_{args.cross_mi_loss_scale}_diff_{args.diffusion_loss_weight}.pt"
ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)


def train(epoch):
    """
    Training function
    """
    model.train()
    b_loss = 0
    # Iterate over the data
    for i, batch in enumerate(train_loader):
        data, _ = unpack_data(batch)
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

        optimizer.step()
        b_loss += loss.item() * bs

    epoch_loss = b_loss / len(train_loader.dataset)
    wandb.log({"Loss/train": epoch_loss}, step=epoch)  # Loss/*** could automatically categorize into "Loss" in wandb
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))


def validate(epoch):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        # Iterate over validation data
        for i, batch in enumerate(valid_loader):
            data, _ = unpack_data(batch)
            bs = data[0].size(0)

            # Compute validation loss
            loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(model, data, K=args.K)

            b_loss += loss.item() * bs

    # Epoch validation loss
    epoch_loss = b_loss / len(valid_loader.dataset)
    wandb.log({"Loss/validation": epoch_loss}, step=epoch)
    print('====> Epoch: {:03d} Validation loss: {:.4f}'.format(epoch, epoch_loss))
    val_clfs = train_tcga_latent_classifiers(model, train_loader, device, args)
    test_results = eval_tcga_latent_classifiers(model, test_loader, val_clfs, device, args)
    for key, value in test_results.items():
        wandb.log({f"Validation/{key}": value}, step=epoch)
    print(f'Validation results: {test_results}')


def test(epoch):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        # Iterate over test data
        for i, batch in enumerate(test_loader):
            data, _ = unpack_data(batch)
            bs = data[0].size(0)

            # Compute test loss
            loss, recon_kl_sum_loss, llik_recon_loss, kl_div_loss, cross_mi_loss, gen_aug_loss, diffusion_loss = compute_idmvae_loss(model, data, K=args.K)

            b_loss += loss.item() * bs

    # Epoch test loss
    epoch_loss = b_loss / len(test_loader.dataset)
    wandb.log({"Loss/test": epoch_loss}, step=epoch)
    print('====>             Test loss: {:.4f}'.format(epoch_loss))

    clfs = train_tcga_latent_classifiers(model, train_loader, device, args)
    test_results = eval_tcga_latent_classifiers(model, test_loader, clfs, device, args)
    for key, value in test_results.items():
        wandb.log({f"Test/{key}": value}, step=epoch)
    print(f'Test results: {test_results}')


if __name__ == '__main__':
    # Load checkpoint if it exists
    if os.path.isfile(ckpt_path):
        try:
            payload = torch.load(ckpt_path, map_location=device)
            state = payload.get('model', payload)
            model.load_state_dict(state, strict=False)
            if isinstance(payload, dict) and 'optimizer' in payload:
                try:
                    optimizer.load_state_dict(payload['optimizer'])
                except Exception as opt_e:
                    print(f"Warning: failed to load optimizer state: {opt_e}")
            print(f"Loaded checkpoint from {ckpt_path}. Running test only.")
            test(args.epochs)
            sys.exit(0)
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validate(epoch)
        test_epoch_freq = 5
        if epoch % test_epoch_freq == 0:
            test(epoch)
        # Save at last epoch
        if epoch == args.epochs:
            try:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args)
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"Warning: failed to save checkpoint {ckpt_path}: {e}")
