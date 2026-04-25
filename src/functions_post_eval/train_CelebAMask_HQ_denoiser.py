# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import argparse
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from datetime import datetime
import sys
from tqdm import tqdm

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers.models import AutoencoderKL

from models.dit_diffusion import create_diffusion
from models.dit_denoiser import DiT_models

from dataset_CelebAMask_HQ import CelebAMask_HQ_pregen_4x32x32


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Resolve results directory (default: alongside data_path under a '_denoiser' subdir)
    data_base_dir = os.path.dirname(args.data_path.rstrip("/"))
    results_root = args.results_dir or os.path.join(data_base_dir, "_denoiser")  # , str(args.batch_size)

    # Setup an experiment folder:
    os.makedirs(results_root, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{results_root}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)

    data_name = args.data_path.rstrip("/")
    data_name = data_name.split("/")[-1]

    experiment_dir = f"{results_root}/{data_name}_{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    log_dir = f"{experiment_dir}/logs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Mirror stdout/stderr to a log file while keeping terminal output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.out")
    log_fh = open(log_path, "w", buffering=1)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
            for s in self.streams:
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)
    print(f"[INFO] Logging to {log_path}")

    # Create model:
    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
    )

    if args.pretrain_ckpt:
        ckpt_path = args.pretrain_ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Reduced learning rate for finetuning.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)

    # Full train set
    dataset = CelebAMask_HQ_pregen_4x32x32(
        datadir=args.high_res_data_path,
        latent_subdir=args.data_path,
        split='train',
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        print(f"Beginning epoch {epoch}...")
        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for data in progress:
            _, _, _, x, noisy_x = data
            x = x.to(device)
            noisy_x = noisy_x.to(device)
            # import pdb;pdb.set_trace()

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            y = model.num_classes * torch.ones(x.shape[0], dtype=torch.int32, device=x.device)
            model_kwargs = dict(y=y, noisy_x=noisy_x)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            progress.set_postfix(loss=loss.item())
            if train_steps % args.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                tqdm.write(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout

    print("Done!")
    # Restore std streams and close log file
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_fh.close()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-res-data-path", type=str, default="/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    # image-size is the one after vae encoding.
    parser.add_argument("--image-size", type=int, choices=[32], default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--pretrain-ckpt", type=str, default="")
    args = parser.parse_args()
    main(args)


"""
DiT pretrain checkpoint: ./DiT-XL-2-256x256.pt 
 -- (https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) @ https://github.com/facebookresearch/DiT

DATA=/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM/pregen_4x32x32
ckpt_dir=/data/backed_up/shared/Data/CUB/DiT-XL-2-256x256.pt

CUDA_VISIBLE_DEVICES=3 python train_CelebAMask_HQ_denoiser.py --model DiT-XL/2 --epochs 20 --batch-size 32 --ckpt-every 2500 --log-every 50 --pretrain-ckpt ${ckpt_dir} --data-path ${DATA}/IDMVAE_11_30_5_ep100
CUDA_VISIBLE_DEVICES=3 python train_CelebAMask_HQ_denoiser.py --model DiT-XL/2 --epochs 20 --batch-size 32 --ckpt-every 5000 --log-every 50 --pretrain-ckpt ${ckpt_dir} --data-path ${DATA}/IDMVAE_11_30_5_ep100
CUDA_VISIBLE_DEVICES=3 python train_CelebAMask_HQ_denoiser.py --model DiT-XL/2 --epochs 50 --batch-size 32 --ckpt-every 5000 --log-every 50 --pretrain-ckpt ${ckpt_dir} --data-path ${DATA}/IDMVAE_11_30_5_ep100

CUDA_VISIBLE_DEVICES=2 python train_CelebAMask_HQ_denoiser.py --model DiT-XL/2 --epochs 20 --batch-size 32 --ckpt-every 2500 --log-every 50 --pretrain-ckpt ${ckpt_dir} --data-path ${DATA}/MMVAEplus_11_29_0_ep100
CUDA_VISIBLE_DEVICES=2 python train_CelebAMask_HQ_denoiser.py --model DiT-XL/2 --epochs 20 --batch-size 32 --ckpt-every 5000 --log-every 50 --pretrain-ckpt ${ckpt_dir} --data-path ${DATA}/MMVAEplus_11_29_0_ep100
CUDA_VISIBLE_DEVICES=2 python train_CelebAMask_HQ_denoiser.py --model DiT-XL/2 --epochs 50 --batch-size 32 --ckpt-every 5000 --log-every 50 --pretrain-ckpt ${ckpt_dir} --data-path ${DATA}/MMVAEplus_11_29_0_ep100
"""

