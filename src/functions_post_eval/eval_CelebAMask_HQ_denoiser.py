# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a DiT denoiser for CelebAMask-HQ.
Supports multiple generation modes in a single run.
"""
import os
import argparse
import json
import numpy as np
import random
from types import SimpleNamespace
from tqdm import tqdm

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.io import write_png
from torch.utils.data import DataLoader

from diffusers.models import AutoencoderKL
from utils import get_mean, load_model_light

from models.dit_diffusion import create_diffusion
from models.dit_denoiser import DiT_models

from dataset_CelebAMask_HQ import CelebAMask_HQ_pregen_4x32x32
import models

LATENT_CHANNELS = 4
LATENT_SIZE = 32
VAE_LATENT_SCALE = 0.18215

# Attributes to select (18 out of 40)
ATTR_VISIBLE = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]

# Note: evaluation is not fully deterministic unless all random seeds/cudnn flags are fixed by caller.

def load_args_namespace(path: str) -> SimpleNamespace:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Args file not found: {path}")
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        return SimpleNamespace(**data)
    args_obj = torch.load(path, map_location="cpu")
    if isinstance(args_obj, dict):
        return SimpleNamespace(**args_obj)
    if isinstance(args_obj, argparse.Namespace):
        return args_obj
    if hasattr(args_obj, "__dict__"):
        return SimpleNamespace(**vars(args_obj))
    raise TypeError(f"Unsupported args format loaded from {path!r}")


def prepare_model_args(train_args: SimpleNamespace,
                       data_dir: str,
                       device: torch.device,
                       sd_vae_variant: str = None) -> SimpleNamespace:
    train_args.datadir = data_dir
    train_args.use_pretrain_feats = True
    train_args.img_channels = LATENT_CHANNELS
    train_args.img_size = LATENT_SIZE
    train_args.no_cuda = (device.type == "cpu")
    if sd_vae_variant:
        train_args.vae = sd_vae_variant
    elif not hasattr(train_args, "vae"):
        train_args.vae = "mse"
    return train_args


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    # Load a custom DiT checkpoint:
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)

    # supports checkpoints from train_CelebAMask_HQ_denoiser.py
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    else:
        checkpoint = checkpoint["model"]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    assert args.ckpt is not None
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load ResNet checkpoint if any multimodal generation is requested.
    if any([args.attr2img_qzpw, args.mask2img_qzpw, args.img2img_qzpw, args.img2img_qwpz, args.img_random, args.attr_mask2img_qzpw]):
        train_args = load_args_namespace(args.resnet_model_args)
        train_args = prepare_model_args(train_args, args.data_path, device, args.vae)
        resnet_model_cls = getattr(models, "CelebA_IDMVAE")
        print(f"Loading IDMVAE checkpoint from {args.resnet_checkpoint}")
        resnet_model = load_model_light(args.resnet_checkpoint, resnet_model_cls, train_args, device)
        resnet_model.eval()
        img_vae = resnet_model.vaes[0]
        mask_vae = resnet_model.vaes[1]
        attr_vae = resnet_model.vaes[2]
    else:
        resnet_model = None
        img_vae = None
        mask_vae = None
        attr_vae = None

    dataset = CelebAMask_HQ_pregen_4x32x32(
        datadir=args.high_res_data_path,
        latent_subdir=args.data_path,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Build job list based on flags; allow multiple modes sequentially.
    jobs = []
    if not any([args.write_orig_image, args.attr2img_qzpw, args.mask2img_qzpw, args.img2img_qzpw, args.img2img_qwpz, args.img_random, args.attr_mask2img_qzpw]):
        jobs.append("denoise_eval")
    else:
        if args.write_orig_image:
            jobs.append("orig")
        if args.attr2img_qzpw:
            jobs.append("attr2img_qzpw")
        if args.mask2img_qzpw:
            jobs.append("mask2img_qzpw")
        if args.img2img_qzpw:
            jobs.append("img2img_qzpw")
        if args.img2img_qwpz:
            jobs.append("img2img_qwpz")
        if args.img_random:
            jobs.append("img_random")
        if args.attr_mask2img_qzpw:
            if args.eval_fusion_random:
                jobs.append("attr_mask2img_qzpw_rand")
            if args.eval_fusion_average:
                jobs.append("attr_mask2img_qzpw_avg")

    # if len(jobs) == 1:
    #     image_dirs = {jobs[0]: os.path.join(args.output_path, "images")}
    # else:
    # Always use job suffix
    image_dirs = {job: os.path.join(args.output_path, f"images_{job}") for job in jobs}
    for path in image_dirs.values():
        os.makedirs(path, exist_ok=True)

    counter = 0
    iter_desc = ",".join(jobs)

    for data in tqdm(loader, desc=iter_desc):
        img, mask, attr, x, noisy_x = data
        img = img.to(device)
        mask = mask.to(device)
        attr = attr.to(device)
        x = x.to(device)
        noisy_x = noisy_x.to(device)
        batch_size = x.shape[0]
        base_idx = counter + 1

        # Cache shared encodings
        img_us = None
        if any(job in ["img2img_qzpw", "img2img_qwpz"] for job in jobs):
             _, _, img_us = img_vae(img, K=1)
        
        mask_us = None

        if "mask2img_qzpw" in jobs or "attr_mask2img_qzpw_rand" in jobs or "attr_mask2img_qzpw_avg" in jobs:
            _, _, mask_us = mask_vae(mask, K=1)
            
        if "attr2img_qzpw" in jobs or "attr_mask2img_qzpw_rand" in jobs or "attr_mask2img_qzpw_avg" in jobs:
            # attr is [B, 40]. attr_vae expects [B, 18].
            attr_subset = attr[:, ATTR_VISIBLE].float()
            _, _, attr_us = attr_vae(attr_subset, K=1)

        for job in jobs:
            samples = None

            if job == "denoise_eval":
                class_labels = model.num_classes * torch.ones(batch_size, dtype=torch.int32, device=device)
                z = torch.randn(len(class_labels), LATENT_CHANNELS, latent_size, latent_size, device=device)
                y = torch.tensor(class_labels, device=device)
                model_kwargs = dict(y=y, noisy_x=noisy_x)
                denoised = diffusion.p_sample_loop(
                    model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )
                samples = vae.decode(denoised / VAE_LATENT_SCALE).sample
                samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)
            elif job == "orig":
                samples = img # This is 128x128.
                # If we want to save SD VAE decoded version of x (256x256), we should decode x.
                # But "orig" usually means the ground truth.
                # In CUB script, `samples = img`.
                # But `img` in CUB might be 256x256?
                # CelebA dataset returns 128x128 images.
                # SD VAE works on 256x256.
                # If we want to compare with generated images (which will be 256x256 from SD VAE),
                # maybe we should upscale img or decode x?
                # Let's stick to `samples = img` (128x128) for now, or decode x if we want 256.
                # Actually, let's decode x to be consistent with "what the model sees as ground truth".
                # `x` is the SD VAE encoded latent of the 256x256 image (resized from 128).
                # So decoding x gives the 256x256 GT.
                # But `img` is the original 128x128.
                # Let's use `img` as it is the true original.
                pass
            elif job == "attr2img_qzpw":
                with torch.no_grad():
                    p_w = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    _, latents_z = torch.split(attr_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    latents_w_new = p_w.rsample(torch.Size([attr_us.size()[0], attr_us.size()[1]])).squeeze(2)
                    latents_img = torch.cat((latents_w_new, latents_z), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img))
                    recon_img = get_mean(px_img).squeeze(0) # This is 128x128 reconstruction from IDMVAE
                    
                    # Now encode with SD VAE to get noisy_x for DiT
                    # We need to resize to 256 first?
                    # Yes, SD VAE expects 256.
                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            elif job == "mask2img_qzpw":
                with torch.no_grad():
                    p_w = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    _, latents_z = torch.split(mask_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    latents_w_new = p_w.rsample(torch.Size([mask_us.size()[0], mask_us.size()[1]])).squeeze(2)
                    latents_img = torch.cat((latents_w_new, latents_z), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img))
                    recon_img = get_mean(px_img).squeeze(0)
                    
                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            elif job == "img2img_qzpw":
                with torch.no_grad():
                    p_w_img = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    _, latents_z_img = torch.split(img_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    latents_w_new = p_w_img.rsample(torch.Size([img_us.size()[0], img_us.size()[1]])).squeeze(2)
                    latents_img_new = torch.cat((latents_w_new, latents_z_img), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img_new))
                    recon_img = get_mean(px_img).squeeze(0)
                    
                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            elif job == "img2img_qwpz":
                with torch.no_grad():
                    p_z = resnet_model.pz_diffusion if args.use_diffusion_prior and hasattr(resnet_model, "pz_diffusion") else resnet_model.get_simple_prior_z()
                    latents_w_img, _ = torch.split(img_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    latents_z_new = p_z.rsample(torch.Size([img_us.size()[0], img_us.size()[1]])).squeeze(2)
                    latents_img_new = torch.cat((latents_w_img, latents_z_new), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img_new))
                    recon_img = get_mean(px_img).squeeze(0)
                    
                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            elif job == "img_random":
                with torch.no_grad():
                    p_z = resnet_model.pz_diffusion if args.use_diffusion_prior and hasattr(resnet_model, "pz_diffusion") else resnet_model.get_simple_prior_z()
                    p_w_img = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    
                    latents_z_new = p_z.rsample(torch.Size([1, batch_size])).squeeze(2).to(device)
                    latents_w_new = p_w_img.rsample(torch.Size([1, batch_size])).squeeze(2).to(device)
                    
                    latents_img_new = torch.cat((latents_w_new, latents_z_new), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img_new))
                    recon_img = get_mean(px_img).squeeze(0)
                    
                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            elif job == "attr_mask2img_qzpw_rand":
                with torch.no_grad():
                    # Randomly select one present modality per batch (consistent with train logic)
                    source_idx = random.choice([1, 2])
                    if source_idx == 1:
                        us_source = mask_us
                    else:
                        us_source = attr_us
                    
                    _, latents_z = torch.split(us_source, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    
                    p_w = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    latents_w_new = p_w.rsample(torch.Size([us_source.size(0), us_source.size(1)])).squeeze(2)

                    latents_img = torch.cat((latents_w_new, latents_z), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img))
                    recon_img = get_mean(px_img).squeeze(0)

                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            elif job == "attr_mask2img_qzpw_avg":
                with torch.no_grad():
                    # Average fusion: average z from all present views (mask and attr)
                    _, z_mask = torch.split(mask_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    _, z_attr = torch.split(attr_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    
                    z_avg = (z_mask + z_attr) / 2.0
                    
                    p_w = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    latents_w_new = p_w.rsample(torch.Size([mask_us.size(0), mask_us.size(1)])).squeeze(2)

                    latents_img = torch.cat((latents_w_new, z_avg), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img))
                    recon_img = get_mean(px_img).squeeze(0)

                    recon_img_256 = torch.nn.functional.interpolate(recon_img, size=(256, 256), mode='bilinear', align_corners=False)
                    recon_img_256 = (recon_img_256 * 2.0) - 1.0 # Convert [0, 1] to [-1, 1] for SD VAE
                    recon_latent = vae.encode(recon_img_256).latent_dist.sample().mul_(VAE_LATENT_SCALE)
                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)

            if samples is not None:
                # Downscale to 128x128 if samples are 256x256 (from SD VAE)
                if samples.shape[-1] == 256:
                    samples = torch.nn.functional.interpolate(samples, size=(128, 128), mode='bilinear', align_corners=False)  # vs mode='bicubic'
                
                samples_uint8 = (samples * 255.0).to("cpu", dtype=torch.uint8)
                for i in range(batch_size):
                    im = samples_uint8[i]
                    img_idx = base_idx + i
                    image_name = os.path.join(image_dirs[job], f"image_{img_idx}.png")
                    write_png(im, image_name)

        counter += batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-res-data-path", type=str, default="/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to use.")
    parser.add_argument("--write-orig-image", action="store_true", default=False)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["mse", "ema"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--attr2img_qzpw", action='store_true', default=False,
                        help="If True, generate attr->image (q_z from attr + p_w image) and save decoded images.")
    parser.add_argument("--mask2img_qzpw", action='store_true', default=False,
                        help="If True, generate mask->image (q_z from mask + p_w image) and save decoded images.")
    parser.add_argument("--img2img_qzpw", action='store_true', default=False,
                        help="If True, generate image->image (q_z from image + p_w image) and save decoded images.")
    parser.add_argument("--img2img_qwpz", action='store_true', default=False,
                        help="If True, generate image->image (q_w from image + p_z shared) and save decoded images.")
    parser.add_argument("--img_random", action='store_true', default=False,
                        help="If True, generate random images (p_w image + p_z shared) and save decoded images.")
    parser.add_argument("--attr_mask2img_qzpw", action='store_true', default=False,
                        help="If True, generate image from attr and mask. Requires --eval_fusion_random or --eval_fusion_average.")
    parser.add_argument("--eval_fusion_random", action='store_true', default=False,
                        help="Enable random fusion for multimodal generation.")
    parser.add_argument("--eval_fusion_average", action='store_true', default=False,
                        help="Enable average fusion for multimodal generation.")
    parser.add_argument("--use_diffusion_prior", action='store_true', default=False,
                        help="If True, use the diffusion prior instead of the simple Gaussian prior for p_w/p_z.")
    parser.add_argument("--resnet_model_args", type=str, default=None,
                        help="Path to the resnet model args namespace (required for qzpw/qwpz generation modes).")
    parser.add_argument("--resnet_checkpoint", type=str, default=None,
                        help="Path to the resnet model checkpoint (required for qzpw/qwpz generation modes).")

    args = parser.parse_args()
    main(args)
