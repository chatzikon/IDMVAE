# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a DiT denoiser.
Supports multiple generation modes in a single run (orig/denoise/text2img/img2text/img2img).
"""
import os
import argparse
import json
import numpy as np
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

from dataset_CUBcluster8 import CUBcluster8_pregen_4x32x32_10x
import models

LATENT_CHANNELS = 4
LATENT_SIZE = 32
VAE_LATENT_SCALE = 0.18215


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

    # supports checkpoints from train_CUB.py
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
    if any([args.text2img_qzpw, args.img2text_qzpw, args.img2img_qzpw, args.img2img_qwpz, args.img_random, args.text_random]):
        train_args = load_args_namespace(args.resnet_model_args)
        train_args = prepare_model_args(train_args, args.data_path, device, args.vae)
        resnet_model_cls = getattr(models, "IDMVAE_CUB_Image_Captions")
        print(f"Loading IDMVAE checkpoint from {args.resnet_checkpoint}")
        resnet_model = load_model_light(args.resnet_checkpoint, resnet_model_cls, train_args, device)
        resnet_model.eval()
        img_vae = resnet_model.vaes[0]
        text_vae = resnet_model.vaes[1]
    else:
        resnet_model = None
        img_vae = None
        text_vae = None

    dataset = CUBcluster8_pregen_4x32x32_10x(
        datadir=args.high_res_data_path,
        latent_subdir=args.data_path,
        split="test",
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
    if not any([args.write_orig_image, args.text2img_qzpw, args.img2text_qzpw, args.img2img_qzpw, args.img2img_qwpz, args.img_random, args.text_random]):
        jobs.append("denoise_eval")
    else:
        if args.write_orig_image:
            jobs.append("orig")
        if args.text2img_qzpw:
            jobs.append("text2img_qzpw")
        if args.img2text_qzpw:
            jobs.append("img2text_qzpw")
        if args.img2img_qzpw:
            jobs.append("img2img_qzpw")
        if args.img2img_qwpz:
            jobs.append("img2img_qwpz")
        if args.img_random:
            jobs.append("img_random")
        if args.text_random:
            jobs.append("text_random")

    if len(jobs) == 1:
        image_dirs = {jobs[0]: os.path.join(args.output_path, "images")}
    else:
        image_dirs = {job: os.path.join(args.output_path, f"images_{job}") for job in jobs}
    for path in image_dirs.values():
        os.makedirs(path, exist_ok=True)

    counter = 0
    refs_by_job = {job: {} for job in jobs}
    gens_by_job = {job: {} for job in jobs}
    special_tokens = [dataset.pad_token, dataset.eos_token]
    iter_desc = ",".join(jobs)

    for data, _ in tqdm(loader, desc=iter_desc):
        img, captions, x, noisy_x = data
        img = img.to(device)
        x = x.to(device)
        noisy_x = noisy_x.to(device)
        batch_size = x.shape[0]
        base_idx = counter + 1

        # Cache shared encodings
        img_us = None
        if any(job in ["img2text_qzpw", "img2img_qzpw", "img2img_qwpz"] for job in jobs):
            _, _, img_us = img_vae(x, K=1)
        text_us = None
        if "text2img_qzpw" in jobs:
            _, _, text_us = text_vae(captions.to(device), K=1)

        for job in jobs:
            samples = None
            recon_txt = None

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
                samples = img
            elif job == "text2img_qzpw":
                with torch.no_grad():
                    p_w = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    _, latents_z = torch.split(text_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)  # (1,B,Z+W)
                    # import pdb; pdb.set_trace()
                    latents_w_new = p_w.rsample(torch.Size([text_us.size()[0], text_us.size()[1]])).squeeze(2)
                    latents_img = torch.cat((latents_w_new, latents_z), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img))
                    recon_latent = get_mean(px_img).squeeze(0)
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)
            elif job == "img2text_qzpw":
                with torch.no_grad():
                    latents_w_img, latents_z_img = torch.split(img_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1) # (1,B,Z+W)
                    p_w_text = resnet_model.pws_diffusion[1] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=1, aux=False)
                    # import pdb; pdb.set_trace()
                    latents_w_new = p_w_text.rsample(torch.Size([img_us.size()[0], img_us.size()[1]])).squeeze(2)
                    latents_txt = torch.cat((latents_w_new, latents_z_img), dim=-1)
                    px_txt = text_vae.px_u(*text_vae.dec(latents_txt))
                    recon_txt = get_mean(px_txt).squeeze(0).cpu()
            elif job == "img2img_qzpw":
                with torch.no_grad():
                    p_w_img = resnet_model.pws_diffusion[0] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=0, aux=False)
                    _, latents_z_img = torch.split(img_us, [resnet_model.params.latent_dim_w, resnet_model.params.latent_dim_z], dim=-1)
                    latents_w_new = p_w_img.rsample(torch.Size([img_us.size()[0], img_us.size()[1]])).squeeze(2)
                    latents_img_new = torch.cat((latents_w_new, latents_z_img), dim=-1)
                    px_img = img_vae.px_u(*img_vae.dec(latents_img_new))
                    recon_latent = get_mean(px_img).squeeze(0)
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
                    recon_latent = get_mean(px_img).squeeze(0)
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
                    recon_latent = get_mean(px_img).squeeze(0)

                    # Debug prints
                    # print(f"DEBUG: batch_size={batch_size}")
                    # print(f"DEBUG: latents_z_new shape: {latents_z_new.shape}")
                    # print(f"DEBUG: latents_img_new shape: {latents_img_new.shape}")
                    # print(f"DEBUG: px_img mean shape: {get_mean(px_img).shape}")
                    # print(f"DEBUG: recon_latent shape: {recon_latent.shape}")

                    
                    z0 = torch.randn_like(recon_latent).to(device)
                    y0 = torch.tensor([model.num_classes] * recon_latent.size(0), dtype=torch.int32, device=device)
                    model_kwargs = dict(y=y0, noisy_x=recon_latent.to(device))
                    denoised_latent = diffusion.p_sample_loop(
                        model.forward, z0.shape, z0, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                    )
                    samples = vae.decode(denoised_latent / VAE_LATENT_SCALE).sample
                    samples = 0.5 * (torch.clamp(samples, min=-1, max=1) + 1)
            elif job == "text_random":
                with torch.no_grad():
                    p_z = resnet_model.pz_diffusion if args.use_diffusion_prior and hasattr(resnet_model, "pz_diffusion") else resnet_model.get_simple_prior_z()
                    p_w_text = resnet_model.pws_diffusion[1] if args.use_diffusion_prior else resnet_model.get_simple_prior_w(view=1, aux=False)
                    
                    latents_z_new = p_z.rsample(torch.Size([1, batch_size])).squeeze(2).to(device)
                    latents_w_new = p_w_text.rsample(torch.Size([1, batch_size])).squeeze(2).to(device)
                    
                    latents_txt_new = torch.cat((latents_w_new, latents_z_new), dim=-1)
                    px_txt = text_vae.px_u(*text_vae.dec(latents_txt_new))
                    recon_txt = get_mean(px_txt).squeeze(0).cpu()

            if samples is not None:
                samples_uint8 = (samples * 255.0).to("cpu", dtype=torch.uint8)
                for i in range(batch_size):
                    im = samples_uint8[i]
                    img_idx = base_idx + i
                    image_name = os.path.join(image_dirs[job], f"image_{img_idx}.png")
                    write_png(im, image_name)

                    cap = captions[i].numpy()
                    words = []
                    for j in range(cap.shape[0]):
                        idx = np.argmax(cap[j, :])
                        tok = dataset.i2w[str(idx)]
                        if tok not in special_tokens:
                            words.append(tok)
                    refs_by_job[job][f"image_{img_idx}"] = " ".join(words)

            if job in ["img2text_qzpw", "text_random"]:
                for i in range(batch_size):
                    gen_cap = recon_txt[i].numpy()
                    words = []
                    for j in range(gen_cap.shape[0]):
                        idx = np.argmax(gen_cap[j, :])
                        tok = dataset.i2w[str(idx)]
                        if tok not in special_tokens:
                            words.append(tok)
                    gens_by_job[job][f"image_{base_idx + i}"] = " ".join(words)

        counter += batch_size

    # Save references/captions
    if len(jobs) == 1:
        job = jobs[0]
        with open(os.path.join(args.output_path, "refs.json"), "w") as f:
            json.dump(refs_by_job[job], f, indent=4)
        if job in ["img2text_qzpw", "text_random"]:
            with open(os.path.join(args.output_path, "gens.json"), "w") as f:
                json.dump(gens_by_job[job], f, indent=4)
    else:
        for job in jobs:
            with open(os.path.join(args.output_path, f"refs_{job}.json"), "w") as f:
                json.dump(refs_by_job[job], f, indent=4)
            if job in ["img2text_qzpw", "text_random"]:
                with open(os.path.join(args.output_path, f"gens_{job}.json"), "w") as f:
                    json.dump(gens_by_job[job], f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-res-data-path", type=str, default="/data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
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
    parser.add_argument("--text2img_qzpw", action='store_true', default=False,
                        help="If True, generate text->image (q_z from text + p_w image) and save decoded images.")
    parser.add_argument("--img2text_qzpw", action='store_true', default=False,
                        help="If True, generate image->text (q_z from image + p_w text) and save captions to gens.json.")
    parser.add_argument("--img2img_qzpw", action='store_true', default=False,
                        help="If True, generate image->image (q_z from image + p_w image) and save decoded images.")
    parser.add_argument("--img2img_qwpz", action='store_true', default=False,
                        help="If True, generate image->image (q_w from image + p_z shared) and save decoded images.")
    parser.add_argument("--img_random", action='store_true', default=False,
                        help="If True, generate random images (p_w image + p_z shared) and save decoded images.")
    parser.add_argument("--text_random", action='store_true', default=False,
                        help="If True, generate random text (p_w text + p_z shared) and save captions.")
    parser.add_argument("--use_diffusion_prior", action='store_true', default=False,
                        help="If True, use the diffusion prior instead of the simple Gaussian prior for p_w/p_z.")
    parser.add_argument("--resnet_model_args", type=str, default=None,
                        help="Path to the resnet model args namespace (required for qzpw/qwpz generation modes).")
    parser.add_argument("--resnet_checkpoint", type=str, default=None,
                        help="Path to the resnet model checkpoint (required for qzpw/qwpz generation modes).")

    args = parser.parse_args()
    main(args)


"""
export https_proxy=http://proxy.divms.uiowa.edu:8888

DATA=/data/backed_up/shared/Data/CUB/CUBcluster8_256/cats22_256px_70_15_15_nonbbox/pregen_4x32x32

python eval_CUB_denoiser.py --vae mse --ckpt results/MMVAEplus_11_15_56_ep50_000-DiT-XL-2/checkpoints/0040000.pt --data-path ${DATA}/MMVAEplus_11-15_56_ep50

python eval_CUB_denoiser.py --vae mse --ckpt results/IDMVAE_Cross40_11_15_55_ep50_003-DiT-XL-2//checkpoints/0040000.pt --data-path ${DATA}/IDMVAE_Cross40_11_15_55_ep50-15_56_ep50

python eval_CUB_denoiser.py --vae mse --ckpt results/IDMVAE_Aug10_Cross40_11_15_53_ep50_002-DiT-XL-2/checkpoints/0040000.pt --data-path ${DATA}/IDMVAE_Aug10_Cross40_11_15_53_ep50


CUDA_VISIBLE_DEVICES=0 python eval_CUB_denoiser.py --vae mse --ckpt results/IDMVAE_Aug10_Cross40_11_15_53_ep50_002-DiT-XL-2/checkpoints/0050000.pt --data-path ${DATA}/IDMVAE_Aug10_Cross40_11_15_53_ep50 --output-path IDMVAE_Aug10_Cross40_11_15_53_ep50 --batch-size 256

CUDA_VISIBLE_DEVICES=0 python eval_CUB_denoiser.py --vae mse --ckpt results/IDMVAE_Aug10_Cross40_11_15_53_ep50_002-DiT-XL-2/checkpoints/0050000.pt --data-path ${DATA}/IDMVAE_Aug10_Cross40_11_15_53_ep50 --output-path test_original --batch-size 256 --write-orig-image

python clipscore.py  /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/refs.json /data/backed_up/shared/Data/CUB/weiran_dit_denoisers/test_original/images/

"""
