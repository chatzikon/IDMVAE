import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import glob
import os
from PIL import Image
from diffusers.models import AutoencoderKL
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

class CelebAHQMaskDS(Dataset):
    def __init__(self, size=128, datapath='./ddata/data/CelebAMask-HQ/', ds_type='train'):
        """
            Args:
                datapath: folder path containing train, val, and test folders of images and mask and celeba attribute text file
                transform: torchvision transform for the images and masks
                ds_type: train, val, or test
        """
        super().__init__()
        self.size = size
        self.ds_type = ds_type
        self.datapath = datapath
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Resize(self.size)])

        self.img_files = glob.glob(os.path.join(self.datapath + self.ds_type + '_img', "*.jpg"))
        self.mask_files = glob.glob(os.path.join(self.datapath + self.ds_type + '_mask', "*.png"))
        self.img_files.sort()  # Lexicographic sort is expected by current pre-generated file naming.
        self.mask_files.sort()
        assert len(self.img_files) == len(self.mask_files)
        
        self.attr_tensor = torch.zeros((len(self.img_files),40), dtype=int)
        self.img_tensor = torch.zeros(len(self.img_files),3,self.size,self.size)
        self.mask_tensor = torch.zeros(len(self.img_files),1,self.size,self.size)
        
        # Read attr text file
        attr_txt_file = open(self.datapath + 'CelebAMask-HQ-attribute-anno.txt')
        attr_list = attr_txt_file.readlines()
        self.attributes = attr_list[1].strip().split(" ")
        assert len(self.attributes) == 40
        
        for i in range(len(self.img_files)):
            assert self.img_files[i].split("/")[-1][:-4] == self.mask_files[i].split("/")[-1][:-4]
            self.img_tensor[i] = self.transform(Image.open(self.img_files[i]))
            self.mask_tensor[i] = self.transform(Image.open(self.mask_files[i]))
            
            img_idx = int(self.img_files[i].split("/")[-1][:-4])
            attr_i = attr_list[img_idx + 2].strip().split(" ")
            assert img_idx == int(attr_i[0][:-4])
            attr_i01 = torch.tensor([1 if a == '1' else 0 for a in attr_i[2:]])
            self.attr_tensor[i] = attr_i01

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Returns a tuple of (image, mask, attribute).
        """
        return (self.img_tensor[index], self.mask_tensor[index], self.attr_tensor[index])



class CelebAMask_HQ_pregen_4x32x32(CelebAHQMaskDS):
    """
    Dataset variant that yields original images/masks/attrs plus pre-generated SD-VAE
    latents:
        - inputs_4x32x32.pt   (encoded latents from SD-VAE)
        - outputs_4x32x32.pt  (reconstructed latents from MMVAE+ decoder)

    Returns:
        datas  = (image_tensor, mask_tensor, attr_tensor, latent_input, latent_output)
    """
    def __init__(self, datadir, split='train', size=128, latent_subdir=None, resize_mode="128to256"):
        # Initialize parent class
        # Note: CelebAHQMaskDS expects datapath to end with '/'
        datapath = datadir if datadir.endswith('/') else datadir + '/'
        super().__init__(size=size, datapath=datapath, ds_type=split)
        
        if latent_subdir:
            latent_dir = latent_subdir if os.path.isabs(latent_subdir) else os.path.join(datadir, latent_subdir)
            # If the files are not directly in latent_dir, try appending split/resize_mode
            if not os.path.exists(os.path.join(latent_dir, 'inputs_4x32x32.pt')):
                candidate = os.path.join(latent_dir, split, resize_mode)
                if os.path.exists(candidate) or os.path.exists(os.path.join(candidate, 'inputs_4x32x32.pt')):
                    latent_dir = candidate
        else:
            latent_dir = os.path.join(datadir, f"pregen_4x32x32/{split}/{resize_mode}")
            
        # Handle potential underscore replacement in path if needed (matching pregen script logic)
        if not os.path.exists(latent_dir) and '-' in latent_dir:
             latent_dir = latent_dir.replace('-', '_')

        print(f"Loading latents from {latent_dir}")
        self.latent_inputs = torch.load(os.path.join(latent_dir, 'inputs_4x32x32.pt'))
        self.latent_outputs = torch.load(os.path.join(latent_dir, 'outputs_4x32x32.pt'))

        num_images = len(self.img_files)
        if self.latent_inputs.shape[0] != num_images or self.latent_outputs.shape[0] != num_images:
             print(f"Warning: Latent count mismatch! Images: {num_images}, Inputs: {self.latent_inputs.shape[0]}, Outputs: {self.latent_outputs.shape[0]}")
             # In case of mismatch, we might need to truncate or handle it. 
             # For now, assuming strict alignment as per pregen script.

    def __getitem__(self, idx):
        # Get original data
        img, mask, attr = super().__getitem__(idx)
        
        # Get latents
        latent_input = self.latent_inputs[idx]
        latent_output = self.latent_outputs[idx]
        
        # Return structure expected by training loop:
        # The training loop expects: _, _, _, x, noisy_x = data
        # So return: (img, mask, attr, latent_input, latent_output)
        # x -> latent_input (clean/target for some tasks, or input)
        # noisy_x -> latent_output (reconstruction/noisy version)
        
        return img, mask, attr, latent_input, latent_output
    


class CelebAHQMaskDS_pt(Dataset):
    def __init__(self, image_size=256, mask_size=128, datapath='./data/CelebAMask_HQ_pt', ds_type='train', use_pretrain_feats=False, args=None):
        """
        Args:
            image_size: target image size
            mask_size: target mask size
            datapath: directory containing images.pt, masks.pt, attributes.pt, splits_idx.pt
            ds_type: 'train', 'val', or 'test'
            use_pretrain_feats: if True, encode images with SD-VAE to latent space
            args: optional namespace containing a 'vae' field to pick ft-ema or ft-mse
        """
        super().__init__()
        self.image_size = image_size
        self.mask_size = mask_size
        self.ds_type = ds_type
        self.use_pretrain_feats = use_pretrain_feats
        self.args = args
        if use_pretrain_feats:
            assert torch.cuda.is_available(), "use_pretrain_feats=True requires CUDA for VAE encoding"
            self.device = torch.device("cuda")
            sd_vae_ft = args.vae if args and hasattr(args, 'vae') else 'mse'
            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{sd_vae_ft}").to(self.device)
            # Freeze the parameters of frozen_module
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.device = torch.device("cpu")
        
        # Load splits
        split_path = os.path.join(datapath, 'splits_idx.pt')
        if not os.path.exists(split_path):
             raise FileNotFoundError(f"Split file not found at {split_path}")
        splits = torch.load(split_path)
        if ds_type not in splits:
             raise ValueError(f"Invalid ds_type: {ds_type}. Must be one of {list(splits.keys())}")
        
        self.indices = splits[ds_type]
        
        print(f"Loading {ds_type} data from {datapath}...")
        
        # Load Images
        img_path = os.path.join(datapath, 'images.pt')
        if os.path.exists(img_path):
            all_imgs = torch.load(img_path, map_location='cpu')
            self.img_tensor = all_imgs[self.indices]
            del all_imgs
        else:
            raise FileNotFoundError(f"Images not found at {img_path}")

        # Load Masks
        mask_path = os.path.join(datapath, 'masks.pt')
        if os.path.exists(mask_path):
            all_masks = torch.load(mask_path, map_location='cpu')
            self.mask_tensor = all_masks[self.indices]
            del all_masks
        else:
             raise FileNotFoundError(f"Masks not found at {mask_path}")

        # Load Attributes
        attr_path = os.path.join(datapath, 'attributes.pt')
        if os.path.exists(attr_path):
            all_attrs = torch.load(attr_path, map_location='cpu')
            self.attr_tensor = all_attrs[self.indices]
            del all_attrs
        else:
             raise FileNotFoundError(f"Attributes not found at {attr_path}")

        # Resize if necessary
        # Images
        if self.img_tensor.shape[-1] != self.image_size:
            print(f"Resizing images from {self.img_tensor.shape[-1]} to {self.image_size}")
            self.img_tensor = torch.nn.functional.interpolate(
                self.img_tensor, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
            )
            
        # Masks
        if self.mask_tensor.shape[-1] != self.mask_size:
            print(f"Resizing masks from {self.mask_tensor.shape[-1]} to {self.mask_size}")
            self.mask_tensor = torch.nn.functional.interpolate(
                self.mask_tensor, size=(self.mask_size, self.mask_size), mode='nearest'
            )
            
        # Convert attributes to 0/1 and Long
        # Original values are -1 and 1. We want 0 and 1.
        self.attr_tensor = ((self.attr_tensor + 1) // 2).long()

    def __len__(self):
        return len(self.img_tensor)

    def __getitem__(self, index):
        img = self.img_tensor[index]
        if self.use_pretrain_feats:
            img_for_vae = img.mul(2).sub(1)  # [0,1] -> [-1,1]
            with torch.no_grad():
                # Configuration for training latent diffusion, output has 32x32 resolution and 4 channels.
                img = self.vae.encode(img_for_vae.unsqueeze(0).to(self.device)).latent_dist.sample().mul_(0.18215).squeeze(0)  # [1,4,32,32] -> [4,32,32]
        return (img, self.mask_tensor[index], self.attr_tensor[index])


LATENT_CHANNELS = 4
LATENT_SIZE = 32
VAE_LATENT_SCALE = 0.18215


def parse_pregen_args():
    parser = argparse.ArgumentParser(description="CelebAMask-HQ 4x32x32 latent dataset generator")
    parser.add_argument("--data-dir", required=True, help="Root directory of CelebAMask-HQ dataset")
    parser.add_argument("--model-args", required=True, help="Path to args.json/rar for the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the IDMVAE checkpoint")
    parser.add_argument("--output-dir", default="pregen_4x32x32", help="Output directory")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all", help="Split to process")
    parser.add_argument("--resize-mode", choices=["128to256", "256_direct"], default="128to256",
                        help="Resizing strategy for inputs generation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sd-vae", default="mse", choices=["mse", "ema"], help="SD VAE variant")
    return parser.parse_args()


def load_args_namespace(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Args file not found: {path}")

    if path.endswith(".json"):
        with open(path, "r") as f:
            return SimpleNamespace(**json.load(f))

    args_obj = torch.load(path, map_location="cpu")
    if isinstance(args_obj, argparse.Namespace):
        return args_obj
    if isinstance(args_obj, dict):
        return SimpleNamespace(**args_obj)
    return args_obj


def prepare_model_args(train_args, data_dir, device):
    train_args.datadir = data_dir
    train_args.use_pretrain_feats = True
    train_args.img_channels = LATENT_CHANNELS
    train_args.img_size = LATENT_SIZE
    train_args.no_cuda = (device == "cpu")
    return train_args


def process_pregen_split(split, args, sd_vae, resn_vae, device, mm_utils):
    print(f"Processing split: {split}")
    ds_size = 256 if args.resize_mode == "256_direct" else 128
    data_dir = args.data_dir if args.data_dir.endswith("/") else args.data_dir + "/"

    dataset = CelebAHQMaskDS(size=ds_size, datapath=data_dir, ds_type=split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    inputs_list = []
    outputs_list = []
    sd_vae.eval()
    resn_vae.eval()

    examples_saved = False
    example_dir = Path(args.output_dir) / split / args.resize_mode / "examples"
    example_dir = Path(str(example_dir).replace("-", "_"))
    example_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {split}"):
            imgs = batch[0].to(device)

            if args.resize_mode == "128to256":
                imgs_for_sd = torch.nn.functional.interpolate(imgs, size=(256, 256), mode="bilinear", align_corners=False)
            else:
                imgs_for_sd = imgs

            imgs_for_sd_norm = imgs_for_sd * 2.0 - 1.0
            latents_in = sd_vae.encode(imgs_for_sd_norm).latent_dist.sample() * VAE_LATENT_SCALE
            inputs_list.append(latents_in.cpu())

            if args.resize_mode == "256_direct":
                imgs_for_resn = torch.nn.functional.interpolate(imgs, size=(128, 128), mode="bilinear", align_corners=False)
            else:
                imgs_for_resn = imgs

            recon_128 = resn_vae.reconstruct(imgs_for_resn)
            if recon_128.dim() == 5:
                recon_128 = recon_128.squeeze(0)

            recon_256 = torch.nn.functional.interpolate(recon_128, size=(256, 256), mode="bilinear", align_corners=False)
            recon_256_norm = recon_256 * 2.0 - 1.0
            latents_out = sd_vae.encode(recon_256_norm).latent_dist.sample() * VAE_LATENT_SCALE
            outputs_list.append(latents_out.cpu())

            if not examples_saved:
                n = min(8, imgs.shape[0])
                orig = imgs_for_resn[:n]
                rec128 = recon_128[:n]
                rec256 = recon_256[:n]
                comparison_128 = torch.cat([orig, rec128], dim=0)
                mm_utils.save_image(comparison_128, example_dir / "comparison_128.png", nrow=n)
                orig_up = torch.nn.functional.interpolate(orig, size=(256, 256), mode="bilinear", align_corners=False)
                comparison_256 = torch.cat([orig_up, rec256], dim=0)
                mm_utils.save_image(comparison_256, example_dir / "comparison_256.png", nrow=n)
                examples_saved = True

    inputs_all = torch.cat(inputs_list, dim=0)
    outputs_all = torch.cat(outputs_list, dim=0)
    out_path = Path(args.output_dir) / split / args.resize_mode
    out_path = Path(str(out_path).replace("-", "_"))
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(inputs_all, out_path / "inputs_4x32x32.pt")
    torch.save(outputs_all, out_path / "outputs_4x32x32.pt")
    print(f"Saved {inputs_all.shape} inputs and {outputs_all.shape} outputs to {out_path}")


def main_pregen():
    args = parse_pregen_args()
    device = torch.device(args.device)

    import utils as mm_utils
    import models

    print(f"Loading SD VAE ({args.sd_vae})...")
    sd_vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.sd_vae}").to(device)

    print(f"Loading IDMVAE checkpoint from {args.checkpoint}...")
    train_args = load_args_namespace(args.model_args)
    train_args = prepare_model_args(train_args, args.data_dir, device)
    model_cls = models.CelebA_IDMVAE
    model = mm_utils.load_model_light(args.checkpoint, model_cls, train_args, device)
    model.eval()
    resn_vae = model.vaes[0]

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_pregen_split(split, args, sd_vae, resn_vae, device, mm_utils)


if __name__ == "__main__":
    main_pregen()

