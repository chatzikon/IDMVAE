import os
import json
import argparse
from functools import lru_cache
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple
from diffusers.models import AutoencoderKL # pip install diffusers transformers accelerate


def cub_caption_vocab_path(datadir, vocab_file=None):
    """
    Path to the CUB captions JSON vocab (keys w2i, i2w).

    Args:
        datadir: Directory containing cub.vocab (with images.pt, captions.pt, etc.).
        vocab_file: If set, basename or absolute path. Relative paths join datadir.
    """
    if vocab_file:
        return vocab_file if os.path.isabs(vocab_file) else os.path.join(datadir, vocab_file)
    return os.path.join(datadir, "cub.vocab")


def build_cub_caption_vocab_file(captions, vocab_path: str, *, min_occ: int = 3) -> None:
    """
    Build and write cub.vocab from captions.pt content (list of per-image caption lists).

    Same tokenization as CUBSentences: NLTK word_tokenize, min token frequency ``min_occ``,
    special tokens <exc>, <pad>, <eos> first.
    """
    token_counts = {}
    token_order = []
    for caps in captions:
        for raw in caps:
            for tok in word_tokenize(raw.lower()):
                if tok not in token_counts:
                    token_counts[tok] = 0
                    token_order.append(tok)
                token_counts[tok] += 1

    w2i = {}
    i2w = {}
    special_tokens = ["<exc>", "<pad>", "<eos>"]
    for st in special_tokens:
        idx = len(w2i)
        w2i[st] = idx
        i2w[idx] = st

    for tok in token_order:
        if token_counts[tok] > min_occ and tok not in w2i:
            idx = len(w2i)
            w2i[tok] = idx
            i2w[idx] = tok

    vocab_dir = os.path.dirname(os.path.abspath(vocab_path))
    if vocab_dir:
        os.makedirs(vocab_dir, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump({"w2i": w2i, "i2w": i2w}, f)


def ensure_cub_caption_vocab(datadir, vocab_file=None, min_occ=3, captions=None):
    """
    Ensure cub.vocab exists. If missing, build from ``captions`` or from ``captions.pt`` in datadir.

    Raises FileNotFoundError if there is no vocab and no way to load captions.
    """
    path = cub_caption_vocab_path(datadir, vocab_file)
    if os.path.isfile(path):
        return path
    if captions is None:
        captions_path = os.path.join(datadir, "captions.pt")
        if not os.path.isfile(captions_path):
            raise FileNotFoundError(
                f"No caption vocab at {path} and no captions.pt at {captions_path} to build one. "
                "Add cub.vocab or place captions.pt in datadir."
            )
        captions = torch.load(captions_path)
    build_cub_caption_vocab_file(captions, path, min_occ=min_occ)
    return path


def load_cub_caption_vocab(datadir, vocab_file=None, min_occ=3, captions=None):
    """Load full vocab JSON; creates cub.vocab when missing (see ``ensure_cub_caption_vocab``)."""
    ensure_cub_caption_vocab(datadir, vocab_file, min_occ=min_occ, captions=captions)
    path = cub_caption_vocab_path(datadir, vocab_file)
    with open(path, "r") as vf:
        return json.load(vf)


@lru_cache(maxsize=32)
def load_cub_i2w(datadir, vocab_file=None, min_occ=3):
    """Index-to-word mapping for decoding one-hot caption rows (cached by datadir / options)."""
    return load_cub_caption_vocab(datadir, vocab_file, min_occ=min_occ)["i2w"]


class CUBcluster8Dataset(Dataset):
    """
    PyTorch Dataset for CUBcluster8 producing (image, onehot_caption) pairs.

    Data files in datadir:
      - images.pt             # tensor [N,3,256,256]
      - captions.pt           # list of N lists, each 10 raw caption strings
      - labels_cluster.pt     # LongTensor [N] in 0..8
      - labels_category.pt    # LongTensor [N] in 1..200
      - train_idx.npy         # indices for train (all clusters + Other)
      - train_cluster_idx.npy # indices for train clusters only
      - val_cluster_idx.npy   # indices for val clusters only
      - test_cluster_idx.npy  # indices for test clusters only
      - cub.vocab             # JSON with {'w2i':..., 'i2w':...}

    Each sample is one image tensor and one one-hot caption tensor [max_len, vocab_size].

    Args:
        datadir (str): directory containing the above files
        split (str): 'train', 'val', or 'test'
        cluster_only (bool): when split='train', use train_cluster_idx.npy if True
        transform (callable, optional): applied to image tensors
    """
    def __init__(self, datadir, split='train', cluster_only=False, transform=None, use_pretrain_feats=True,
                 args=None, shared_data=None, vae=None, device=None):
        self.datadir = datadir
        self.args = args
        self.use_pretrain_feats = use_pretrain_feats
        self.thres_deg_int = int(args.degree_away_center_threshold) if args else 0




        if shared_data is not None:
            self.images = shared_data["images"]
            self.captions = shared_data["captions"]
            self.labels_cluster = shared_data["labels_cluster"]
            self.labels_category = shared_data["labels_category"]
            self.labels_direction=shared_data["labels_direction"]
            self.image_ids= shared_data["image_ids"]
        else:
            self.images = torch.load(os.path.join(datadir, 'images.pt'), map_location="cpu")
            self.captions = torch.load(os.path.join(datadir, 'captions.pt'), map_location="cpu")
            self.labels_cluster = torch.load(os.path.join(datadir, 'labels_cluster.pt'), map_location="cpu")
            self.labels_category = torch.load(os.path.join(datadir, 'labels_category.pt'), map_location="cpu")
            self.labels_direction= torch.load(
                os.path.join(self.datadir, 'labels_direction_deg_{}.pt'.format(self.thres_deg_int)), weights_only=False,
                map_location="cpu") \
                if os.path.exists(os.path.join(self.datadir, 'labels_direction_deg_{}.pt'.format(self.thres_deg_int))) else None,
            self.image_ids= torch.load(os.path.join(self.datadir, 'image_ids.pt'), weights_only=False, map_location="cpu") \
                if os.path.exists(os.path.join(self.datadir, 'image_ids.pt')) else None



        # load vocab (same JSON as CUBSentences / eval text rendering)
        vocab = load_cub_caption_vocab(datadir)
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']  # Make index-to-word mapping available, e.g., for WandB
        self.vocab_size = len(self.w2i)
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_idx = self.w2i.get('<unk>', self.w2i.get('<exc>', 0))
        self.pad_idx = self.w2i.get(self.pad_token, 0)
        self.max_sent_len = 32
        # choose index file
        split = split.lower()
        if split == 'train':
            idx_file = 'train_cluster_idx.npy' if cluster_only else 'train_idx.npy'
        elif split == 'val':
            idx_file = 'val_cluster_idx.npy'
        elif split == 'test':
            idx_file = 'test_cluster_idx.npy'
        else:
            raise ValueError(f"Unknown split '{split}'")
        raw_indices = np.load(os.path.join(datadir, idx_file))
        # flatten into (img_idx, cap_idx)
        self.pairs = []
        for img_idx in raw_indices:
            img_idx = int(img_idx)
            for cap_idx in range(len(self.captions[img_idx])):
                # img_idx is the index of the image, range in total images dataset size
                # cap_idx is the index of the caption for that image, range in 0..9
                self.pairs.append((img_idx, cap_idx)) # one image have exactly 10 captions in a pair
        self.transform = transform

        # === Pretrained VAE from DiT ===
        """https://github.com/facebookresearch/DiT/blob/main/sample.py#L44
        original images (3*256*256) -> pretrained_vae.encode -> pretrained_latent (4*32*32)
        -> [our model vae.enc->dec] -> posttrained_latent (4*32*32) -> pretrained_vae.decode 
        -> reconstructed/generated images (3*256*256)
        """
        self.use_pretrain_feats = use_pretrain_feats
        if use_pretrain_feats:
            self.device = device or torch.device("cuda")
            self.vae = vae
            assert self.vae is not None
        else:
            self.device = torch.device("cpu")
            self.vae = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        range of idx = range of img_idx * range of cap_idx (10)
        img_idx is the index of the image in the dataset,
        cap_idx is the index of the caption for that image (0 to 9)        
        """
        img_idx, cap_idx = self.pairs[idx]
        # image
        img = self.images[img_idx]
        if self.transform:
            img = self.transform(img)
        if self.use_pretrain_feats:
            img_for_vae = img.mul(2).sub(1)  # map [0,1] -> [-1,1] (https://github.com/facebookresearch/DiT/blob/main/train.py#L162)
            with torch.no_grad():
                # import pdb; pdb.set_trace()
                # Configuration for training latent diffusion, output has 32x32 resolution and 4 channels.
                img = self.vae.encode(img_for_vae.unsqueeze(0).to(self.device)).latent_dist.sample().mul_(0.18215).squeeze(0) # [1,4,32,32] -> [4,32,32]
                # import pdb; pdb.set_trace()
        # raw caption string
        raw = self.captions[img_idx][cap_idx]
        # tokenize + truncate or pad
        toks = word_tokenize(raw.lower())
        # truncate and add eos
        if len(toks) >= self.max_sent_len:
            toks = toks[:self.max_sent_len-1] + [self.eos_token]
        else:
            toks = toks + [self.eos_token]
        # pad
        if len(toks) < self.max_sent_len:
            toks = toks + [self.pad_token] * (self.max_sent_len - len(toks))
        # indices
        idxs = [self.w2i.get(t, self.unk_idx) for t in toks]
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        # one-hot
        cap_tensor = F.one_hot(idx_tensor, num_classes=self.vocab_size).float()
        # labels
        lbl_cluster = int(self.labels_cluster[img_idx])
        lbl_dir = int(self.labels_direction[img_idx]) if self.labels_direction is not None else None
        lbl_cat = int(self.labels_category[img_idx])
        img_id = self.image_ids[img_idx] if self.image_ids is not None else None
        dataset_index = img_idx  # This is the index of the image in the original dataset
        subset_index = idx  # This is the index in the subset of pairs (img_idx, cap_idx)

        datas = (img, cap_tensor)
        labels = (lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index)
        return datas, labels


# --- Helper Wrapper Datasets ---
class CUBImageViewDataset(Dataset):
    """
    Wrapper dataset to provide only the image view from CUBcluster8Dataset.
    Use case: t-SNE/UMAP visualization of image latents.
    """
    def __init__(self, original_cub_dataset):
        self.original_cub_dataset = original_cub_dataset
        # Expose attributes needed by visualize_latents_with_priors or its internal functions
        # if they rely on them (e.g., for label names or number of classes)
        # For CUBcluster8, the label is lbl_cluster.
        # If visualize_latents_with_priors needs specific label info, ensure it's accessible.

    def __len__(self):
        return len(self.original_cub_dataset)

    def __getitem__(self, idx):
        (img, _), (lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index) = self.original_cub_dataset[idx]
        return img, (lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index)

class CUBCaptionViewDataset(Dataset):
    """
    Wrapper dataset to provide only the caption view from CUBcluster8Dataset.
    """
    def __init__(self, original_cub_dataset):
        self.original_cub_dataset = original_cub_dataset
        # Similar to CUBImageViewDataset, expose necessary attributes if needed.

    def __len__(self):
        return len(self.original_cub_dataset)

    def __getitem__(self, idx):
        (_, cap_tensor), (lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index) = self.original_cub_dataset[idx]
        return cap_tensor, (lbl_cluster, lbl_dir, lbl_cat, img_id, dataset_index)


class CUBSentences(Dataset):
    """
    PT-backed sentence dataset for CUB captions.
    Keeps the old constructor interface but reads from captions.pt/cub.vocab.
    """

    def __init__(self, root_data_dir, split, one_hot=False, transpose=False, transform=None, **kwargs):
        super().__init__()
        self.split = split
        self.max_sequence_length = kwargs.get("max_sequence_length", 32)
        self.min_occ = kwargs.get("min_occ", 3)
        self.transform = transform
        self.one_hot = one_hot
        self.transpose = transpose

        root_path = Path(root_data_dir)
        cub_path = root_path / "cub"
        if (root_path / "captions.pt").is_file():
            self.data_dir = str(root_path)
        elif (cub_path / "captions.pt").is_file():
            self.data_dir = str(cub_path)
        else:
            raise FileNotFoundError(
                f"Could not find captions.pt under '{root_path}' or '{cub_path}'"
            )

        captions_path = os.path.join(self.data_dir, "captions.pt")
        self.captions = torch.load(captions_path)
        vocab = load_cub_caption_vocab(
            self.data_dir, min_occ=self.min_occ, captions=self.captions
        )
        self.w2i = vocab["w2i"]
        self.i2w = vocab["i2w"]

        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._unk_idx = self.w2i.get("<unk>", self.w2i.get("<exc>", 0))

        split_lower = split.lower()
        if split_lower == "train":
            candidate_files = ["train_idx.npy", "train_cluster_idx.npy"]
        elif split_lower == "test":
            candidate_files = ["test_cluster_idx.npy", "val_cluster_idx.npy", "test_idx.npy"]
        else:
            raise ValueError("Only train or test split is available")

        idx_path = None
        for name in candidate_files:
            p = os.path.join(self.data_dir, name)
            if os.path.isfile(p):
                idx_path = p
                break
        if idx_path is None:
            raise FileNotFoundError(
                f"Could not find split index file in {self.data_dir}: {candidate_files}"
            )
        image_indices = np.load(idx_path)

        self.data = []
        for img_idx in image_indices:
            img_idx = int(img_idx)
            for raw_caption in self.captions[img_idx]:
                toks = word_tokenize(raw_caption.lower())
                toks = toks[: self.max_sequence_length - 1] + [self.eos_token]
                length = len(toks)
                if length < self.max_sequence_length:
                    toks = toks + [self.pad_token] * (self.max_sequence_length - length)
                idxs = [self.w2i.get(t, self._unk_idx) for t in toks]
                self.data.append({"idx": idxs, "length": length})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = torch.tensor(self.data[idx]["idx"], dtype=torch.long)
        if self.one_hot:
            sent = torch.nn.functional.one_hot(sent, self.vocab_size).float()
        if self.transpose:
            sent = sent.transpose(-2, -1)
        if self.transform is not None:
            sent = self.transform(sent)
        return sent, self.data[idx]["length"]

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i.get("<pad>", 0)

    @property
    def eos_idx(self):
        return self.w2i.get("<eos>", 0)

    @property
    def unk_idx(self):
        return self._unk_idx

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


class CUBcluster8_pregen_4x32x32_10x(CUBcluster8Dataset):
    """
    Dataset variant for 10x latents aligned with captions (one latent per caption per image):
        - inputs_4x32x32_10x.pt   (shape [N, num_caps, 4, 32, 32])
        - outputs_4x32x32_10x.pt  (shape [N, num_caps, 4, 32, 32])

    The second dimension matches the number of captions per image (typically 10).
    """
    def __init__(self, datadir, split='train', cluster_only=False,
                 transform=None, latent_subdir=None, use_pretrain_feats=False, args=None,
                 inputs_name="inputs_4x32x32_10x.pt", outputs_name="outputs_4x32x32_10x.pt"):
        super().__init__(
            datadir=datadir,
            split=split,
            cluster_only=cluster_only,
            transform=transform,
            use_pretrain_feats=use_pretrain_feats,
            args=args,
        )
        if latent_subdir:
            latent_dir = latent_subdir if os.path.isabs(latent_subdir) else os.path.join(datadir, latent_subdir)
        else:
            latent_dir = datadir
        self.latent_inputs = torch.load(os.path.join(latent_dir, inputs_name))
        self.latent_outputs = torch.load(os.path.join(latent_dir, outputs_name))

        num_images = self.images.shape[0]
        if self.latent_inputs.shape[0] != num_images or self.latent_outputs.shape[0] != num_images:
            raise ValueError(
                f"Latent tensors do not match dataset size: "
                f"N_images={num_images}, "
                f"N_inputs={self.latent_inputs.shape[0]}, "
                f"N_outputs={self.latent_outputs.shape[0]}"
            )

        caps_per_image = len(self.captions[0]) if isinstance(self.captions, list) and len(self.captions) > 0 else None
        if caps_per_image and (self.latent_inputs.shape[1] != caps_per_image or self.latent_outputs.shape[1] != caps_per_image):
            raise ValueError(
                f"Latent tensors do not match captions per image: "
                f"caps_per_image={caps_per_image}, "
                f"N_inputs_caps={self.latent_inputs.shape[1]}, "
                f"N_outputs_caps={self.latent_outputs.shape[1]}"
            )

    def __getitem__(self, idx):
        (img, cap_tensor), labels = super().__getitem__(idx)
        img_idx, cap_idx = self.pairs[idx]
        latent_input = self.latent_inputs[img_idx, cap_idx]
        latent_output = self.latent_outputs[img_idx, cap_idx]
        datas = (img, cap_tensor, latent_input, latent_output)
        return datas, labels

LATENT_CHANNELS = 4
LATENT_SIZE = 32
VAE_LATENT_SCALE = 0.18215


class TensorDataset1D(Dataset):
    """Simple Dataset wrapper over a tensor to avoid unnecessary copies."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


def parse_cub_pregen_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUB 256px -> 4x32x32 latent dataset generator")
    parser.add_argument("--data-dir", required=True, help="Directory containing images.pt/captions.pt/labels_*.pt")
    parser.add_argument("--model-args", required=True, help="Path to the args.json/args.rar used for training the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by save_model_light() (e.g. model_50.rar)")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs (default: same as --data-dir)")
    parser.add_argument("--inputs-name", default="inputs_4x32x32.pt", help="Filename for encoded SD-VAE latents")
    parser.add_argument("--outputs-name", default="outputs_4x32x32.pt", help="Filename for IDMVAE reconstructions")
    parser.add_argument("--generate-1x", action="store_true", help="Also generate single-sample latents (opt-in; 10x is default)")
    parser.add_argument("--skip-10x", action="store_true", help="Skip the default 10x latent generation")
    parser.add_argument("--inputs-name-10x", default="inputs_4x32x32_10x.pt", help="Filename for 10x SD-VAE latents")
    parser.add_argument("--outputs-name-10x", default="outputs_4x32x32_10x.pt", help="Filename for 10x IDMVAE reconstructions")
    parser.add_argument("--samples-per-image", type=int, default=10, help="How many latent samples to draw per image")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding/decoding")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of DataLoader workers")
    parser.add_argument("--device", default=None, help="PyTorch device (e.g. cuda, cuda:1, cpu). Defaults to CUDA when available.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for debugging")
    parser.add_argument("--sd-vae", choices=["ema", "mse"], default=None, help="Override Stable Diffusion VAE variant if desired")
    parser.add_argument("--text2img-output", default=None,
                        help="Optional path to save text->image prior_shared generations (4x32x32).")
    parser.add_argument("--text2img-caption-index", type=int, default=0, help="Which caption index to use per image (0-9).")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    if images.dtype != torch.float32:
        images = images.float()
    if images.max() > 1.0:
        images = images / 255.0
    return images.clamp(0.0, 1.0).contiguous()


def encode_images_to_latents(dataloader: DataLoader,
                             sd_vae,
                             resn_vae,
                             device: torch.device,
                             total_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.empty((total_samples, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float32)
    outputs = torch.empty_like(inputs)

    sd_vae.eval()
    resn_vae.eval()
    for p in sd_vae.parameters():
        p.requires_grad = False
    for p in resn_vae.parameters():
        p.requires_grad = False

    offset = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Encoding images + reconstructing latents"):
            imgs = batch.to(device)
            imgs = imgs.mul(2).sub(1) # Normalize images [0, 1] to [-1, 1]
            latents = sd_vae.encode(imgs).latent_dist.sample() * VAE_LATENT_SCALE

            recon = resn_vae.reconstruct(latents)
            if recon.dim() == 5 and recon.size(0) == 1:
                recon = recon.squeeze(0)

            batch_size = latents.size(0)
            inputs[offset:offset + batch_size] = latents.cpu()
            outputs[offset:offset + batch_size] = recon.cpu()
            offset += batch_size

    if offset != total_samples:
        raise RuntimeError(f"Processed {offset} samples but expected {total_samples}")

    return inputs, outputs


def encode_images_to_latents_10x(dataloader: DataLoader,
                                 sd_vae,
                                 resn_vae,
                                 device: torch.device,
                                 total_samples: int,
                                 samples_per_image: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs: list = [None] * total_samples
    outputs: list = [None] * total_samples

    sd_vae.eval()
    resn_vae.eval()
    for p in sd_vae.parameters():
        p.requires_grad = False
    for p in resn_vae.parameters():
        p.requires_grad = False

    offset = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Encoding images + reconstructing latents (10x)"):
            imgs = batch.to(device).mul(2).sub(1)
            batch_size = imgs.size(0)
            for local_idx in range(batch_size):
                img_idx = offset + local_idx
                latent_dist = sd_vae.encode(imgs[local_idx : local_idx + 1]).latent_dist

                latents_per_img = []
                recons_per_img = []
                for _ in range(samples_per_image):
                    latents = latent_dist.sample() * VAE_LATENT_SCALE
                    recon = resn_vae.reconstruct(latents)
                    if recon.dim() == 5 and recon.size(0) == 1:
                        recon = recon.squeeze(0)
                    latents_per_img.append(latents.squeeze(0).cpu())
                    recons_per_img.append(recon.squeeze(0).cpu())

                inputs[img_idx] = torch.stack(latents_per_img, dim=0)
                outputs[img_idx] = torch.stack(recons_per_img, dim=0)
            offset += batch_size

    if any(x is None for x in inputs) or any(x is None for x in outputs):
        raise RuntimeError(f"Processed {offset} samples but expected {total_samples}")

    return torch.stack(inputs, dim=0), torch.stack(outputs, dim=0)


def build_caption_tensor_for_all(captions, w2i, max_len=32, cap_idx=0):
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_idx = w2i.get("<unk>", w2i.get("<exc>", 0))
    cap_tensors = []
    for caps in captions:
        raw = caps[cap_idx] if cap_idx < len(caps) else caps[0]
        toks = word_tokenize(raw.lower())
        if len(toks) >= max_len:
            toks = toks[: max_len - 1] + [eos_token]
        else:
            toks = toks + [eos_token]
        if len(toks) < max_len:
            toks = toks + [pad_token] * (max_len - len(toks))
        idxs = [w2i.get(t, unk_idx) for t in toks]
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        cap_tensor = torch.nn.functional.one_hot(idx_tensor, num_classes=len(w2i)).float()
        cap_tensors.append(cap_tensor)
    return torch.stack(cap_tensors, dim=0)


def generate_text2img_prior_shared(model, cap_tensor_full, device, batch_size, mm_utils) -> torch.Tensor:
    dataset = TensorDataset1D(cap_tensor_full)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = cap_tensor_full.shape[0]
    outputs = torch.empty((total, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float32)

    text_vae = model.vaes[1]
    img_vae = model.vaes[0]
    p_w = model.get_simple_prior_w(view=0, aux=False)
    idx = 0
    with torch.no_grad():
        for cap_tensor in loader:
            cap_tensor = cap_tensor.to(device)
            _, _, us = text_vae(cap_tensor, K=1)
            _, latents_z = torch.split(us, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
            latents_w_new = p_w.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
            latents_img = torch.cat((latents_w_new, latents_z), dim=-1)
            px_img = img_vae.px_u(*img_vae.dec(latents_img))
            recon = mm_utils.get_mean(px_img).squeeze(0)
            outputs[idx : idx + recon.size(0)] = recon.cpu()
            idx += recon.size(0)
    return outputs


def load_label_tensors(data_dir: str, expected_len: int) -> Dict[str, torch.Tensor]:
    labels: Dict[str, torch.Tensor] = {}
    for entry in sorted(os.listdir(data_dir)):
        if not (entry.startswith("labels_") and entry.endswith(".pt")):
            continue
        path = os.path.join(data_dir, entry)
        tensor = torch.load(path, map_location="cpu")
        if torch.is_tensor(tensor) and tensor.shape[0] == expected_len:
            labels[entry] = tensor
        else:
            print(f"[WARN] Skipping {entry}: shape {getattr(tensor, 'shape', None)} does not match {expected_len}")
    return labels


def summarize_data(images: torch.Tensor, captions, labels: Dict[str, torch.Tensor]) -> None:
    num_images = images.shape[0]
    print(f"Loaded {num_images} images: dtype={images.dtype}, shape={tuple(images.shape)}, min={images.min().item():.4f}, max={images.max().item():.4f}")
    if isinstance(captions, list):
        print(f"Loaded captions.pt with {len(captions)} entries (expected {num_images})")
    else:
        print(f"Loaded captions.pt of type {type(captions)}")
    for name, tensor in labels.items():
        print(f"Label {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def main_pregen_cub() -> None:
    args = parse_cub_pregen_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir or args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import models
    import utils as mm_utils
    try:
        mm_utils.utils = mm_utils
    except Exception:
        pass

    images = torch.load(os.path.join(args.data_dir, "images.pt"), map_location="cpu")
    captions = torch.load(os.path.join(args.data_dir, "captions.pt"))
    images_tensor_full = images if torch.is_tensor(images) else torch.stack(images)
    images_tensor_full = normalize_images(images_tensor_full)

    full_num_samples = images_tensor_full.shape[0]
    label_tensors = load_label_tensors(args.data_dir, expected_len=full_num_samples)
    summarize_data(images_tensor_full, captions, label_tensors)
    if isinstance(captions, list) and len(captions) != full_num_samples:
        raise ValueError(f"Captions count {len(captions)} does not match number of images {full_num_samples}")

    if args.max_samples is not None:
        limit = min(args.max_samples, full_num_samples)
        images_tensor = images_tensor_full[:limit]
        print(f"Processing subset of images: {limit}/{full_num_samples}")
    else:
        images_tensor = images_tensor_full

    num_samples = images_tensor.shape[0]
    samples_per_image = args.samples_per_image

    train_args = load_args_namespace(args.model_args)
    train_args = prepare_model_args(train_args, args.data_dir, device, args.sd_vae)
    model_cls = getattr(models, "IDMVAE_CUB_Image_Captions")
    print(f"Loading IDMVAE checkpoint from {args.checkpoint}")
    model = mm_utils.load_model_light(args.checkpoint, model_cls, train_args, device)
    model.eval()

    sd_vae = model.pretrained_vae
    resn_vae = model.vaes[0]
    dataset = TensorDataset1D(images_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    generated_any = False
    inputs_10x_path = output_dir / args.inputs_name_10x
    outputs_10x_path = output_dir / args.outputs_name_10x

    if not args.skip_10x:
        print(f"Reconstructing 10x latents for {num_samples} images with {samples_per_image} samples per image...")
        inputs_10x, outputs_10x = encode_images_to_latents_10x(
            dataloader, sd_vae, resn_vae, device, num_samples, samples_per_image=samples_per_image
        )
        torch.save(inputs_10x, inputs_10x_path)
        torch.save(outputs_10x, outputs_10x_path)
        print(f"Saved 10x encoded latents to {inputs_10x_path}")
        print(f"Saved 10x reconstructions to {outputs_10x_path}")
        generated_any = True

    if args.generate_1x:
        print(f"Reconstructing 1x latents for {num_samples} images...")
        inputs_latents, outputs_latents = encode_images_to_latents(dataloader, sd_vae, resn_vae, device, num_samples)
        inputs_path = output_dir / args.inputs_name
        outputs_path = output_dir / args.outputs_name
        torch.save(inputs_latents, inputs_path)
        torch.save(outputs_latents, outputs_path)
        print(f"Saved encoded latents to {inputs_path}")
        print(f"Saved reconstructions to {outputs_path}")
        generated_any = True

    if not generated_any:
        print("[WARN] No latent outputs were generated. Disable --skip-10x or enable --generate-1x.")

    if args.text2img_output:
        print(f"Generating text->image prior_shared latents for {num_samples} images...")
        vocab = load_cub_caption_vocab(args.data_dir)
        w2i = vocab["w2i"]
        cap_tensor_full = build_caption_tensor_for_all(captions, w2i, max_len=32, cap_idx=args.text2img_caption_index)
        text2img = generate_text2img_prior_shared(
            model, cap_tensor_full, device, batch_size=args.batch_size, mm_utils=mm_utils
        )
        text2img_path = Path(args.text2img_output)
        text2img_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(text2img, text2img_path)
        print(f"Saved text->image prior_shared latents to {text2img_path}")

    print("Dataset alignment check complete. All modalities share the same ordering.")


if __name__ == "__main__":
    main_pregen_cub()
