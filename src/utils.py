
import os
import sys
import time
import shutil
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from torch.distributions import Normal, Independent
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset, Dataset, DataLoader
from collections import OrderedDict
from enum import Enum


class CrossModalEvalForwardMode(str, Enum):
    """How ``forward``-style cross-modal generation is run during **eval** (metrics, plots).

    **Grids (qualitative):** In typical cross-modal image grids, **each column** shares one
    conditioning posterior ``z`` or ``w``. The ``_CTRL`` modes **fix the other latent along
    each row** (one sampled prior or controlled posterior facet per row, reused across
    columns) so rows stay coherent. **Without** ``_CTRL``, draws **within a row** are
    independent across columns—better for **quantitative** Monte Carlo summaries.

    - **PRIOR** — i.i.d. prior samples per slot; quantitative / independent draws.

    - **PRIOR_CTRL** — one prior sample **per row**, broadcast across columns; qualitative
      grid layout.

    - **POSTERIOR** — target latents from posteriors with **random cyclic shifts**
      (shuffled pairing); stochastic aggregate metrics.

    - **POSTERIOR_CTRL** — deterministic pairing via a separate minibatch ``data_ctrl``
      (encoder posteriors on those inputs), one posterior sample **per row**, broadcast 
      across columns; qualitative grid layout. Requires ``data_ctrl``.

    - **POSTERIOR_NONSHUF** — **Special:** rows and columns are built from the **same**
      underlying batch so conditioning is aligned; the grid **diagonal** corresponds to
      using matching row/column inputs, so diagonal cells are **the same as or close to**
      the true inputs (self-consistent / reconstruction-like), unlike the other modes.
      See ``idmvae_self_and_cross_modal_generation_impl`` for diagonal handling.

    """

    PRIOR = "prior"
    PRIOR_CTRL = "prior_ctrl"
    POSTERIOR = "posterior"
    POSTERIOR_CTRL = "posterior_ctrl"
    POSTERIOR_NONSHUF = "posterior_nonshuf"


# =============================
# Functions from utils_mvp.py
# =============================


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


class Constants(object):
    eta = 1e-8


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print(
            "====> [{}] Time: {:7.3f}s or {}".format(
                self.name, self.elapsed, time.strftime("%H:%M:%S", self.elapsedH)
            )
        )


def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, "{}.old".format(filepath))
    torch.save(vs, filepath)


def save_model_light(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


# Added for convenience:
def load_model_light(filepath, modelC, args, device="cpu"):
    """
    Loads a saved MMVAE+ model from the given filepath.

    Args:
        filepath (str): Path to the saved model file (expects a `.rar` format).
        modelC (type): The model class constructor, e.g., IDMVAE_PolyMNIST_5modalities.
        args (argparse.Namespace): Arguments required to instantiate the model.
        device (str): The device to load the model on ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model = modelC(args).to(
        device
    )  # Instantiate the model.
    model.load_state_dict(
        torch.load(filepath, map_location=device)
    )  # Load the state dictionary: weights, biases, etc.
    return model


def unpack_data_PM(data, device="cuda"):
    data_nolabel = data[0]
    n_idxs = len(data_nolabel)
    return [data_nolabel[idx].to(device) for idx in range(n_idxs)], data[1].to(device)


def unpack_data_PM_quadrant(data, device="cuda"):
    data_nolabel = data[0]  # A list of tensors (5 modalities)
    labels = (
        data[1] if len(data) > 1 else None
    )  # A tuple (digit labels, quadrant labels)
    n_idxs = len(data_nolabel)  # Should be 5 (for 5 modalities)
    # Move input images (5 modalities) to device
    data_nolabel = [data_nolabel[idx].to(device) for idx in range(n_idxs)]

    # Move labels (digit and quadrant) to device
    if len(labels) == 1:  # Only digit labels are present
        labels = data[1].to(device)  # Move digit labels to device
    elif len(labels) > 1:  # Both digit and quadrant labels are present, as well as pair indices
        labels = (
            # labels[0].to(device),  # Move digit labels to device
            # NOTE: pair indices (labels[2]) is not used in training or evaluation, all the upcoming labels
            # are put in the rear part of the tuple, which is easier to be compatible or ignored in other functions.
            [labels[0][idx].to(device) for idx in range(n_idxs)],  # Move digit labels to device
            [labels[1][idx].to(device) for idx in range(n_idxs)],  # Move quadrant labels to device
            [labels[2][idx].to(device) for idx in range(n_idxs)]   # Move pair indices to device
        )

    return data_nolabel, labels  # Returns the modified tensors


# CUB dataset
def unpack_data_CUB(data, device="cuda"):
    return [data[0][0].to(device), data[1][0].to(device)]

# CUBICC dataset
def unpack_data_CUBICC(data, device="cuda"):
    data_nolabel = data[0]
    n_idxs = len(data_nolabel)
    return [data_nolabel[idx].to(device) for idx in range(n_idxs)], data[1].to(device)

# CUBCluster8 dataset
def unpack_data_CUBcluster8(data, device="cuda"):
    data_nolabel = data[0]
    n_idxs = len(data_nolabel)  # number of modalities, should be 2 of CUB
    num_labelTypes = len(data[1])  # number of types of labels, should be 5 of CUB: (lbl_cluster, lbl_cat, lbl_dir, img_id, dataset_index)
    # [[128, 3, 64, 64], [128, 32, 1590]], [128]
    return [data_nolabel[idx].to(device) for idx in range(n_idxs)], tuple(data[1][idx].to(device) for idx in range(num_labelTypes))


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


class NonLinearLatent_Classifier(nn.Module):
    """Non-linear Latent classifier defintion."""

    def __init__(self, in_n, out_n):
        super(NonLinearLatent_Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_n, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_n),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


def make_mlp(input_dim, hidden_dim, num_layers, num_classes):
    layers = []
    prev = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        prev = hidden_dim
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


def get_and_save_structured_polymnist_samples(
    dataset, num_testing_images, device, save_dir, num_rows=4, num_cols=10
    ):
    """
    Finds, saves, and returns a structured grid of PolyMNIST samples.
    Saves samples as individual PNGs and a single .pt file for easy reloading.
    Also returns a verification grid for wandb and prints the index mapping.
    Function to get a structured grid of PolyMNIST samples for qualitative examples.
    It finds a specific (digit, quadrant) pair for each cell in a grid,
    following a predefined pattern.
    This version correctly finds an individual image for each modality that
    matches the required (digit, quadrant) pair for each of the 40 grid slots.
    0.0, 1.1, 2.2, 3.3, 4.0, 5.1, 6.2, 7.3, 8.0, 9.1,
    0.1, 1.2, 2.3, 3.0, 4.1, 5.2, 6.3, 7.0, 8.1, 9.2,
    0.2, 1.3, 2.0, 3.1, 4.2, 5.3, 6.0, 7.1, 8.2, 9.3,
    0.3, 1.0, 2.1, 3.2, 4.3, 5.0, 6.1, 7.2, 8.3, 9.0,
    """
    print(f"Searching for and saving structured grid of samples to {save_dir}...")
    
    # 1. Define the target (digit, quadrant) pairs for the grid
    target_pairs = []
    for r in range(num_rows):
        for c in range(num_cols):
            digit = c
            quadrant = (c + r) % 4
            target_pairs.append((digit, quadrant))

    # 2. Data structures for results
    num_modalities = 5
    num_samples = num_rows * num_cols
    
    # This will hold the final tensors, one for each modality
    final_images_per_mod = [[None] * num_samples for _ in range(num_modalities)]
    # This will hold the info for logging, one for each modality
    final_info_per_mod = [[None] * num_samples for _ in range(num_modalities)]
    
    # 3. Find one image for each of the 40 pairs, for each of the 5 modalities
    for mod_idx in range(num_modalities):
        print(f"--- Searching for samples for Modality {mod_idx} ---")
        
        # For each modality, we need to find all 40 pairs
        found_mask = [False] * num_samples

        dataset_indices = list(range(num_testing_images))
        random.shuffle(dataset_indices)

        for data_idx in dataset_indices:
            imgs, target = dataset[data_idx]
            digit_label = target[0][0]  # labels = (digit_labels, quadrant_labels, pair_indices)
            quadrant_label = target[1][mod_idx] # Get the quadrant for the current modality
            current_pair = (digit_label, quadrant_label)
            
            # Check if this pair is needed for any of the 40 slots
            for target_idx, required_pair in enumerate(target_pairs):
                if current_pair == required_pair and not found_mask[target_idx]:
                    # Found a match for this slot for this modality
                    final_images_per_mod[mod_idx][target_idx] = imgs[mod_idx].to(device)
                    final_info_per_mod[mod_idx][target_idx] = (data_idx, digit_label, quadrant_label)
                    found_mask[target_idx] = True
                    # We found a sample for this slot, but we continue searching the dataset
                    # to fill other slots. We break here to avoid using the same data_idx
                    # to fill multiple slots for the same modality in one go.
                    break 
            
            if all(found_mask):
                print(f"Found all samples for Modality {mod_idx}.")
                break
        
        if not all(found_mask):
            raise RuntimeError(f"Could not find all required samples for Modality {mod_idx}.")

    # 4. Print the mapping
    print("\n--- Found Sample Mapping (Grid Index -> Original Index, Digit.Quadrant) ---")
    for mod_idx in range(num_modalities):
        print(f"m{mod_idx}:")
        log_lines = [""] * num_rows
        for i in range(num_samples):
            row = i // num_cols
            orig_idx, d, q = final_info_per_mod[mod_idx][i]
            log_lines[row] += f"{i}_{d}.{q} (idx:{orig_idx:04d}), "
        for line in log_lines:
            print(line)
    print("--------------------------------------------------------------------")

    # 5. Stack the lists of tensors into the final output tensors
    outputs = [torch.stack(mod_imgs) for mod_imgs in final_images_per_mod]

    # 6. Save and create verification grid
    os.makedirs(save_dir, exist_ok=True)
    torch.save(outputs, os.path.join(save_dir, 'qualitative_samples.pt'))
    print(f"Saved sample tensors to {os.path.join(save_dir, 'qualitative_samples.pt')}")
    
    # Save individual images for inspection
    png_save_dir = os.path.join(save_dir, 'PNGs')
    os.makedirs(png_save_dir, exist_ok=True)
    for mod_idx in range(num_modalities):
        for sample_idx in range(num_samples):
            img_tensor = outputs[mod_idx][sample_idx]
            orig_idx, d, q = final_info_per_mod[mod_idx][sample_idx]
            save_image(img_tensor, os.path.join(png_save_dir, f'mod_{mod_idx}_sample_{sample_idx:02d}_{orig_idx:04d}.{d}.{q}.png'))
    print(f"Saved individual sample images to {png_save_dir}")

    all_modality_imgs = torch.cat(outputs, dim=0)
    verification_grid = make_grid(all_modality_imgs, nrow=num_cols)

    # The final output is a list of 5 tensors, each of shape (40, C, H, V)
    return outputs, verification_grid


def load_structured_samples(save_dir, device, num_cols=10):
    """
    Loads the structured samples saved by get_and_save_structured_samples.
    Also returns a verification grid for wandb.
    """
    load_path = os.path.join(save_dir, 'qualitative_samples.pt')
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Could not find saved samples at {load_path}")
    
    samples = torch.load(load_path)
    samples_on_device = [s.to(device) for s in samples]
    print(f"Successfully loaded qualitative samples from {load_path}")
    
    # Create the verification grid from the loaded samples (using ALL modalities)
    all_modality_imgs = torch.cat(samples_on_device, dim=0)
    verification_grid = make_grid(all_modality_imgs, nrow=num_cols)

    return samples_on_device, verification_grid


def get_test_CUBcluster8_samples(CUBcluster8, num_testing_images, device, args, pretrained_vae=None, seed=42):
    """
    Function to get CUBcluster8 samples for qualitative examples of cross-reconstruction at test time.
    It randomly samples one image from each of the 8 clusters.
    A seed is used to make the random sampling deterministic.
    """
    random.seed(seed)
    samples_data = []
    imgs = []
    cap_tensors = []
    samples = []
    labels = []
    raw_selected_captions = []
    raw_all_captions = []

    for i in range(8):
        while True:
            random_idx = random.randint(0, num_testing_images - 1)
            datas, targets_tuple = CUBcluster8.__getitem__(random_idx)
            
            target_cluster_label = targets_tuple[0]

            # cluster label is from 1 to 8, the 0 only in train set as "Other" cluster
            if target_cluster_label == i + 1:
                img, cap_tensor = datas

                # Get raw caption text
                _img_idx, cap_idx = CUBcluster8.pairs[random_idx]
                raw_selected_caption = CUBcluster8.captions[_img_idx][cap_idx]
                raw_all_caption = CUBcluster8.captions[_img_idx]

                img = img.to(device)
                cap_tensor = cap_tensor.to(device)
                
                samples_data.append((img, cap_tensor))
                imgs.append(img)
                cap_tensors.append(cap_tensor)
                labels.append(targets_tuple)
                raw_selected_captions.append(raw_selected_caption)
                raw_all_captions.append(raw_all_caption)
                
                break
    
    if not imgs:
        raise RuntimeError("No samples found for the given indices or IDs.")

    samples = [torch.stack(imgs, dim=0), torch.stack(cap_tensors, dim=0)]
        
    return samples, labels, raw_selected_captions, raw_all_captions
# =============================
def get_and_log_CUBcluster8_samples_by_Idx_or_ID(
        CUBcluster8, num_testing_images, device, args, indices=None, ids=None
):
    """
    indices_group is the group of dataset indices to sample from.
    Get samples from the CUBcluster8 dataset by dataset indices or image IDs,
    and log them using the provided WandB logger.

    By checking the prediction accuracy and score of cluster img and 1st caption, and direction
    potential indices group:
    CUBcluster8 64x64:
        Test Set:
            Cluster : 1     2     3     4     5     6     7     8
            Left1   : 238   9974  2993  10487 5914  6082  7716  8624
            Right1  : 222   9978  3076  10391 5898  6071  8050  11774

        Validation Set:
            Cluster : 1     2     3     4     5     6     7     8
            Left1   : 254   2498  3085  10444 5911  6061  7688  8566
            Right1  : 271   9970  3087  10447 5826  6062  7675  11767

    return:
        samples: list in modality, torch.stack each modality data sensor
        labels: list of labels corresponding to each sample
    """

    # group of selected dataset_index
    if args.dataset == "CUBcluster8":
        indices_group = indices if indices is not None \
                        else [
                # Test Set:
                # Cluster : 1      2      3      4      5      6      7      8
                            238,   9974,  2993,  10487,                              # Left1   : 4 (default)
                                                        # 5914,  6082,  7716,  8624,   # Left1   : 4
                            # 222,   9978,  3076,  10391,                              # Right1  : 4
                                                        5898,  6071,  8050,  11774,  # Right1  : 4 (default)

                # Validation Set:
                # Cluster : 1      2      3      4      5      6      7      8
                            254,   2498,  3085,  10444,                              # Left1   : 4 (default)
                                                        # 5911,  6061,  7688,  8566,   # Left1   : 4
                            # 271,   9970,  3087,  10447,                              # Right1  : 4
                                                        5826,  6062,  7675,  11767,  # Right1  : 4 (default)
                        ]
    elif args.dataset == "CUBcluster8_256":
        indices_group = indices if indices is not None \
                        else [
            # Test Set:
            # Cluster : 1      2      3      4      5      6      7      8

                        ]
    ids_group = ids if ids is not None \
                else None

    if not indices_group and not ids_group:
        raise ValueError("Either 'indices' or 'ids' must be provided.")

    imgs = []
    cap_tensors = []
    samples = []
    labels = []
    raw_selected_captions = []  # just the first of the image
    raw_all_captions = []       # all 10 captions for the image

    if indices_group is not None:
        # Create a reverse map from the full dataset's img_idx to the local pair index
        # take the first caption for each image, so we map img_idx -> first pair_idx due to the reverse map
        # or if without the reverse, we will that the last caption
        img_idx_to_pair_idx = {dataset_idx_pair[0]: pair_idx for pair_idx, 
                               dataset_idx_pair in reversed(list(enumerate(CUBcluster8.pairs)))}

        # Optional extension: choose alternative caption indices per image.
        img_cap_idxs_to_pair_idx = {}
        for pair_idx, dataset_idx_pair in enumerate(CUBcluster8.pairs):
            if dataset_idx_pair[0] in indices_group:
                img_cap_idxs_to_pair_idx.setdefault(dataset_idx_pair[0], []).append(pair_idx)

        for img_idx in indices_group:
            pair_idx = img_idx_to_pair_idx.get(img_idx)
            if pair_idx is not None:
                datas, target_labels_tuple = CUBcluster8.__getitem__(pair_idx) # idx input of the __getitem__ is not the dataset_index, just the inner index of the subset
                img, cap_tensor = datas
                # Get the raw caption text directly
                _img_idx, cap_idx = CUBcluster8.pairs[pair_idx]
                assert _img_idx == img_idx, f"Image index mismatch: {img_idx} != {_img_idx}"
                raw_selected_caption = CUBcluster8.captions[_img_idx][cap_idx]  # _img_idx = img_idx
                raw_all_caption = CUBcluster8.captions[_img_idx]  # all 10 captions for the image

                img = img.to(device)
                cap_tensor = cap_tensor.to(device)
                imgs.append(img)
                cap_tensors.append(cap_tensor)
                labels.append(target_labels_tuple) # list of tuples
                raw_selected_captions.append(raw_selected_caption)
                raw_all_captions.append(raw_all_caption)

    if not imgs:
        raise RuntimeError("No samples found for the given indices or IDs.")

    samples = [torch.stack(imgs, dim=0), torch.stack(cap_tensors, dim=0)]

    return samples, labels, raw_selected_captions, raw_all_captions



# Helper function to plot captions so to smoothly log on WandB
def plot_text_as_image_tensor(
    sentences_lists_of_words, pixel_width=64, pixel_height=384, fontsize=8
):
    imgs = []
    for sentence in sentences_lists_of_words:
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
        fig = plt.figure(figsize=(pixel_width * px, pixel_height * px))
        plt.text(
            x=1,
            y=0.5,
            s="{}".format(
                " ".join(
                    i + "\n" if (n + 1) % 1 == 0 else i
                    for n, i in enumerate(
                        [word for word in sentence.split() if word != "<eos>"]
                    )
                )
            ),
            fontsize=fontsize,  # 8 for 64*64, default is 7
            verticalalignment="center_baseline",
            horizontalalignment="right",
        )
        plt.axis("off")

        # Draw the canvas and retrieve the image as a NumPy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # Matplotlib backends differ (Agg vs TkAgg, etc.). Prefer RGBA buffer APIs.
        if hasattr(fig.canvas, "buffer_rgba"):
            image_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
            image_np = image_rgba[:, :, :3]
        elif hasattr(fig.canvas, "tostring_rgb"):
            image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
        elif hasattr(fig.canvas, "tostring_argb"):
            # ARGB -> RGB
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
            image_np = argb[:, :, 1:4]
        else:
            raise RuntimeError(f"Unsupported Matplotlib canvas type: {type(fig.canvas).__name__}")

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = (
            torch.from_numpy(image_np).permute(2, 0, 1).float() / 255
        )  # Normalize to [0, 1]
        imgs.append(image_tensor)
        # Clean up the figure
        plt.close(fig)
    return torch.stack(imgs, dim=0)


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class DigitClassifier(nn.Module):
    def __init__(self, input_size=28, res_s0=7, res_nf=64):
        super().__init__()
        s0 = self.s0 = res_s0
        nf = self.nf = res_nf
        nf_max = self.nf_max = 1024
        size = input_size

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(actvn(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Below are adapted from Junwen's C-DSVAE implementation.
class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(ContrastiveLoss, self).__init__()
        # Temperature.
        self.tau = tau
        # Unit-length normalization.
        self.normalize = normalize

    def forward(self, xi, xj):
        x = torch.cat((xi, xj), dim=0)
        device = x.device

        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(
                torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T
            )
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0), device=device) / self.tau)
        mi = torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum))

        return mi, None


# =============================
# Functions from utils_vcca_eval.py
# =============================


class EmbeddedDatasetWithPriors(Dataset):
    """
    Extends embedding of a dataset with posterior latent means by also generating
    samples from the prior distribution in the same latent space.

    posterior samples are colored by their true labels;
    prior samples carry label = None (or a sentinel) so you can color them differently.
    """

    def __init__(
        self,
        base_dataset,
        vae,
        encoder,
        device="cpu",
        n_prior=None,
        prior_sampler=None,
        block_size=128,
        condition_type=None,
    ):
        """
        Args:
            base_dataset: torch Dataset yielding (x, y) pairs
            encoder: function or nn.Module(x) -> (mu, logvar), returns posterior parameters / means
            device: device for computing embeddings
            n_prior: number of prior samples to generate (if None, defaults to len(base_dataset))
            prior_sampler: callable taking shape (n, dim) and returning array/tensor of prior samples.
                           If None, standard normal is used.
            block_size: batch size for embedding
            condition_type: passed to encoder if it needs conditioning; currently unused
        """
        self.device = device
        self.block_size = block_size

        # Obtain posterior embeddings
        post_reps, post_labels = self._embed_posterior(
            vae, encoder, base_dataset, condition_type=condition_type
        )
        self.post_reps = post_reps
        self.post_labels = post_labels

        # Determine latent dimension
        latent_dim = self.post_reps.shape[1]

        # Setup prior sampler
        if prior_sampler is None:
            prior_sampler = lambda n, d: torch.randn(n, d)

        if n_prior is None:
            n_prior = len(base_dataset)

        # Generate prior samples
        self.prior_reps = prior_sampler(n_prior, latent_dim).to(device)
        # Assign sentinel label for priors (e.g. -1)
        self.prior_labels = torch.full((n_prior,), -1, dtype=torch.long)

        # Concatenate posterior and prior
        self.reps = torch.cat([self.post_reps, self.prior_reps.cpu()], dim=0)
        self.labels = torch.cat([self.post_labels.cpu(), self.prior_labels], dim=0)

    def _embed_posterior(
        self, vae, encoder, dataset, device, condition_type=None, use_mean=True
    ):

        if encoder is not None:
            encoder.eval()

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.block_size, shuffle=False
        )

        ys = []
        reps = []
        with torch.no_grad():
            for x, y in data_loader:
                # NOTE: currently, data_loader here returns only one modality but in list
                if isinstance(x, list):  # If x contains multiple modalities
                    x = [mod.to(device) for mod in x]  # Move each modality to device
                    x = x[0]
                else:
                    x = x.to(device)

                if condition_type is None:
                    if isinstance(y, list):
                        y = y[0].to(device)
                    else:
                        y = y.to(device)
                elif condition_type == "shared":
                    if isinstance(y, (tuple, list)):  # If y contains multiple labels
                        y = y[0][0].to(device)
                    else:
                        y = y.to(device)
                elif condition_type == "private":
                    if isinstance(y, (tuple, list)):
                        y = y[1][0].to(device)
                    else:
                        y = y.to(device)

                if encoder is not None:
                    u_given_x_mean, u_given_x_logvar = encoder(x)  # [B, W+Z]

                    if use_mean:
                        reps.append(u_given_x_mean.detach())
                    else:
                        reps_dist = vae.qu_x(u_given_x_mean, u_given_x_logvar)
                        u_given_x_sample = reps_dist.rsample(torch.Size([1]))
                        reps.append(u_given_x_sample.detach())

                else:
                    reps.append(x)
                ys.append(y)

            ys = torch.cat(ys, 0)

        return reps, ys

    def __getitem__(self, index):
        y = self.target[index]
        x = self.means[index // self.block_size][index % self.block_size]  # [W+Z]

        return x, y

    def __len__(self):
        return self.target.size(0)


class EmbeddedDataset_visualization: # tSNE, UMAP, PCA, etc.
    """
    Suitable for PolyMNIST and CUB datasets.
    Single view input. -> y = y[x][0] for PolyMNIST
    """
    
    def __init__(self, base_dataloader, vae, encoder, device='cpu', condition_type=None, use_mean=False, view_idx=None, batch_size=128):
        self.block_size = batch_size
        if encoder is not None:
            encoder = encoder.to(device)
        self.means, self.target = self._embed(
            vae, encoder, base_dataloader, device, condition_type=condition_type, use_mean=use_mean, view_idx=view_idx
            ) # (BLS,W+Z), (N)

    def _embed(self, vae, encoder, data_loader, device, condition_type=None, use_mean=False, view_idx=None): # use_mean=True by default
        if encoder is not None:
            encoder.eval()

        ys = []
        reps = []
        with torch.no_grad():
            for x, y in data_loader:

                # NOTE: currently, data_loader here returns only one modality but in list
                if isinstance(x, list):  # If x contains multiple modalities
                    x = x[view_idx].to(device) if view_idx is not None else x[0].to(device)
                else:
                    x = x.to(device)

                # Compatibility path for condition_type=None after __getitem__ updates.
                if condition_type is None: 
                    if isinstance(y, list):
                        y = y[0].to(device) 
                    else:
                        y = y.to(device)
                elif condition_type == 'shared':
                    # Shared-label extraction for multimodal label containers.
                    if isinstance(y, (tuple, list)):  # If y contains multiple labels
                        if isinstance(y[0], (tuple, list)):
                            y = y[0][view_idx].to(device) # suitable for PolyMNIST
                        else:
                            y = y[0].to(device)  # suitable for CUBcluster8
                    else:
                        y = y.to(device)
                elif condition_type == 'private':
                    if isinstance(y, (tuple, list)):
                        if isinstance(y[1], (tuple, list)):
                            y = y[1][view_idx].to(device)  # suitable for PolyMNIST
                        else:
                            y = y[1].to(device)  # suitable for CUBcluster8
                    else:
                        y = y.to(device)              

                if encoder is not None:
                    u_given_x_mean, u_given_x_logvar = encoder(x) # [B, W+Z]

                    if use_mean:
                        reps.append(u_given_x_mean.detach())
                    else:
                        reps_dist = vae.qu_x(u_given_x_mean, u_given_x_logvar)
                        u_given_x_sample = reps_dist.rsample(torch.Size([1]))
                        u_given_x_sample = u_given_x_sample.squeeze(0)
                        reps.append(u_given_x_sample.detach())
                else:
                    reps.append(x)
                ys.append(y)

            ys = torch.cat(ys, 0)

        return reps, ys

    def __getitem__(self, index):
        y = self.target[index]
        x = self.means[index // self.block_size][index % self.block_size] # [W+Z]

        return x, y

    def __len__(self):
        return self.target.size(0)


class EmbeddedDataset:
    # BLOCK_SIZE = 128  # vcca: 256

    def __init__(self, base_dataset, vae, encoder, device="cpu", condition_type=None, batch_size=128):
        self.block_size = batch_size
        if encoder is not None:
            encoder = encoder.to(device)
        self.means, self.target = self._embed(
            vae, encoder, base_dataset, device, condition_type=condition_type
        )  # (num_blocks, W+Z), (N,)

    def _embed(
        self, vae, encoder, dataset, device, condition_type=None, use_mean=True
    ):
        # Put encoder into eval mode
        if encoder is not None:
            encoder.eval()

        loader = DataLoader(dataset, batch_size=self.block_size, shuffle=False)

        reps = []
        ys = []

        with torch.no_grad():
            for batch in loader:
                # unpack either (x, y) or (views, mask, y)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    views_list, mask, y = batch
                    # pick the first view (since `encoder` is e.g. model.encoders[0])
                    x = views_list[0].to(device)
                else:
                    x, y = batch
                    if isinstance(x, list):
                        x = x[0].to(device)
                    else:
                        x = x.to(device)
                    y = y.to(device)

                # encode + optionally sample
                mu, logvar = encoder(x)  # [B, W+Z]
                if use_mean:
                    z = mu.detach()
                else:
                    dist = vae.qu_x(mu, logvar)
                    z = dist.rsample(torch.Size([1])).squeeze(0).detach()

                reps.append(z)
                ys.append(y)

        # concatenate blocks of shape [1,B,W+Z] → [N, W+Z];  ys → [N]
        reps = torch.cat(reps, dim=0)
        ys = torch.cat(ys, dim=0)
        return reps, ys

    def __getitem__(self, idx):
        return self.means[idx], self.target[idx]

    def __len__(self):
        return self.target.size(0)


def split(dataset, size, split_type):
    if split_type == "Random":
        data_split, _ = torch.utils.data.random_split(
            dataset, [size, len(dataset) - size]
        )
    elif split_type == "Balanced":
        class_ids = {}
        for idx, (_, y) in enumerate(dataset):
            if isinstance(y, torch.Tensor):
                y = y.item()
            if y not in class_ids:
                class_ids[y] = []
            class_ids[y].append(idx)

        ids_per_class = size // len(class_ids)

        selected_ids = []

        for ids in class_ids.values():
            selected_ids += list(
                np.random.choice(ids, min(ids_per_class, len(ids)), replace=False)
            )
        data_split = Subset(dataset, selected_ids)

    return data_split


def build_matrix(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    xs = []
    ys = []

    for x, y in data_loader:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)

    if xs.is_cuda:
        xs = xs.cpu()
    if ys.is_cuda:
        ys = ys.cpu()

    return xs.data.numpy(), ys.data.numpy()


def train_and_evaluate_linear_model_from_matrices(
    x_train, y_train, solver="saga", multi_class="multinomial", tol=0.1, C=10
):
    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x_train, y_train)
    return model


def train_and_evaluate_linear_model(
    train_set, test_set, solver="saga", multi_class="multinomial", tol=0.1, C=10
):
    x_train, y_train = build_matrix(train_set)
    x_test, y_test = build_matrix(test_set)

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x_train, y_train)

    test_accuracy = model.score(x_test, y_test)
    train_accuracy = model.score(x_train, y_train)

    return train_accuracy, test_accuracy


