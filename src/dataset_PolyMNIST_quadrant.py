import numpy as np
import argparse
import torch
import os
import glob

from torch.utils.data import Dataset, Subset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image

import wandb


class PolyMNISTDataset(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, unimodal_datapaths, transform=None, target_transform=None):
        """
            Args:
                unimodal_datapaths (list of str): list of paths to weakly-supervised
                    unimodal datasets with samples that correspond by index.
                    Therefore the numbers of samples of all datapaths should
                    match.
                transform: tranforms on colored MNIST digits.
                target_transform: transforms on labels.
                condition_type (str): 'shared' or 'private', determines which label to return.
        """
        super().__init__()
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform
        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        # self.file_paths = {}
        for dp in unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            # Sort files based on index + digit (to enforce alignment across modalities)
            files.sort(key=lambda f: ".".join(os.path.basename(f).split(".")[:2]))  # sort by index.digit
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        # self.num_files = len(next(iter(self.file_paths.values())))
        for files in self.file_paths.values():
            assert len(files) == num_files, (
                f"Folder {dp} has {len(files)} images, but expected {num_files}"
            )
        self.num_files = num_files
        self.quadrant_counts = {m: {q: 0 for q in range(4)} for m in range(self.num_modalities)}
        self.image_quadrant_mapping = {m: [] for m in range(self.num_modalities)}

    @staticmethod
    def _add_background_image(background_image_pil, mnist_image_tensor, view
                              ):
        """
        Conbine a single MNIST image with a random 64x64 crop of a background from
        the background_image_pil. The colors of the background image are inverted
        at the location of the MNIST image for views 1, 2, 3, and for views 0 and 4
        we set the color.

            Args:
                background_image_pil (PIL.Image): background image.
                mnist_image_tensor (torch.Tensor): MNIST image (28x28).
                view (int): used to determine how to color/invert the digit.

            Returns:
                PIL.Image: new image with MNIST digit on background.
                (torch.Tensor, int): (a 3x64x64 image, quadrant label in [0..3])
        """

        # Create a blank 64×64 canvas
        translated_mnist = torch.zeros((64, 64), dtype=mnist_image_tensor.dtype)

        # Ensure input is in (1, 1, 28, 28) format
        mnist_image_tensor = mnist_image_tensor.unsqueeze(0).unsqueeze(0).float()  # Add batch & channel dim

        new_size = 32
        # Rescale MNIST digit
        mnist_image_tensor_scaled = torch.nn.functional.interpolate(
            mnist_image_tensor, 
            size=(32, 32), 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions

        # Compute random position
        # Random quadrant => offsets for top-left corner
        # 0: (32, 32), 1: (0, 32), 2: (0, 0), 3: (32, 0)
        offsets = np.random.randint(low=0, high=2, size=(2,))
        x_start = offsets[0] * new_size
        y_start = offsets[1] * new_size
        # in quadrant [0, 1, 2, 3]

        # [1(0,0), 0(0,1),
        #  2(1,0), 3(1,1)]
        # Approach 1: Convert offsets to a tuple:
        offsets_tuple = tuple(offsets)
        if offsets_tuple == (0, 1):
            quadrant_label = 0
        elif offsets_tuple == (0, 0):
            quadrant_label = 1
        elif offsets_tuple == (1, 0):
            quadrant_label = 2
        elif offsets_tuple == (1, 1):
            quadrant_label = 3

        # Place the scaled MNIST digit at the random quadrant
        translated_mnist[x_start:x_start+new_size, y_start:y_start+new_size] = mnist_image_tensor_scaled
        mnist_image_tensor = translated_mnist  # Ensure correct shape after modification

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 16)
        # import pdb;pdb.set_trace()

        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 64)
        y_c = np.random.randint(0, background_image_pil.size[1] - 64)
        new_img_pil = background_image_pil.crop((x_c, y_c, x_c + 64, y_c + 64))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img_pil)

        # Different color logic for different 'views'
        if view in [1, 2, 3]:
            # Invert the colors at the location of the number
            new_img[:, img_binarized] = 1 - new_img[:, img_binarized]
        elif view == 0: # red
            new_img[0, img_binarized] = (mnist_image_tensor[img_binarized] / 255.0)
            new_img[1, img_binarized] = 0.0
            new_img[2, img_binarized] = 0.0
        else:
            # view==4 or any other case, blue
            new_img[0, img_binarized] = 0.0
            new_img[1, img_binarized] = 0.0
            new_img[2, img_binarized] = (mnist_image_tensor[img_binarized] / 255.0)

        return new_img, quadrant_label

    def __getitem__(self, index):
        """
        Returns a tuple (images, digit_labels/quadrant_labels) where each element is a list of
        length `self.num_modalities`.
        
        Returns:
            images (list of tensors): List of transformed images (one per modality).            
            label (int): Either the digit label or the quadrant label, based on condition_type.
                digit_label (int): The digit label extracted from filename.
                quadrant_label (int): The quadrant label extracted from filename.
        
        """
        # For each modality, get the PNG path at the given index
        files = [self.file_paths[dp][index] for dp in self.unimodal_datapaths]
        # Open each image and get the label
        images = [Image.open(files[m]) for m in range(self.num_modalities)]

        # Parse the digit and quadrant label from filename for each modality
        # e.g. "m0/1234.5.2.png" => global=1234, digit=5, quadrant=2
        pair_indices = [int(os.path.basename(files[m]).split('.')[0]) for m in range(self.num_modalities)]
        digit_labels = [int(os.path.basename(files[m]).split('.')[-3]) for m in range(self.num_modalities)]  # robust for paths like /data/mnist.v1/m0/1234.5.2.png
        quadrant_labels = [int(os.path.basename(files[m]).split('.')[-2]) for m in range(self.num_modalities)]

        if self.target_transform:
            pair_indices = [self.target_transform(label) for label in pair_indices]
            digit_labels = [self.target_transform(label) for label in digit_labels]
            quadrant_labels = [self.target_transform(label) for label in quadrant_labels]

        # Final label format used by this project:
        # (shared_labels, private_labels, sample_indices)
        labels = (digit_labels, quadrant_labels, pair_indices)

        # transforms
        if self.transform:
            images = [self.transform(img) if not isinstance(img, torch.Tensor) else img for img in images]

        return images, labels

    def __len__(self):
        return self.num_files

    def log_quadrant_stats(self):
        for modality in range(self.num_modalities):
            wandb.log({f"quadrant_count/m{modality}": self.quadrant_counts[modality]})

    def log_sample_images(self):
        for modality in range(self.num_modalities):
            for quadrant in range(4):
                sample_images = []
                # for index in range(len(self)):
                #     images, _, quadrants = self.__getitem__(index)
                #     if quadrants[modality] == quadrant:
                for index, q in self.image_quadrant_mapping[modality]:
                    if q == quadrant:
                        images, _ = self.__getitem__(index)
                        sample_images.append(wandb.Image(images[modality]))
                        if len(sample_images) == 2:
                            break
                wandb.log({f"samples/m{modality}_q{quadrant}": sample_images})


class PolyMNISTDataset_pt(Dataset):
    """Multimodal MNIST Dataset from .pt files."""

    def __init__(self, data_dir, transform=None, target_transform=None):
        """
            Args:
                data_dir (str): path to the directory containing the .pt files
                                (e.g., '.../PolyMNIST_pt/train').
                transform: transforms on colored MNIST digits.
                target_transform: transforms on labels.
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images = torch.load(os.path.join(self.data_dir, "images.pt"))
        self.labels = torch.load(os.path.join(self.data_dir, "labels.pt"))

        self.num_files = len(self.images)
        if self.num_files != len(self.labels):
            raise ValueError("Number of images and labels do not match!")

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.

        Returns:
            images (list of tensors): List of transformed images (one per modality).
            labels (tuple): 
                Tuple containing (pair_index, digit_labels, quadrant_labels).
                Or Tuple containing (digit_labels, quadrant_labels).
        
        NOTE: Called by
        def unpack_data_PM_quadrant(data, device="cuda"):
        def get_and_save_structured_polymnist_samples(    
        """
        images = self.images[index]
        labels = self.labels[index]

        # The saved images are already tensors, but we might want to apply
        # additional transforms (e.g. normalization)
        if self.transform:
            images = [self.transform(img) if not isinstance(img, torch.Tensor) else img for img in images]

        if self.target_transform:
            # Assuming labels is a tuple of (digit_labels, quadrant_labels, pair_indices)
            # and target_transform applies to digit_labels and quadrant_labels
            digit_labels, quadrant_labels, pair_indices = labels
            digit_labels = [self.target_transform(l) for l in digit_labels]
            quadrant_labels = [self.target_transform(l) for l in quadrant_labels]
            pair_indices = [self.target_transform(l) for l in pair_indices]
            labels = (digit_labels, quadrant_labels, pair_indices)
            # NOTE: pair_indices is not used in current experiments, so put it at the end

        return images, labels

    def __len__(self):
        return self.num_files


def _create_polymnist_dataset_from_subset(subset_data,
                                          savepath,
                                          backgroundimagepath,
                                          num_modalities,
                                          seed=42,
                                          repetitions=1,
                                          log_wandb=False
                                          ):
    """
    Create a polyMNIST dataset from a *subset* of MNIST data (e.g. 55k images),
    optionally repeating multiple times with different random seeds to produce a
    larger dataset overall. and log samples via wandb if desired.

    If log_wandb=True, we log:
      - Some dataset stats
      - A few sample images for each digit & modality
      - Quadrant stats once the images are created

    Args:
        subset_data (torch.utils.data.Subset): subset of MNIST or entire MNIST dataset.
            Must have .data and .targets, each can be indexed with `subset_data.indices`.
        savepath (str): folder to output results, each modality will go in "m0", "m1", ...
        backgroundimagepath (str): path to .jpg background images
        num_modalities (int): how many distinct modalities (subfolders)
        seed (int): base random seed
        repetitions (int): how many times to re-generate using the same subset for more images
        log_wandb (bool): if True, log some stats and sample images to wandb

    Example usage:
        _create_polymnist_dataset_from_subset(
            train_55k, "train_out", "my_bg/", 5, seed=100, repetitions=6
        )
    """
    # Gather the subset's underlying Tensors
    full_data = subset_data.dataset.data
    full_targets = subset_data.dataset.targets
    subset_indices = subset_data.indices  # which items in the full MNIST
    # Convert to CPU numpy if needed
    # (In new torchvision, data/targets might be already torch.Tensors on CPU)
    full_data_np = full_data[subset_indices] #.numpy()
    full_targets_np = full_targets[subset_indices] #.numpy()

    # Load background images
    background_filepaths = sorted(glob.glob(os.path.join(backgroundimagepath, "*.jpg")))
    if num_modalities > len(background_filepaths):
        raise ValueError("Need at least as many background images as modalities.")
    background_images = [Image.open(fp) for fp in background_filepaths]

    # Create folders: "savepath/m0", "savepath/m1", ...
    for m in range(num_modalities):
        unimodal_path = os.path.join(savepath, f"m{m}")
        os.makedirs(unimodal_path, exist_ok=True)

    # Count how many total items in this subset
    n_sub = len(subset_indices)
    print(f"Subset has {n_sub} MNIST images -> {savepath}")

    # If wandb logging is on, log a small summary
    if log_wandb:
        wandb.log({"subset_size": n_sub, "savepath": savepath})
  
    rep_interval = 0
    accumul_counter = 0

    for rep_i in range(repetitions):
        # Possibly set a new seed each repetition
        # => different random permutations, random crops, etc.
        seed_sel = seed + rep_i
        torch.manual_seed(seed_sel)
        np.random.seed(seed_sel)
        print(f"[Rep {rep_i+1}/{repetitions}] Using seed={seed_sel}")

        # counters
        log_counter = 0 # for logging progress
        accumul_counter = 0 # for counting total images saved in this rep

        # List to store the number of images saved for each digit
        digit_counts = []

        # For each digit
        for digit in range(10):
            # Collect indices of MNIST images for this digit
            ixs_digit = np.where(full_targets_np == digit)[0]
            # If none, skip
            if len(ixs_digit) == 0:
                continue

            # Store a few sample images to log
            sample_images_for_wandb = []

            # Shuffle them => random permutations for each modality
            # Do one random permutation *per modality*
            # so each modality sees the same set of digit images
            # but in a different order
            for m in range(num_modalities):
                ixs_perm = np.random.permutation(ixs_digit)
                for i, real_idx in enumerate(ixs_perm):
                    mnist_tensor = full_data_np[real_idx]

                    # create a PIL/cropped background, and quadrant label
                    new_img, quadrant = PolyMNISTDataset._add_background_image(
                        background_images[m],
                        mnist_tensor,
                        view=m,
                    )
                    # e.g. "m0/1234.5.2.png" => global=1234, digit=5, quadrant=2
                    filepath = os.path.join(savepath, f"m{m}", f"{rep_interval+i}.{digit}.{quadrant}.png")
                    save_image(new_img, filepath)

                    # Note: progress counter is approximate across modalities.
                    # log the progress
                    log_counter += 1
                    if log_counter % 5000 == 0:
                        print("Saved %d/%d images to %s" % (log_counter,
                            len(subset_data)*num_modalities, savepath))
                        
                    # If logging, optionally store a few images (say first, middle, last)
                    if log_wandb:
                        if i in (0, len(ixs_perm)//2, len(ixs_perm)-1):
                            # Convert new_img (Tensor) to a PIL for wandb
                            # or can directly log wandb.Image(new_img) (it will handle it)
                            sample_images_for_wandb.append(
                                wandb.Image(
                                    new_img,
                                    caption=f"Rep {rep_i+1}, modality={m}, idx={i}, digit={digit}, quadrant={quadrant}"
                                )
                            )
                # End of ixs_perm loop
            # End of num_modalities loop
            digit_counts.append(len(ixs_digit))
            accumul_counter += len(ixs_digit)
            print(f"In rep {rep_i+1}/{repetitions}, digit {digit} saved {len(ixs_digit)} images, and total {accumul_counter} images saved.")

            # If wandb logging, push the sample images we collected
            if log_wandb and sample_images_for_wandb:
                # Typically we log them once per digit, per repetition
                # E.g. "samples/digit_3_rep2"
                wandb.log({f"samples/digit_{digit}_rep_{rep_i+1}": sample_images_for_wandb})

        rep_interval += accumul_counter
        print(f"Finished rep {rep_i+1}, total images saved: {rep_interval}.")

    total_imgs = rep_interval * num_modalities
    print(f"Done generating polyMNIST: {rep_interval} total digit instances, "
          f"{total_imgs} images overall in {savepath}.")
    for i, count in enumerate(digit_counts):
        print(f"Digit {i} counts: {count}")
    # print("Total counts: ", sum(digit_counts))
    print("Digit counts list: ", digit_counts)

    # After the images have been created, we can initialize a PolyMNISTDataset to log quadrant stats.
    # This will read in all the newly created images in m0..mN.
    dataset_paths = [os.path.join(savepath, f"m{i}") for i in range(num_modalities)]
    dataset = PolyMNISTDataset(dataset_paths, transform=transforms.ToTensor())

    # Optional: call quadrant stats & sample logging if needed.
    # print("Logging quadrant-related statistics and images...")
    # dataset.log_quadrant_stats()
    # dataset.log_sample_images()



if __name__ == "__main__":
    """Example usage that splits the 60k training set into train+val,
       and also processes test. Then calls the new function above."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed.')
    parser.add_argument('--num-modalities', type=int, default=2)
    parser.add_argument('--backgroundimagepath', type=str, required=True)
    parser.add_argument('--savepath-train', type=str, required=True,
                        help="Where to write the expanded train dataset (folders m0..mN).")
    parser.add_argument('--savepath-val', type=str, required=True,
                        help="Where to write the val dataset.")
    parser.add_argument('--savepath-test', type=str, required=True,
                        help="Where to write the test dataset.")
    parser.add_argument('--repetitions', type=int, default=6,
                        help="How many times to replicate the train subset for a bigger dataset.")
    parser.add_argument('--train-split-size', type=int, default=55000,
                        help="Number of MNIST train images to use for train. (the rest goes to val)")
    parser.add_argument('--wandb-project', type=str, default="PolyMNIST_DatasetCreation",
                        help="WandB project name")
    parser.add_argument('--wandb-runname', type=str, default="dataset_creation_run",
                        help="WandB run name")
    parser.add_argument('--wandb-off', action='store_true',
                        help="Disable wandb logging if set.")
    args = parser.parse_args()

    print("\nARGS:\n", args)

    # If not turning off wandb, init
    if not args.wandb_off:
        wandb.init(project=args.wandb_project, name=args.wandb_runname, config=args)

    # 1) Load the standard MNIST train set (60k)
    full_mnist_train = datasets.MNIST(
        root="mnist_data", # Note: use argparse/env override if a different root is needed.
        train=True,
        download=True,
        transform=None
    )
    # Make subsets for train & val
    indices = list(range(60000))
    # E.g. use 55k for train, 5k for val
    train_indices = indices[:args.train_split_size]
    val_indices   = indices[args.train_split_size:]
    train_sub = Subset(full_mnist_train, train_indices)
    val_sub   = Subset(full_mnist_train, val_indices)

    # 2) Generate the polyMNIST train set, with expansions
    _create_polymnist_dataset_from_subset(
        subset_data=train_sub,
        savepath=args.savepath_train,
        backgroundimagepath=args.backgroundimagepath,
        num_modalities=args.num_modalities,
        seed=args.seed,
        repetitions=args.repetitions,
        log_wandb=(not args.wandb_off)
    )

    # 3) Generate the polyMNIST val set (usually do not repeat)
    _create_polymnist_dataset_from_subset(
        subset_data=val_sub,
        savepath=args.savepath_val,
        backgroundimagepath=args.backgroundimagepath,
        num_modalities=args.num_modalities,
        seed=args.seed + 99,  # optional different seed
        repetitions=1,
        log_wandb=(not args.wandb_off)
    )

    # 4) Load the standard MNIST test set
    mnist_test = datasets.MNIST(
        root="mnist_data",
        train=False,
        download=True,
        transform=None
    )
    # you can use the entire 10k or some subset
    test_indices = list(range(len(mnist_test)))
    test_sub = Subset(mnist_test, test_indices)

    # 5) Generate the polyMNIST test set
    _create_polymnist_dataset_from_subset(
        subset_data=test_sub,
        savepath=args.savepath_test,
        backgroundimagepath=args.backgroundimagepath,
        num_modalities=args.num_modalities,
        seed=args.seed + 999,
        repetitions=1,  # usually no expansions for test
        log_wandb=(not args.wandb_off)
    )

    print("All done!")
    if not args.wandb_off:
        wandb.finish()
