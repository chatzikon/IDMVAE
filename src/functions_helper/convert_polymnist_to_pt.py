import torch
import os
import argparse
import wandb
from torchvision import transforms
from torchvision.utils import make_grid
from dataset_PolyMNIST_quadrant import PolyMNISTDataset, PolyMNISTDataset_pt

def convert_and_save(datadir, outputdir, split, num_modalities=5):
    """
    Converts a split of the PolyMNIST dataset to .pt files.
    """
    print(f"--- Starting Conversion for split: {split} ---")

    unim_datapaths = [os.path.join(datadir, split, "m" + str(i)) for i in range(num_modalities)]

    # Check if the data paths exist
    if not all(os.path.exists(p) for p in unim_datapaths):
        print(f"Skipping conversion for split {split}: not all modality paths exist.")
        return

    tx = transforms.ToTensor()
    dataset = PolyMNISTDataset(unim_datapaths, transform=tx)

    # Create output directory in "split" subfolder
    os.makedirs(os.path.join(outputdir, split), exist_ok=True)

    all_images = []
    all_labels = []
    for i in range(len(dataset)):
        images, labels = dataset[i]
        all_images.append(images)
        all_labels.append(labels)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples for split {split}.")

    # Save all images to a single file
    torch.save(all_images, os.path.join(outputdir, split, "images.pt"))
    # Save all labels to a single file
    torch.save(all_labels, os.path.join(outputdir, split, "labels.pt"))

    print(f"--- Finished Conversion for split: {split} ---")

def verify_conversion(original_datadir, pt_datadir, split, num_modalities=5, num_samples_to_check=20):
    """
    Verifies the conversion of a split of the PolyMNIST dataset to .pt format.
    """
    print(f"\n--- Starting Verification for split: {split} ---")

    # Initialize original dataset
    original_unim_datapaths = [os.path.join(original_datadir, split, "m" + str(i)) for i in range(num_modalities)]
    if not all(os.path.exists(p) for p in original_unim_datapaths):
        print(f"Skipping verification for split {split}: not all original modality paths exist.")
        return

    tx = transforms.ToTensor()
    original_dataset = PolyMNISTDataset(original_unim_datapaths, transform=tx)

    # Initialize .pt dataset
    pt_split_dir = os.path.join(pt_datadir, split)
    if not os.path.exists(pt_split_dir):
        print(f"Skipping verification for split {split}: .pt directory does not exist.")
        return

    pt_dataset = PolyMNISTDataset_pt(pt_split_dir)

    if len(original_dataset) != len(pt_dataset):
        print(f"Error: Length mismatch for split {split}!")
        print(f"  Original dataset length: {len(original_dataset)}")
        print(f"  .pt dataset length:     {len(pt_dataset)}")
        return

    print(f"Dataset ({split}) lengths match: {len(original_dataset)}")

    for i in range(num_samples_to_check):
        original_images, original_labels = original_dataset[i]
        pt_images, pt_labels = pt_dataset[i]

        if original_labels != pt_labels:
            print(f"Error: Label mismatch at index {i} for split {split}!")
            print(f"  Original labels: {original_labels}")
            print(f"  .pt labels:      {pt_labels}")
            return

        if len(original_images) != len(pt_images):
            print(f"Error: Number of modalities mismatch at index {i} for split {split}!")
            return

        for m in range(len(original_images)):
            if not torch.equal(original_images[m], pt_images[m]):
                print(f"Error: Image mismatch at index {i}, modality {m} for split {split}!")
                return
    
    print(f"Successfully verified {num_samples_to_check} samples. Data and labels are consistent.")

    print(f"Logging samples for split {split} to wandb...")
    # wandb.init(project="PolyMNIST_Conversion_Verification", name=f"verification_{note}", job_type="verification")

    for i in range(min(num_samples_to_check, 20)):  # Log up to 20 samples
        original_images, original_labels = original_dataset[i]
        pt_images, pt_labels = pt_dataset[i]

        log_dict = {}
        for m in range(num_modalities):
            # Extract labels for caption
            digit_lbls_orig, quadrant_lbls_orig, pair_idx_orig = original_labels
            digit_lbls_pt, quadrant_lbls_pt, pair_idx_pt = pt_labels
            caption_text_orig = f"Pair Idx: {pair_idx_orig[m]}, Digit: {digit_lbls_orig[m]}, Quadrant: {quadrant_lbls_orig[m]}"
            caption_text_pt = f"Pair Idx: {pair_idx_pt[m]}, Digit: {digit_lbls_pt[m]}, Quadrant: {quadrant_lbls_pt[m]}"

            # Create a combined image for side-by-side comparison
            comparison_image = torch.cat((original_images[m], pt_images[m]), dim=2)
            
            # Add modality-specific info to caption
            modality_caption = f"Modality {m} | Left: Original, Right: .pt\nLeft: {caption_text_orig}\nRight: {caption_text_pt}"

            log_dict[f"split_{split.replace('/', '_')}/sample_{i}/modality_{m}_comparison"] = wandb.Image(
                comparison_image,
                caption=modality_caption
            )
        wandb.log(log_dict)

    # Plot 5*10 (5 modalities * 10 pair unique digits) grid of original and/or .pt image for paper/poster visual inspection.
    print(f"Generating image grid for split {split} and Paper/Poster Visual Inspection...")

    digit_samples = {}
    for i in range(len(original_dataset)):
        images, labels = original_dataset[i]
        digit = labels[0][0]
        if digit not in digit_samples:
            digit_samples[digit] = images
        if len(digit_samples) == 10:
            break

    if len(digit_samples) < 10:
        print(f"Warning: Could not find all 10 unique digits in the dataset for split {split}.")

    sorted_samples = [digit_samples[d] for d in sorted(digit_samples.keys())]

    grid_images = []
    for m in range(num_modalities):
        for sample in sorted_samples:
            grid_images.append(sample[m])

    if grid_images:
        image_grid = make_grid(grid_images, nrow=10)
        wandb.log({f"Image_grids/split_{split.replace('/', '_')}": wandb.Image(image_grid, caption="5x10 grid of modalities x digits")})
        print(f"Image grid for split {split} logged to wandb.")

    # wandb.finish()
    # print("Finished logging to wandb.")
    print(f"--- Finished Verification for split: {split} ---\n")

def main():
    parser = argparse.ArgumentParser(description="Convert and/or verify PolyMNIST dataset from PNG to PT format.")
    parser.add_argument('--datadir', type=str, required=True, help='Root directory of the original PolyMNIST dataset (image format).')
    parser.add_argument('--outputdir', type=str, required=True, help='Directory to save/load the .pt files.')

    parser.add_argument('--wandb_run_name', type=str, default="verification_run", help='Name for the wandb run.')

    parser.add_argument('--convert', action='store_true', default=False, help='Run the conversion from images to .pt files.')
    parser.add_argument('--verify', action='store_true', default=False, help='Run the verification of the .pt files.')

    args = parser.parse_args()

    splits = ["train", "val", "test", "Subsample/train", "Subsample/debug_mini"]  # 

    if not args.convert and not args.verify:
        print("Please specify at least one action: --convert or --verify")
        return

    if args.verify:
        # Keep explicit login for environments where W&B auth is not preconfigured.
        wandb.login()
        wandb.init(project="PolyMNIST_Conversion_Verification", name=args.wandb_run_name, job_type="verification")

    for split in splits:
        if args.convert:
            convert_and_save(args.datadir, args.outputdir, split)
        
        if args.verify:
            verify_conversion(args.datadir, args.outputdir, split)

    if args.verify:
        wandb.finish()
        print("Finished logging to wandb.")

if __name__ == "__main__":
    main()
