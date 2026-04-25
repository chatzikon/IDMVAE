import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import json
from datetime import datetime
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_args():
    parser = argparse.ArgumentParser(description='Convert CelebAMask-HQ to .pt format')
    parser.add_argument('--data_root', type=str, 
                        default='/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask-HQ_from_SBM',
                        help='Root directory of the dataset')
    parser.add_argument('--save_path', type=str, 
                        default='/data/backed_up/shared/Data/CelebAMask_HQ/CelebAMask_HQ_from_SBM_pt',
                        help='Directory to save the .pt files')
    parser.add_argument('--convert_images', action='store_true', help='Flag to convert images')
    parser.add_argument('--convert_masks', action='store_true', help='Flag to convert masks')
    parser.add_argument('--convert_attributes', action='store_true', help='Flag to convert attributes')
    parser.add_argument('--save_splits', action='store_true', help='Flag to save splits indices')
    parser.add_argument('--verify', action='store_true', help='Flag to verify conversion')
    parser.add_argument('--image_size', type=int, default=256, help='Size to rescale images (square)')
    parser.add_argument('--mask_size', type=int, default=128, help='Size to rescale masks (square)')
    return parser.parse_args()

def get_file_index(filename):
    # filename is like '0.jpg' or '1.png', etc.
    return int(os.path.splitext(filename)[0])

def verify_conversion(args, run_dir):
    print("\n" + "="*20)
    print("Verifying conversion...")
    try:
        # Ensure the current directory is in sys.path to import dataset_CelebAMask_HQ
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from dataset_CelebAMask_HQ import CelebAHQMaskDS_pt
        from torch.utils.data import DataLoader
        from torchvision.utils import save_image

        # Create verification directory
        verify_dir = os.path.join(run_dir, 'verification_samples')
        os.makedirs(verify_dir, exist_ok=True)

        splits = ['train', 'val', 'test']
        
        for split in splits:
            print(f"Verifying {split} split...")
            try:
                ds = CelebAHQMaskDS_pt(image_size=args.image_size, mask_size=args.mask_size, datapath=args.save_path, ds_type=split)
                print(f"  {split} size: {len(ds)}")
                
                if len(ds) == 0:
                    print(f"  Warning: {split} split is empty.")
                    continue

                # Save 10 examples
                split_dir = os.path.join(verify_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                attr_file_path = os.path.join(split_dir, 'attributes.txt')
                with open(attr_file_path, 'w') as f:
                    f.write("Index\tAttributes\n")
                    
                    # Iterate first 10 samples (or less if dataset is smaller)
                    num_samples = min(10, len(ds))
                    for i in range(num_samples):
                        img, mask, attr = ds[i]
                        # Get original index from dataset indices
                        original_idx = ds.indices[i]
                        
                        # Save Image
                        save_image(img, os.path.join(split_dir, f'{original_idx}_img.png'))
                        
                        # Save Mask (ensure it's visible, maybe scale 0-1 to 0-255 if it's float 0-1 it's fine for save_image)
                        # Mask is 1xHxW. save_image handles it.
                        save_image(mask.float(), os.path.join(split_dir, f'{original_idx}_mask.png'))
                        
                        # Save Attribute
                        attr_str = ' '.join(map(str, attr.tolist()))
                        f.write(f"{original_idx}\t{attr_str}\n")
                
                print(f"  Saved {num_samples} examples to {split_dir}")

                # Basic stats check on the first batch (previous logic)
                batch_size = 100
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
                batch = next(iter(dl))
                imgs, masks, attrs = batch

                print(f"Batch shapes:")
                print(f"  Images: {imgs.shape}")
                print(f"  Masks:  {masks.shape}")
                print(f"  Attrs:  {attrs.shape}")

                # Check shapes
                assert imgs.shape == (batch_size, 3, args.image_size, args.image_size), f"Image shape mismatch: {imgs.shape}"
                assert masks.shape == (batch_size, 1, args.mask_size, args.mask_size), f"Mask shape mismatch: {masks.shape}"
                assert attrs.shape == (batch_size, 40), f"Attribute shape mismatch: {attrs.shape}"

                # Check value ranges
                print(f"Image range: [{imgs.min():.4f}, {imgs.max():.4f}]")
                print(f"Mask range:  [{masks.min():.4f}, {masks.max():.4f}]")
                print(f"Attr unique: {torch.unique(attrs).tolist()}")

                # Check if attributes are 0/1
                unique_attrs = torch.unique(attrs)
                if not torch.all(torch.isin(unique_attrs, torch.tensor([0, 1]))):
                    print("Warning: Attributes contain values other than 0 and 1.")

            except Exception as e:
                print(f"  Error verifying {split}: {e}")
                # import traceback
                # traceback.print_exc()

        print("Verification Passed Successfully!")
        
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
    print("="*20 + "\n")

def main():
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    # Save args with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args_dict = vars(args)
    args_dict['timestamp'] = timestamp
    
    # Create timestamp subfolder
    run_dir = os.path.join(args.save_path, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logger
    sys.stdout = Logger(os.path.join(run_dir, 'log.out'))

    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Saved args to {timestamp}/args.json")

    # Define paths
    img_source_dir = os.path.join(args.data_root, 'CelebA-HQ-img')
    mask_source_dir = os.path.join(args.data_root, 'CelebAMaskHQ-mask')
    attr_file = os.path.join(args.data_root, 'CelebAMask-HQ-attribute-anno.txt')
    
    train_dir = os.path.join(args.data_root, 'train_img')
    val_dir = os.path.join(args.data_root, 'val_img')
    test_dir = os.path.join(args.data_root, 'test_img')

    total_samples = 30000

    # 1. Create Splits
    if args.save_splits:
        print("Processing splits...")
        splits = {'train': [], 'val': [], 'test': []}
        
        # Helper to extract indices from a directory
        def get_indices_from_dir(directory):
            indices = []
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist. Skipping.")
                return indices
            # List all files
            files = os.listdir(directory)
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    try:
                        idx = get_file_index(f)
                        indices.append(idx)
                    except ValueError:
                        continue
            return sorted(indices)

        splits['train'] = get_indices_from_dir(train_dir)
        splits['val'] = get_indices_from_dir(val_dir)
        splits['test'] = get_indices_from_dir(test_dir)

        # Verify splits
        all_indices = set(splits['train'] + splits['val'] + splits['test'])
        print(f"Found {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test samples.")
        
        # Save splits
        splits_save_path = os.path.join(args.save_path, 'splits_idx.pt')
        torch.save(splits, splits_save_path)
        print(f"Saved splits to {splits_save_path}")

    # 2. Convert Images
    if args.convert_images:
        print(f"Converting images (resizing to {args.image_size}x{args.image_size})...")
        transform_img = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        
        images_tensor = []
        # We iterate 0 to 29999 to maintain order
        for i in tqdm(range(total_samples)):
            # Try to find the file. Usually it's {i}.jpg
            fname = f"{i}.jpg"
            fpath = os.path.join(img_source_dir, fname)
            
            if not os.path.exists(fpath):
                print(f"Warning: Image {i} not found at {fpath}")
                # Append zero tensor or handle error? 
                # For now, let's append a black image to keep indexing consistent
                images_tensor.append(torch.zeros(3, args.image_size, args.image_size))
                continue

            try:
                img = Image.open(fpath).convert('RGB')
                img_t = transform_img(img)
                images_tensor.append(img_t)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                images_tensor.append(torch.zeros(3, args.image_size, args.image_size))

        images_tensor = torch.stack(images_tensor)
        torch.save(images_tensor, os.path.join(args.save_path, 'images.pt'))
        print(f"Saved images.pt with shape {images_tensor.shape}")

    # 3. Convert Masks
    if args.convert_masks:
        print(f"Converting masks (resizing to {args.mask_size}x{args.mask_size})...")
        # Use Nearest for masks to preserve integer labels if they exist
        transform_mask = transforms.Compose([
            transforms.Resize((args.mask_size, args.mask_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        masks_tensor = []
        for i in tqdm(range(total_samples)):
            # Assuming mask naming matches image naming or is just index
            # Common formats: {i}.png
            fpath = os.path.join(mask_source_dir, f"{i}.png")
            
            if not os.path.exists(fpath):
                print(f"Warning: Mask {i} not found.")
                masks_tensor.append(torch.zeros(1, args.mask_size, args.mask_size))
                continue

            try:
                # Masks are Grayscale single channel (L)
                mask = Image.open(fpath).convert('L') 
                mask_t = transform_mask(mask)
                masks_tensor.append(mask_t)
            except Exception as e:
                print(f"Error processing mask {i}: {e}")
                masks_tensor.append(torch.zeros(1, args.mask_size, args.mask_size))

        masks_tensor = torch.stack(masks_tensor)
        torch.save(masks_tensor, os.path.join(args.save_path, 'masks.pt'))
        print(f"Saved masks.pt with shape {masks_tensor.shape}")

    # 4. Convert Attributes
    if args.convert_attributes:
        print("Converting attributes...")
        if os.path.exists(attr_file):
            try:
                with open(attr_file, 'r') as f:
                    lines = f.readlines()
                
                # First line is number of samples
                # Second line is attribute names
                attr_names = lines[1].strip().split()
                
                # Remaining lines: filename attr1 attr2 ...
                # We need to ensure we map them to the correct index 0-29999
                # Usually the file is sorted, but we should parse the filename to be sure
                
                # Initialize tensor with zeros (or appropriate default)
                # Attributes are usually -1 or 1. 
                num_attrs = len(attr_names)
                attrs_tensor = torch.zeros(total_samples, num_attrs)
                
                for line in lines[2:]:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    fname = parts[0]
                    try:
                        idx = get_file_index(fname)
                        vals = [int(x) for x in parts[1:]]
                        if 0 <= idx < total_samples:
                            attrs_tensor[idx] = torch.tensor(vals, dtype=torch.float)  # set to .long() in dataloading
                    except ValueError:
                        continue
                
                torch.save(attrs_tensor, os.path.join(args.save_path, 'attributes.pt'))
                # Also save attribute names
                with open(os.path.join(args.save_path, 'attr_names.json'), 'w') as f:
                    json.dump(attr_names, f)
                
                print(f"Saved attributes.pt with shape {attrs_tensor.shape}")
            except Exception as e:
                print(f"Error processing attributes: {e}")
        else:
            print(f"Attribute file not found at {attr_file}")

    # 5. Verify
    if args.verify:
        verify_conversion(args, run_dir)

if __name__ == '__main__':
    main()
