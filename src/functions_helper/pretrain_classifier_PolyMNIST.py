# This script trains a classifier on the PolyMNIST dataset for each modality.
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_PolyMNIST_quadrant import PolyMNISTDataset, PolyMNISTDataset_pt
from utils import DigitClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
RESNET_S0 = 8


# Load PolyMNIST dataset
def get_dataloaders(datadir, batch_size=128): # default batch size is 128
    """
    Returns train and test DataLoaders for all 5 modalities
    NOTE: _dataset get item returns images and labels,
    where images is a list of tensors (one for each modality),
    and labels is a tuple of two lists: labels[0] is the digit label, labels[1] is the quadrant label.
    """
    transform = transforms.ToTensor()
    
    train_paths = [f"{datadir}/PolyMNIST/train/m{i}" for i in range(5)]
    valid_paths = [f"{datadir}/PolyMNIST/val/m{i}" for i in range(5)]
    test_paths = [f"{datadir}/PolyMNIST/test/m{i}" for i in range(5)]

    # This uses the final shared/private label format from PolyMNISTDataset.
    train_datasets = [PolyMNISTDataset([path], transform=transform) for path in train_paths]
    valid_datasets = [PolyMNISTDataset([path], transform=transform) for path in valid_paths]
    test_datasets = [PolyMNISTDataset([path], transform=transform) for path in test_paths]
    
    train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8) for dataset in train_datasets]
    valid_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8) for dataset in valid_datasets]
    test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8) for dataset in test_datasets]

    return train_loaders, valid_loaders, test_loaders


# Training function
def train_classifier(modality_idx, train_loader, valid_loader,
                    condition_type='shared',
                    num_epochs=10
                    ):
    model = DigitClassifier(input_size=IMG_SIZE, res_s0=RESNET_S0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            # Select digit or quadrant labels based on condition_type
            if condition_type is not None:
                if condition_type == 'shared':
                    labels = labels[0]  # Use digit labels
                elif condition_type == 'private':
                    labels = labels[1]
                else:
                    raise ValueError("Invalid condition_type. Choose 'shared' or 'private'.")

            # images is a list with only one modality
            # images: list, len(images) = 1, images[0]: torch.Size([128, 3, 28, 28])
            if isinstance(images, list):
                images = images[0]  # Extract the tensor
            if isinstance(labels, list):
                labels = labels[0]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Modality {modality_idx}, Epoch {epoch+1}/{num_epochs}: Loss = {running_loss/len(train_loader):.4f}, Acc = {correct/total:.4f}")

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:

            # Select digit or quadrant labels based on condition_type
            if condition_type is not None:
                if condition_type == 'shared':
                    labels = labels[0]  # Use digit labels
                elif condition_type == 'private':
                    labels = labels[1]
                else:
                    raise ValueError("Invalid condition_type. Choose 'shared' or 'private'.")
            
            # Unpack if images is a list (use only one modality)
            if isinstance(images, list):
                images = images[0]  # Extract the tensor
            if isinstance(labels, list):
                labels = labels[0]

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Final validating Accuracy for Modality {modality_idx}: {correct/total:.4f}")

    return model


# Main function
def main():

    # Argument to parse
    parser = argparse.ArgumentParser(description='MMVAEplus PolyMNIST Classifier Training')
    parser.add_argument('--num_modalities', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--datadir', type=str, default="data/polymnist_translated_6x_scalep8_nocc", help='Path to the PolyMNIST dataset')
    parser.add_argument('--save_dir', type=str, default="data/polymnist_translated_6x_scalep8_nocc", help='Directory to save the trained classifiers')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--condition_type', type=str, default='shared', help='Type of label to use, e.g: digit or quadrant')

    # Args
    args = parser.parse_args()
    
    print(f"Training {args.condition_type} classifier!")
    print(f"Number of epochs: {args.num_epochs}")
    print("Using classifier: DigitClassifier")
    print(f"Training classifiers on PolyMNIST dataset at {args.datadir}")
    print(f"Saving trained classifiers to {args.save_dir}")
    
    save_dir = args.save_dir + '/DigitClassifier/trained_clfs_polyMNIST'
    os.makedirs(save_dir, exist_ok=True)
    train_loaders, valid_loaders, test_loaders = get_dataloaders(args.datadir, args.batch_size) # NOTE: batch size is 128 by default, even though it is set to 512 in the args (Mar12.2025)

    for i in range(args.num_modalities):
        print(f"\nTraining classifier for modality {i}...")
        model = train_classifier(i, train_loaders[i], valid_loaders[i], args.condition_type, args.num_epochs)
        if args.condition_type is None or args.condition_type == 'shared':
            save_path = f"{save_dir}/pretrained_img_to_digit_clf_m{i}"
        elif args.condition_type == 'private':
            save_path = f"{save_dir}/pretrained_img_to_quadrant_clf_m{i}"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {args.condition_type} classifier for modality {i} to {save_path}\n")

    print("All classifiers trained and saved successfully!")


if __name__ == "__main__":
    main()
