import os
import shutil
import argparse
import torch
import random
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import SiameseNetwork
from config import *


class ImageDataset(Dataset):
    """Custom Dataset to load images and apply transformations."""
    def __init__(self, image_paths: list, transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, img_path


def pairwise_dissimilarity_siamese(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor):
    """Compute dissimilarity between two images using the Siamese network."""
    model.eval()
    with torch.no_grad():
        # Compute similarity using the Siamese model (assuming it returns cosine similarity)
        similarity = model(img1, img2)
        dissimilarity = 1 - similarity  # Cosine dissimilarity = 1 - similarity
    return dissimilarity


def compute_pairwise_dissimilarity(model: torch.nn.Module, images: list):
    """Compute pairwise dissimilarities between all images in the current set."""
    dissimilarity_matrix = np.zeros((len(images), len(images)))  # To store dissimilarities

    for i in tqdm(range(len(images)), desc="Computing pairwise dissimilarities"):
        for j in range(i + 1, len(images)):
            img1 = images[i].unsqueeze(0).to(DEVICE)
            img2 = images[j].unsqueeze(0).to(DEVICE)
            dissimilarity = pairwise_dissimilarity_siamese(model, img1, img2)
            dissimilarity_matrix[i, j] = dissimilarity.item()
            dissimilarity_matrix[j, i] = dissimilarity.item()

    return dissimilarity_matrix


def select_top_dissimilar_images(dissimilarity_matrix, topk):
    """Select K most dissimilar images using the dissimilarity matrix."""
    dissimilarity_scores = dissimilarity_matrix.sum(axis=1)  # Sum dissimilarities for each image
    top_indices = np.argsort(dissimilarity_scores)[-topk:]  # Select top K dissimilar images
    return top_indices


def iterative_dissimilarity_selection(model: torch.nn.Module, image_paths: list, num_refs: int, topk: int, batch_size: int):
    """Iteratively select K dissimilar images from subsets of N images."""
    selected_images = []

    while len(image_paths) > 0:
        # Select next batch of num_refs images
        current_batch_paths = image_paths[:num_refs]
        image_paths = image_paths[num_refs:]  # Remove these images from the remaining set

        # Create dataset and dataloader for batch processing
        dataset = ImageDataset(current_batch_paths, transform=BASIC_TRANSFORM)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        # Load images into memory and keep track of their paths
        images, batch_image_paths = [], []
        for img_batch, img_paths in dataloader:
            images.append(img_batch)
            batch_image_paths.extend(img_paths)

        images = torch.cat(images, dim=0)

        # Compute pairwise dissimilarity for this batch
        dissimilarity_matrix = compute_pairwise_dissimilarity(model, images)

        # Select the top K dissimilar images from this batch
        top_indices = select_top_dissimilar_images(dissimilarity_matrix, topk)
        selected_images += [batch_image_paths[i] for i in top_indices]

    return selected_images


def iterative_selection_and_refinement(model: torch.nn.Module, image_paths: list, num_refs: int, topk: int, batch_size: int):
    """Iteratively select N dissimilar images with refinement."""
    while len(image_paths) > num_refs:
        image_paths = iterative_dissimilarity_selection(
            model, image_paths, num_refs, topk, batch_size
        )

    return image_paths


def save_reference_images(reference_images, output_dir):
    """Save selected reference images into the given output directory, maintaining the class folder structure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name, image_paths in reference_images.items():
        # Create class sub-directory in the output directory if it doesn't exist
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for image_path in image_paths:
            # Copy the image to the new output directory
            output_image_path = os.path.join(class_output_dir, os.path.basename(image_path))
            shutil.copy(image_path, output_image_path)
            print(f"Saved {image_path} to {output_image_path}")


def find_highest_dissimilar_images_per_class(
    model: torch.nn.Module,
    data_root: str,    
    num_refs: int,
    topk: int,
    batch_size: int,
):
    """Find N images from each class that are the most dissimilar (within the class)."""    
    reference_images = {}

    for class_name in CLASS_NAMES:        
        class_dir = os.path.join(data_root, class_name)
        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".png")]
        print(f"Processing class {class_name} with {len(image_paths)}...")

        num_refs_class = min([num_refs, len(image_paths)])

        selected_images = iterative_selection_and_refinement(
            model,
            image_paths,            
            num_refs_class,
            topk,
            batch_size
        )

        reference_images[class_name] = selected_images

    return reference_images


def get_args():
    parser = argparse.ArgumentParser(description='Select the most diverse reference images for Siamese similarity model.')
    parser.add_argument('-d', '--data-path', required=True, help='Path to the root data folder.')
    parser.add_argument('-m', '--model-path', required=True, help='Path to the pre-trained Siamese model.')
    parser.add_argument('-o', '--output-path', required=True, help='Path to save reference images.')
    parser.add_argument('-n', '--num_refs', default=100, required=False, help='Number of reference images per class.')
    parser.add_argument('-k', '--topk', default=100, required=False, help='Number of top dissimilar images per batch.')
    parser.add_argument('-b', '--batch-size', default=64, required=False, help='Batch size for DataLoader.')
    return parser.parse_args()


def main(args):
    random.seed(42)

    # Load model
    model_path = args.model_path
    model = SiameseNetwork(model_name=MODEL_BACKBONE, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    # Find the highest dissimlar images per class
    reference_images = find_highest_dissimilar_images_per_class(model, args.data_path, args.num_refs, args.topk, args.batch_size)

    # Save the selected reference images to the output directory
    save_reference_images(reference_images, args.output_path)

    # Output the saved reference images for each class
    for class_name, image_paths in reference_images.items():
        print(f"Most dissimilar images from {class_name} saved to {args.output_path}/{class_name} | {len(image_paths)}")


if __name__ == "__main__":
    args = get_args()
    main(args)
