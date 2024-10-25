import os
import random
import torch

from collections import defaultdict
from typing import List, Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SiameseDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Union[transforms.Compose, None] = None
    ):
        self.image_paths = image_paths
        self.labels = labels                
        self.transform = transform

        self.class_to_images = defaultdict(list)
        for label, image_path in zip(labels, image_paths):
            self.class_to_images[label].append(image_path)

        self.num_classes = len(self.class_to_images.keys())
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Randomly sample a pair of images for training
        img_path1 = self.image_paths[idx]
        label1 = self.labels[idx]

        # Select the second image with 50% chance to get a similar pair
        same_class = random.random() < 0.5 

        if same_class:
            img_path2 = random.choice(self.class_to_images[label1])
            label = 1.0  # Label 1 for same class (similar)
        else:
            label2 = random.choice([lbl for lbl in range(self.num_classes) if lbl != label1])
            img_path2 = random.choice(self.class_to_images[label2])
            label = 0.0  # Label 0 for different classes (dissimilar)

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
