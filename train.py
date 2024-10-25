import os
import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from typing import List, Union

from model import SiameseNetwork
from dataset import SiameseDataset
from loss import ContrastiveLossCosine
from utility import set_random_seed
from config import *

    
def train_siamese(model, dataloader, criterion, optimizer, device):
    """Run train loop."""
    model.train()
    total_loss = 0.0
    
    for img1, img2, labels in tqdm(dataloader, desc="TRAIN Phase", leave=False):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        similarity_scores = model(img1, img2)
        loss = criterion(similarity_scores, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_siamese(model, dataloader, criterion, device):
    """Run validation loop."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="VAL Phase", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            similarity_scores = model(img1, img2)
            loss = criterion(similarity_scores, labels)
            total_loss += loss.item()
            
            # Calculate accuracy by thresholding similarity score
            predictions = (similarity_scores > VALIDATION_SIMILARITY_THRESHOLD).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def get_all_image_labels(data_path: str, class_names: Union[List[str], None] = None):
    if class_names is None:
        class_names = os.listdir(data_path)
    else:
        class_names = class_names.split(',')

    all_image_paths = []
    all_labels = []
        
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        img_paths = [
            os.path.join(class_path, img) for img in os.listdir(class_path)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
        all_image_paths.extend(img_paths)
        all_labels.extend([class_idx] * len(img_paths))
    return np.array(all_image_paths), np.array(all_labels)


def main(args):    
    prefix = args.data_path.replace("/", "_")
    checkpoint_path = os.path.join(f"weights", prefix, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.use_wandb:
        wandb.login()
        experiment = wandb.init(project=f'SiameseSimilarityTraining', resume='allow')
        wandb_url = wandb.run.get_url()
        with open(os.path.join(checkpoint_path, "wandb_url.txt"), "w") as f:
            f.write(wandb_url)        

    # Set random seeds for produciblity
    set_random_seed(seed=SEED, deterministic=True)

    # Read all image-labels
    all_image_paths, all_labels = get_all_image_labels(args.data_path, args.class_names)

    # Stratified K-fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_image_paths, all_labels)):
        fold += 1
                
        print("*"*30)
        print(f"Fold {fold}/{NUM_FOLDS}".center(30))
        print("*"*30)

        train_subset = SiameseDataset(
            image_paths=all_image_paths[train_idx].tolist(), labels=all_labels[train_idx].tolist(), transform=TRANSFORM
        )

        val_subset = SiameseDataset(
            image_paths=all_image_paths[val_idx].tolist(), labels=all_labels[val_idx].tolist(), transform=BASIC_TRANSFORM
        )

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = SiameseNetwork(model_name=MODEL_BACKBONE, pretrained=True).to(DEVICE)
        criterion = ContrastiveLossCosine(margin=CONTRASIVE_MARGIN)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0        
        for epoch in range(NUM_EPOCHS):
            train_loss = train_siamese(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate_siamese(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if args.use_wandb:
                experiment.log({
                    'Train Loss': train_loss,
                    'Val Loss': val_loss,
                    'Val Acc': val_acc,
                    'Epoch': epoch+1,
                })
            
            # Save the best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(checkpoint_path, f"{prefix}_best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model for <fold {fold}> with accuracy {val_acc:.4f}")
                with open(save_path.replace(".pth", ".txt"), "w") as f:
                    f.write(f"{best_val_acc}\n")

            if best_val_acc == 1.0:
                break


def get_args():
    parser = argparse.ArgumentParser(description='Train a Siamese similarity model.')
    parser.add_argument('-d', '--data-path', required=True, help='Path to the root data folder.')
    parser.add_argument('-c', '--class_names', default=None, type=str, required=False, help='[Optional] Comma-separated class names. Class indices will follow the order.')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='Use WandB for web logging?')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
