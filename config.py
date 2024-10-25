import torch
import numpy as np

from torchvision import transforms

SEED = 42
RNG = np.random.default_rng(SEED)

MODEL_BACKBONE = 'resnet18'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_FOLDS = 4

# A contrasive margin of 0.5 is considered to be optimum
CONTRASIVE_MARGIN = 0.5

# This is only for validation phase which is supposed to be optimum
VALIDATION_SIMILARITY_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Essential data transformation
BASIC_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training data augmentation, if you need you can add some augmentation here
TRANSFORM = transforms.Compose([BASIC_TRANSFORM])
