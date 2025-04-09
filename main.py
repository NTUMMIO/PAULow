import os
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from utils.dataset import SegmentationDataset
from model.unet import AttentionUNet
from training.trainer import train_model
from utils.stack_splitter import split_all_images
from utils.stack_splitter import split_all_masks
from utils.cropping_image import crop_images
from utils.split_dataset import split_dataset
from utils.clear_images import clear_images_in_folder
from training.trainer import EarlyStopping  

# Preprocessing and splitting the dataset
split_all_images()
split_all_masks()
split_dataset()
crop_images()

# Ask user for the model name
model_name = input("Name of the model: ")

# Define paths
data_path = "Model_Training"
dataset_path = os.path.join(data_path, "Training_Dataset")
images_path = os.path.join(dataset_path, "Training_Images")
masks_path = os.path.join(dataset_path, "Training_Masks")

# Hyperparameters
batch_size = 4
num_epochs = 100
num_folds = 5
learning_rate = 0.000001

# Data Transformations
transform = transforms.Compose([transforms.ToTensor()])

# Initialize dataset
dataset = SegmentationDataset(images_path, masks_path, transform=transform)

# K-fold Cross-validation setup
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize the model
model = AttentionUNet()

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize EarlyStopping (add your patience, delta, and threshold values here)
early_stopping = EarlyStopping(patience=10, delta=0.01, threshold=0.001)

# Call training function and pass model name along with early_stopping
train_model(model, dataset, kf, batch_size, num_epochs, early_stopping, model_name=model_name)