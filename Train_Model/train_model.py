import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from utils.datasetloader.dataset import SegmentationDataset
from utils.model.unet import AttentionUNet
from utils.training.trainer import train_model
from utils.stack_splitter import process_all_images, process_all_masks
from utils.cropping_image import crop_images
from utils.split_dataset import split_dataset
from utils.clear_images import clear_images_in_folder
from utils.mean_SNR import calculate_mean_snr
from utils.generate_mask import process_images_across_folds
from utils.mask_compare import evaluate_and_save_best_model

# Model Training Settings
batch_size = 4
num_epochs = 100
num_folds = 5
learning_rate = 0.00001

# Patch Settings
crop_size = 128
overlap = 64

# Preprocessing and splitting the dataset
process_all_images(input_folder="Train_Model/TRAINING_IMAGES", output_folder="utils/temp_files/Images")
process_all_masks(input_folder="Train_Model/TRAINING_MASKS", output_folder="utils/temp_files/Masks")
split_dataset(
    images_folder = "utils/temp_files/Images",
    masks_folder = "utils/temp_files/Masks",
    train_output_folder = "utils/temp_files/Model_Training/Training_Dataset",
    test_output_folder = "utils/temp_files/Model_Training/Test_Dataset",
    split_ratio = 0.9
)
crop_images(
    image_folder="utils/temp_files/Model_Training/Training_Dataset/Images",
    mask_folder="utils/temp_files/Model_Training/Training_Dataset/Masks",
    output_images_folder="utils/temp_files/Model_Training/Training_Dataset/Training_Images",
    output_masks_folder="utils/temp_files/Model_Training/Training_Dataset/Training_Masks",
    crop_size = crop_size,
    overlap = overlap
)

# Ask user for the model name, avoid duplicates
while True:
    model_name = input("\n[INPUT] Name of the model: ").strip()
    model_folder = "Use_Model/saved_models"

    # Check if any file already starts with this model_name
    existing_models = [
        f for f in os.listdir(model_folder)
        if f.startswith(model_name + "_")
        or f == f"{model_name}.pth"
    ]

    if existing_models:
        print(f"[WARNING] A model with the name '{model_name}' already exists!")
        print("Existing files:", existing_models)
        print("Please enter a different name.")
    else:
        break

# Define paths
images_path = os.path.join("utils/temp_files/Model_Training/Training_Dataset/Training_Images")
masks_path = os.path.join("utils/temp_files/Model_Training/Training_Dataset/Training_Masks")

# Data Transformations
transform = transforms.Compose([transforms.ToTensor()])

# Initialize dataset
dataset = SegmentationDataset(images_path, masks_path, transform=transform)

# Dynamically determine the number of input channels based on the dataset
input_channels = dataset.get_input_channels()

# K-fold Cross-validation setup
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize model function
model_fn = lambda: AttentionUNet(img_ch=input_channels)

# Call training function and pass the model name along with early_stopping
train_model(model_fn , dataset, kf, batch_size, num_epochs, learning_rate, model_name=model_name)

# After training, process the masks, evaluate the best model, and calculate the mean SNR
process_images_across_folds()
evaluate_and_save_best_model()

# Clear any temporary images
clear_images_in_folder("utils/temp_files", "Train_Model")