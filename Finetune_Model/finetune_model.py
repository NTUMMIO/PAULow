import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from utils.datasetloader.dataset import SegmentationDataset
from utils.model.unet import AttentionUNet
from utils.training.trainer import DynamicLoss
from utils.clear_images import clear_images_in_folder

# Paths and Hyperparameters
images_dir = "utils/temp_files/Model_Training/Training_Dataset/Training_Images"
masks_dir = "utils/temp_files/Model_Training/Training_Dataset/Training_Masks"
model_dir = "Use_Model/saved_models"
epochs = 100
batch_size = 4
learning_rate = 1e-5

transform = transforms.Compose([transforms.ToTensor()])
dataset = SegmentationDataset(images_dir, masks_dir, transform=transform)

# -------------------- Utility Functions --------------------
def natural_sort_key(filename):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]

def save_frame(im, output_folder, base_name_prefix, counter, ext):
    """Save image as-is (no normalization)."""
    output_name = f"{base_name_prefix}{counter}{ext}"
    output_path = os.path.join(output_folder, output_name)
    im.save(output_path)
    return counter + 1

def process_image_file(image_path, output_folder, counter, is_mask=False):
    ext = os.path.splitext(image_path)[1].lower()
    filename = os.path.basename(image_path)
    base_name_prefix = "mask" if is_mask else "image"

    try:
        image = Image.open(image_path)
    except IOError:
        print(f"[ERROR] Could not open image {image_path}")
        return counter

    try:
        image.seek(1)
        is_stack = True
    except EOFError:
        is_stack = False
    image.seek(0)

    if is_stack:
        while True:
            try:
                counter = save_frame(image, output_folder, base_name_prefix, counter, ext)
                image.seek(image.tell() + 1)
            except EOFError:
                break
    else:
        counter = save_frame(image, output_folder, base_name_prefix, counter, ext)

    return counter

def process_all_images(input_folder="Finetune_Model/INPUT_IMAGES", output_folder="utils/temp_files/Images"):
    print("[INFO] Processing Images")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)], key=natural_sort_key)
    counter = 1
    for filename in files_to_process:
        counter = process_image_file(os.path.join(input_folder, filename), output_folder, counter, is_mask=False)
    print("[INFO] Image Processing Complete")

def process_all_masks(input_folder="Finetune_Model/INPUT_MASKS", output_folder="utils/temp_files/Masks"):
    print("[INFO] Processing Masks")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)], key=natural_sort_key)
    counter = 1
    for filename in files_to_process:
        counter = process_image_file(os.path.join(input_folder, filename), output_folder, counter, is_mask=True)
    print("[INFO] Mask Processing Complete")

def pad_image(image, target_height, target_width):
    height, width = image.shape[:2]
    pad_bottom = target_height - height
    pad_right = target_width - width
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    pad_value = (0,) * num_channels
    padded_image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
    return padded_image


def crop_images():
    crop_size = 128
    overlap = 64
    stride = crop_size - overlap
    training_images_folder = os.path.join('utils/temp_files/Model_Training/Training_Dataset/Training_Images')
    training_masks_folder = os.path.join('utils/temp_files/Model_Training/Training_Dataset/Training_Masks')
    image_folder = os.path.join('utils/temp_files/Images')
    mask_folder = os.path.join('utils/temp_files/Masks')
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if img.shape[:2] != mask.shape[:2]:
            print(f"[ERROR] Image and mask size mismatch: {img_file}, {mask_file}")
            continue
        height, width = img.shape[:2]
        pad_height = ((height - crop_size) % stride == 0) and height or ((height - crop_size) // stride + 1) * stride + crop_size
        pad_width = ((width - crop_size) % stride == 0) and width or ((width - crop_size) // stride + 1) * stride + crop_size
        padded_img = pad_image(img, pad_height, pad_width)
        padded_mask = pad_image(mask, pad_height, pad_width)
        img_extension = os.path.splitext(img_file)[1].lower()
        mask_extension = os.path.splitext(mask_file)[1].lower()
        crop_count = 0
        for y in range(0, pad_height - crop_size + 1, stride):
            for x in range(0, pad_width - crop_size + 1, stride):
                cropped_img = padded_img[y:y + crop_size, x:x + crop_size]
                cropped_mask = padded_mask[y:y + crop_size, x:x + crop_size]
                img_name = f'{os.path.splitext(img_file)[0]}_crop_{crop_count + 1}{img_extension}'
                mask_name = f'{os.path.splitext(mask_file)[0]}_crop_{crop_count + 1}{mask_extension}'
                cv2.imwrite(os.path.join(training_images_folder, img_name), cropped_img)
                cv2.imwrite(os.path.join(training_masks_folder, mask_name), cropped_mask)
                crop_count += 1
    print("[INFO] Cropping complete.")

# -------------------- Device Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using device: {device}")

# -------------------- Preprocessing --------------------
process_all_images()
process_all_masks()
crop_images()

# -------------------- Load Dataset --------------------
dataset = SegmentationDataset(images_dir, masks_dir, transform=transforms.ToTensor())
input_channels = dataset.get_input_channels()
total_size = len(dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -------------------- Model Selection --------------------
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
if not model_files:
    print("[ERROR] No model files found.")
    exit()
print("\nAvailable models:")
for idx, model_file in enumerate(model_files, 1):
    print(f"{idx}. {model_file}")
choice = input("\n[INPUT] Enter Number to choose model: ")
try:
    choice = int(choice)
    if choice < 1 or choice > len(model_files):
        print("[ERROR] Invalid choice.")
        exit()
except ValueError:
    print("[ERROR] Invalid input.")
    exit()
selected_model_file = model_files[choice - 1]
model_path = os.path.join(model_dir, selected_model_file)
checkpoint = torch.load(model_path, map_location=device)
first_layer = [k for k in checkpoint if 'conv1' in k and 'weight' in k][0]
expected_input_channels = checkpoint[first_layer].shape[1]
print(f"\n[INFO] Model expects input channels: {expected_input_channels}")

model = AttentionUNet(img_ch=expected_input_channels).to(device)
model.load_state_dict(checkpoint)
model.train()

def finetune_model(model_class, dataset, batch_size, num_epochs, model_path, model_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = DynamicLoss(small_roi_thresh=0.0625, large_roi_thresh=0.0625)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_instance = model_class(img_ch=expected_input_channels).to(device)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
    os.makedirs("utils/temp_files/output", exist_ok=True)

    best_val_loss = float("inf")
    save_path = os.path.join(model_dir, f"{model_name}_finetuned.pth")

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Fine-tuning", ncols=125)

    for epoch in epoch_bar:
        # --- Training ---
        model_instance.train()
        epoch_loss = 0
        for images, masks, _ in train_loader:
            images = images.float().to(device)
            masks = masks.float().to(device)
            optimizer.zero_grad()
            outputs = model_instance(images)
            outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)
            loss = loss_fn(outputs_resized, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        model_instance.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.float().to(device)
                masks = masks.float().to(device)
                outputs = model_instance(images)
                outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)
                loss = loss_fn(outputs_resized, masks)
                val_loss_accum += loss.item()
        val_loss = val_loss_accum / len(val_loader)

        # Update progress bar
        epoch_bar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}"
        })

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_instance.state_dict(), save_path)

finetune_model(
    model_class=AttentionUNet,
    dataset=dataset,
    batch_size=4,
    num_epochs=100,
    model_path=model_path,
    model_name=selected_model_file.replace('.pth', '')
)

clear_images_in_folder("Finetune_Model/INPUT_IMAGES", "Finetune_Model/INPUT_MASKS")