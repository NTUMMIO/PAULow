import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import tifffile as tiff
import shutil
import time
import cv2
from utils.model.unet import AttentionUNet
from utils.cropping_image import get_largest_power_of_2_window

# Utility functions
def get_image_channels(image):
    if image.ndim == 2:
        return 1  # Grayscale
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            return image.shape[2]  # (H, W, C)
        elif image.shape[0] <= 4:
            return image.shape[0]  # (C, H, W)
        else:
            return 1  # Stack of grayscale slices (S, H, W)
    elif image.ndim == 4:
        return image.shape[-1]  # (S, H, W, C)
    return 1

def pad_image(image, target_height, target_width):
    h, w = image.shape[:2]
    top = 0
    bottom = target_height - h
    left = 0
    right = target_width - w
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def clear_folder(folder):
    for f in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, f))
        except Exception as e:
            print(f"[ERROR] Could not delete {f}: {e}")

# Paths
generated_path = "Use_Model"
input_folder = os.path.join(generated_path, "INPUT_IMAGES")
output_folder = os.path.join(generated_path, "OUTPUT_MASKS")
temp_crop_folder = os.path.join(generated_path, "processing_area/temp_crop")
temp_mask_folder = os.path.join(generated_path, "processing_area/temp_mask")
temp_stack_folder = os.path.join(generated_path, "processing_area/temp_stack")
temp_stack_mask_folder = os.path.join(generated_path, "processing_area/temp_stack_mask")

for path in [output_folder, temp_crop_folder, temp_mask_folder, temp_stack_folder, temp_stack_mask_folder]:
    os.makedirs(path, exist_ok=True)

# Clear output
clear_folder(output_folder)
print("[INFO] Previous Output Masks Cleared.")

# Load model
model_folder = os.path.join(generated_path, "saved_models")
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
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
        print(f"[ERROR] Invalid choice. Please select a number between 1 and {len(model_files)}.")
        exit()
except ValueError:
    print("[ERROR] Invalid input.")
    exit()

selected_model_file = model_files[choice - 1]
model_path = os.path.join(model_folder, selected_model_file)

# Detect input channel from a sample image
sample_image_path = os.path.join(input_folder, os.listdir(input_folder)[0])
sample_image = tiff.imread(sample_image_path)

# If it's a stack (4D or 3D with Slices), check per-slice channels
if sample_image.ndim == 4:
    input_channels = sample_image.shape[-1]  # (S, H, W, C)
elif sample_image.ndim == 3 and sample_image.shape[0] > 4:
    input_channels = get_image_channels(sample_image[0])  # First slice
else:
    input_channels = get_image_channels(sample_image)

# Initialize model with correct input channels
model = AttentionUNet(img_ch=input_channels)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def process_image(image, filename, output_dir=output_folder):
    input_channels = get_image_channels(image)
    orig_height, orig_width = image.shape[:2]
    best_patch_size = get_largest_power_of_2_window(image)
    padded_width = ((orig_width + best_patch_size - 1) // best_patch_size) * best_patch_size
    padded_height = ((orig_height + best_patch_size - 1) // best_patch_size) * best_patch_size
    padded_image = pad_image(image, padded_height, padded_width)

    crop_count = 0
    patch_coords = []
    for row in range(0, padded_height, best_patch_size):
        for col in range(0, padded_width, best_patch_size):
            cropped_img = padded_image[row:row + best_patch_size, col:col + best_patch_size]
            if cropped_img.dtype != np.uint8:
                cropped_img = cv2.normalize(cropped_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            patch_name = f"{os.path.splitext(filename)[0]}_crop_{crop_count + 1}{os.path.splitext(filename)[1]}"
            patch_path = os.path.join(temp_crop_folder, patch_name)
            cv2.imwrite(patch_path, cropped_img)
            crop_count += 1

    for patch_filename in os.listdir(temp_crop_folder):
        if not patch_filename.lower().endswith(valid_extensions):
            continue
        patch_path = os.path.join(temp_crop_folder, patch_filename)
        patch_img = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
        if patch_img.ndim == 2:
            patch_img = np.expand_dims(patch_img, axis=-1)
        patch_tensor = transform(patch_img).unsqueeze(0).float()

        with torch.no_grad():
            output = model(patch_tensor)

        output_mask = output.squeeze().cpu().numpy()
        output_mask = (output_mask > 0.5).astype(np.uint8) * 255
        mask_path = os.path.join(temp_mask_folder, patch_filename)
        Image.fromarray(output_mask).save(mask_path)
        patch_coords.append((patch_filename, output_mask))

    full_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
    for patch_filename, output_mask in patch_coords:
        crop_index = int(patch_filename.split("_crop_")[1].split('.')[0])
        row = (crop_index - 1) // (padded_width // best_patch_size)
        col = (crop_index - 1) % (padded_width // best_patch_size)
        full_mask[row * best_patch_size: (row + 1) * best_patch_size,
                  col * best_patch_size: (col + 1) * best_patch_size] = output_mask

    final_mask = full_mask[:orig_height, :orig_width]
    output_name = os.path.splitext(filename)[0] + "_mask" + os.path.splitext(filename)[1]
    output_path = os.path.join(output_dir, output_name)
    tiff.imwrite(output_path, final_mask)

    clear_folder(temp_crop_folder)
    clear_folder(temp_mask_folder)

def process_stack(stack, filename):
    if stack.ndim != 3:
        print(f"[WARNING] Not a stack: {filename}")
        return
    slices = stack.shape[0]

    # Save slices
    for i in range(slices):
        slice_img = stack[i]
        if slice_img.dtype != np.uint8:
            slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        slice_path = os.path.join(temp_stack_folder, f"{os.path.splitext(filename)[0]}_slice_{i}.tif")
        tiff.imwrite(slice_path, slice_img)

    # Process each slice
    for i in range(slices):
        slice_img = tiff.imread(os.path.join(temp_stack_folder, f"{os.path.splitext(filename)[0]}_slice_{i}.tif"))
        if slice_img.ndim == 3 and slice_img.shape[-1] > 4:
            slice_img = slice_img[:, :, :3]  # Truncate or handle accordingly
        process_image(slice_img, f"{os.path.splitext(filename)[0]}_slice_{i}.tif", output_dir=temp_stack_mask_folder)

    # Stack masks
    mask_stack = []
    for i in range(slices):
        mask_slice_path = os.path.join(temp_stack_mask_folder, f"{os.path.splitext(filename)[0]}_slice_{i}_mask.tif")
        mask_slice = tiff.imread(mask_slice_path)
        if mask_slice.ndim == 3:
            mask_slice = mask_slice[:, :, 0]
        mask_stack.append(mask_slice)
        os.remove(mask_slice_path)

    final_stack = np.stack(mask_stack, axis=0)
    output_name = os.path.splitext(filename)[0] + "_mask" + os.path.splitext(filename)[1]
    output_path = os.path.join(output_folder, output_name)
    tiff.imwrite(output_path, final_stack)

    clear_folder(temp_stack_folder)

# Main loop
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(valid_extensions):
        continue

    file_path = os.path.join(input_folder, filename)
    print(f"[INFO] Processing {filename}...")

    try:
        img = tiff.imread(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {filename}: {e}")
        continue

    if img.ndim == 3:
        process_stack(img, filename)
    else:
        process_image(img, filename)

# Clear input folder
clear_folder(input_folder)

print("[INFO] Mask generation complete.")
print("[INFO] Input Images Cleared.") 