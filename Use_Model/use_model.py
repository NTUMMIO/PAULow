import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
import numpy as np
import tifffile as tiff
import cv2
from utils.model.unet import AttentionUNet

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# Paths
generated_path = "Use_Model"
input_folder = os.path.join(generated_path, "INPUT_IMAGES")
output_folder = os.path.join(generated_path, "OUTPUT_MASKS")
os.makedirs(output_folder, exist_ok=True)

# --- Load Model ---
model_folder = os.path.join(generated_path, "saved_models")
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
if not model_files:
    print("[ERROR] No model files found.")
    exit()

print("\nAvailable models:")
for idx, model_file in enumerate(model_files, 1):
    print(f"{idx}. {model_file}")
choice = input("\n[INPUT] Enter Number to choose model: ")

print("\n")

try:
    choice = int(choice)
    if choice < 1 or choice > len(model_files):
        print(f"[ERROR] Invalid choice.")
        exit()
except ValueError:
    print("[ERROR] Invalid input.")
    exit()

selected_model_file = model_files[choice - 1]
model_path = os.path.join(model_folder, selected_model_file)

checkpoint = torch.load(model_path, map_location=device)
first_layer = [k for k in checkpoint if 'conv1' in k and 'weight' in k][0]
expected_input_channels = checkpoint[first_layer].shape[1]
print(f"[INFO] Model expects input channels: {expected_input_channels}")

model = AttentionUNet(img_ch=expected_input_channels).to(device)
model.load_state_dict(checkpoint)
model.eval()

# --- Preprocessing ---
transform = transforms.ToTensor()

def normalize_image(img):
    img = img.astype(np.float32)
    max_val = img.max()
    if max_val > 0:
        img = (img / max_val) * 255.0
    return img.astype(np.uint8)

def pad_image(img, target_h, target_w):
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

def generate_weight_map(patch_size, overlap=64):
    h, w = patch_size
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def adapt_channels(image, expected_channels):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] < expected_channels:
        padding = np.zeros((*image.shape[:2], expected_channels - image.shape[-1]), dtype=image.dtype)
        image = np.concatenate((image, padding), axis=-1)
    elif image.shape[-1] > expected_channels:
        image = image[:, :, :expected_channels]
    return image

def load_and_preprocess_image(path):
    img = tiff.imread(path)
    img = np.asarray(img)

    # Shape [H, W] - grayscale single image
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        return [adapt_channels(img, expected_input_channels)], False

    # Shape [H, W, C] - multi-channel image
    elif img.ndim == 3 and img.shape[2] <= 4:
        return [adapt_channels(img, expected_input_channels)], False

    # Shape [Z, H, W] - grayscale image stack
    elif img.ndim == 3:
        return [adapt_channels(img[z, :, :, np.newaxis], expected_input_channels) for z in range(img.shape[0])], True

    # Shape [Z, H, W, C] - multi-channel image stack
    elif img.ndim == 4:
        return [adapt_channels(img[z], expected_input_channels) for z in range(img.shape[0])], True

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def clear_folder(folder):
    for f in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, f))
        except Exception as e:
            print(f"[ERROR] Could not delete {f}: {e}")

# --- Inference ---
def process_single_image(image):
    h, w = image.shape[:2]
    patch_size = 128
    stride = 64
    padded_h = ((h - 1) // stride + 1) * stride + patch_size - stride
    padded_w = ((w - 1) // stride + 1) * stride + patch_size - stride
    padded = pad_image(image, padded_h, padded_w)

    weight_map = generate_weight_map((patch_size, patch_size), stride)
    accum_mask = np.zeros((padded_h, padded_w), dtype=np.float32)
    accum_weight = np.zeros((padded_h, padded_w), dtype=np.float32)

    for y in range(0, padded_h - patch_size + 1, stride):
        for x in range(0, padded_w - patch_size + 1, stride):
            patch = padded[y:y + patch_size, x:x + patch_size]
            patch_tensor = transform(patch).unsqueeze(0).float().to(device)

            with torch.no_grad():
                pred = model(patch_tensor)

            pred_np = pred.squeeze().cpu().numpy()
            pred_np = (pred_np > 0.5).astype(np.float32)

            accum_mask[y:y + patch_size, x:x + patch_size] += pred_np * weight_map
            accum_weight[y:y + patch_size, x:x + patch_size] += weight_map

    result = np.divide(accum_mask, accum_weight, out=np.zeros_like(accum_mask), where=accum_weight > 0)
    result = (result > 0.5).astype(np.uint8) * 255
    return result[:h, :w]

# --- Run Inference ---
valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
for fname in os.listdir(input_folder):
    if not fname.lower().endswith(valid_exts):
        continue
    try:
        full_path = os.path.join(input_folder, fname)
        images, is_stack = load_and_preprocess_image(full_path)

        if is_stack:
            print(f"--> Processing image stack: {fname}")
            output_stack = [process_single_image(normalize_image(img)) for img in images]
            tiff.imwrite(os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.tif"), np.stack(output_stack))
        else:
            print(f"--> Processing single image: {fname}")
            result = process_single_image(normalize_image(images[0]))
            tiff.imwrite(os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.tif"), result)

    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")

clear_folder(input_folder)

print("[INFO] Mask Generation Complete.")
print("[INFO] Input images cleared.")
