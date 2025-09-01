import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
import numpy as np
import tifffile as tiff
import cv2
from utils.model.unet import AttentionUNet
from utils.clear_images import clear_images_in_folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

generated_path = "Use_Model"
input_folder = os.path.join(generated_path, "INPUT_IMAGES")
output_folder = os.path.join(generated_path, "OUTPUT_MASKS")
os.makedirs(output_folder, exist_ok=True)

model_folder = os.path.join(generated_path, "saved_models")
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
if not model_files:
    print("[ERROR] No model files found.")
    sys.exit()

print("\nAvailable models:")
for idx, model_file in enumerate(model_files, 1):
    print(f"{idx}. {model_file}")

choice = input("\n[INPUT] Enter Number to choose model: ")
print("\n")

try:
    choice = int(choice)
    if not (1 <= choice <= len(model_files)):
        raise ValueError
except ValueError:
    print("[ERROR] Invalid input.")
    sys.exit()

selected_model_file = model_files[choice - 1]
model_path = os.path.join(model_folder, selected_model_file)

checkpoint = torch.load(model_path, map_location=device)
first_layer = [k for k in checkpoint if 'conv1' in k and 'weight' in k][0]
expected_input_channels = checkpoint[first_layer].shape[1]
print(f"[INFO] Model expects input channels: {expected_input_channels}")

model = AttentionUNet(img_ch=expected_input_channels).to(device)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.ToTensor()

def pad_image(img, target_h, target_w):
    """Pad image to (target_h, target_w) with zeros."""
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

def generate_weight_map(patch_size):
    """Generate a smooth weight map for blending overlapping patches."""
    h, w = patch_size
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def adapt_channels(image, expected_channels):
    """Adapt image channels to match model input."""
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] < expected_channels:
        padding = np.zeros((*image.shape[:2], expected_channels - image.shape[-1]), dtype=image.dtype)
        image = np.concatenate((image, padding), axis=-1)
    elif image.shape[-1] > expected_channels:
        image = image[:, :, :expected_channels]
    return image

def load_and_preprocess_image(path):
    """Load TIFF image and adapt to model input format."""
    img = tiff.imread(path)
    img = np.asarray(img)

    # [H, W] - grayscale single image
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        return [adapt_channels(img, expected_input_channels)], False

    # [H, W, C] - multi-channel image
    elif img.ndim == 3 and img.shape[2] <= 4:
        return [adapt_channels(img, expected_input_channels)], False

    # [Z, H, W] - grayscale image stack
    elif img.ndim == 3:
        return [adapt_channels(img[z, :, :, np.newaxis], expected_input_channels) for z in range(img.shape[0])], True

    # [Z, H, W, C] - multi-channel image stack
    elif img.ndim == 4:
        return [adapt_channels(img[z], expected_input_channels) for z in range(img.shape[0])], True

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

# -------------------
# Inference
# -------------------
def process_single_image(image):
    """Run inference on a single 2D image with patching + blending."""
    h, w = image.shape[:2]
    patch_size = 128
    overlap = 64
    stride = patch_size - overlap

    # Compute padded size
    pad_height = ((h - patch_size) % stride == 0) and h or ((h - patch_size) // stride + 1) * stride + patch_size
    pad_width = ((w - patch_size) % stride == 0) and w or ((w - patch_size) // stride + 1) * stride + patch_size

    padded = pad_image(image, pad_height, pad_width)

    weight_map = generate_weight_map((patch_size, patch_size))
    accum_mask = np.zeros((pad_height, pad_width), dtype=np.float32)
    accum_weight = np.zeros((pad_height, pad_width), dtype=np.float32)

    for y in range(0, pad_height - patch_size + 1, stride):
        for x in range(0, pad_width - patch_size + 1, stride):
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

# -------------------
# Run Inference
# -------------------
valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
for fname in os.listdir(input_folder):
    if not fname.lower().endswith(valid_exts):
        continue

    try:
        full_path = os.path.join(input_folder, fname)
        images, is_stack = load_and_preprocess_image(full_path)

        if is_stack:
            print(f"--> Processing image stack: {fname}")
            output_stack = [process_single_image(img) for img in images]
            tiff.imwrite(
                os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.tif"),
                np.stack(output_stack).astype(np.uint8)
            )
        else:
            print(f"--> Processing single image: {fname}")
            result = process_single_image(images[0])
            tiff.imwrite(
                os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.tif"),
                result.astype(np.uint8)
            )

    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")

print("[INFO] Mask Generation Complete.")
clear_images_in_folder("Use_Model/INPUT_IMAGES")
