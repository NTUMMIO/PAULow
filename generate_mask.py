import os
import torch
from model.unet import AttentionUNet
from torchvision import transforms
from PIL import Image
import numpy as np
import tifffile as tiff
import shutil
from utils.cropping_image import get_largest_power_of_2_window
import cv2

# Define paths
data_path = os.path.join("Model_Training", "Test_Dataset")
generated_path = os.path.join("Model_Validation")
folder_path = os.path.join(data_path, "Test_Images")
output_folder = os.path.join(generated_path, "Generated_Masks")

# Temporary folders for processing
temp_crop_folder = os.path.join(generated_path, "temp_crop")
temp_mask_folder = os.path.join(generated_path, "temp_mask")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(temp_crop_folder, exist_ok=True)
os.makedirs(temp_mask_folder, exist_ok=True)

# Load trained model
model_path = os.path.join("output", "straightcell_fold_5.pth")
print(f"Loading model from: {os.path.abspath(model_path)}...")

model = AttentionUNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully!")

# Define valid image extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
transform = transforms.Compose([transforms.ToTensor()])

# Get list of images
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
if not image_files:
    print("!!! No valid images found !!! Please check images in Test_Images")
    exit()

for filename in image_files:
    file_path = os.path.join(folder_path, filename)
    print(f"Processing: {filename}...")

    # Load image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image: {filename}. Skipping...")
        continue

    orig_height, orig_width, _ = image.shape

    # Determine the maximum crop size
    best_patch_size = get_largest_power_of_2_window(image)

    # Compute padded size
    pad_width = ((orig_width // best_patch_size) + 1) * best_patch_size - orig_width
    pad_height = ((orig_height // best_patch_size) + 1) * best_patch_size - orig_height
    padded_width = orig_width + pad_width
    padded_height = orig_height + pad_height

    # Pad image
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Save patches to temp_crop (directly from padded image)
    crop_count = 0
    for row in range(0, padded_height, best_patch_size):
        for col in range(0, padded_width, best_patch_size):
            cropped_img = padded_image[row:row + best_patch_size, col:col + best_patch_size]
            
            # Save the cropped image patch
            patch_name = f"{os.path.splitext(filename)[0]}_crop_{crop_count + 1}{os.path.splitext(filename)[1]}"
            patch_path = os.path.join(temp_crop_folder, patch_name)
            cv2.imwrite(patch_path, cropped_img)

            crop_count += 1

    # Process patches and generate masks
    patch_coords = []
    for patch_filename in os.listdir(temp_crop_folder):
        # Skip non-image files (e.g., .DS_Store)
        if patch_filename.startswith('.') or not patch_filename.lower().endswith(valid_extensions):
            continue

        patch_path = os.path.join(temp_crop_folder, patch_filename)
        patch = Image.open(patch_path)
        patch_tensor = transform(patch).unsqueeze(0)

        with torch.no_grad():
            output = model(patch_tensor)

        # Convert to mask
        output_mask = output.squeeze().cpu().numpy()
        output_mask = (output_mask > 0.5).astype(np.uint8) * 255

        # Resize output mask to match patch size
        output_mask_resized = cv2.resize(output_mask, (best_patch_size, best_patch_size), interpolation=cv2.INTER_NEAREST)

        # Save mask
        mask_path = os.path.join(temp_mask_folder, patch_filename)
        Image.fromarray(output_mask_resized).convert("L").save(mask_path)
        patch_coords.append((patch_filename, output_mask_resized))

    # Reconstruct full mask
    full_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
    for patch_filename, output_mask in patch_coords:
        # Find position of the patch from its filename
        parts = patch_filename.split("_crop_")
        
        # Extract row and column information
        try:
            base_name = parts[0]
            crop_index = parts[1].split('.')[0]
            
            # Calculate row and col based on the crop index
            crop_number = int(crop_index)
            
            # Calculate the row and col based on the patch size and number of patches in the row
            row = (crop_number - 1) // (padded_width // best_patch_size)
            col = (crop_number - 1) % (padded_width // best_patch_size)

            # Debugging: Log patch placement
            print(f"Placing patch {patch_filename} at position (row={row}, col={col})")

        except Exception as e:
            print(f"Error parsing patch filename: {patch_filename}. Error: {e}")
            continue

        # Place patch in full mask
        full_mask[row * best_patch_size: (row + 1) * best_patch_size,
                  col * best_patch_size: (col + 1) * best_patch_size] = output_mask

    # Crop back to original size
    final_mask = full_mask[:orig_height, :orig_width]

    # Save the final mask (no post-processing)
    mask_output_path = os.path.join(output_folder, filename)
    if filename.lower().endswith(('.tif', '.tiff')):
        tiff.imwrite(mask_output_path, final_mask)
    else:
        Image.fromarray(final_mask).convert("L").save(mask_output_path)

    print(f"--> Saved mask: {mask_output_path}")

    # Clear temp folders
    shutil.rmtree(temp_crop_folder)
    shutil.rmtree(temp_mask_folder)
    os.makedirs(temp_crop_folder, exist_ok=True)
    os.makedirs(temp_mask_folder, exist_ok=True)

print("Mask generation complete!")
