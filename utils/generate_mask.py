import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import torch
import cv2
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from utils.model.unet import AttentionUNet
from utils.cropping_image import get_largest_power_of_2_window, pad_image
import tifffile as tiff
from utils.datasetloader.dataset import SegmentationDataset

def safe_rmtree(folder_path):
    def remove_readonly(func, path, _):
        os.chmod(path, 0o777)
        func(path)
    if os.path.exists(folder_path):
        time.sleep(0.2)
        shutil.rmtree(folder_path, onerror=remove_readonly)

def process_images_across_folds():
    data_path = os.path.join("utils/temp_files/Model_Training", "Test_Dataset")
    generated_path = os.path.join("utils/temp_files/Model_Validation")
    folder_path = os.path.join(data_path, "Test_Images")
    output_folder = os.path.join(generated_path, "Generated_Masks")

    temp_crop_folder = os.path.join(generated_path, "temp_crop")
    temp_mask_folder = os.path.join(generated_path, "temp_mask")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_crop_folder, exist_ok=True)
    os.makedirs(temp_mask_folder, exist_ok=True)

    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    transform = transforms.Compose([transforms.ToTensor()])

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print("[ERROR] No valid images found! Please check images in Test_Images folder")
        return

    model_dir = "utils/temp_files/output"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth") and "_fold_" in f]
    model_files.sort(key=lambda x: int(x.split("_fold_")[1].split(".")[0]))

    for model_file in model_files:
        fold_num = model_file.split("_fold_")[1].split(".")[0]
        print(f"[INFO] Processing with model: {model_file} (Fold {fold_num})")

        dataset = SegmentationDataset(folder_path, folder_path, transform=transform)  # using same folder for input/output for simplicity
        input_channels = dataset.get_input_channels()

        model_path = os.path.join(model_dir, model_file)
        model = AttentionUNet(img_ch=input_channels)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        fold_output_folder = os.path.join(output_folder, f"fold_{fold_num}")
        os.makedirs(fold_output_folder, exist_ok=True)

        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            print(f"--> Processing: {filename}...")

            # Load image
            if filename.lower().endswith(('.tif', '.tiff')):
                image = tiff.imread(file_path)
            else:
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # PNG, JPG, etc.

            if image is None:
                print(f"[ERROR] Failed to load image: {filename}. Skipping...")
                continue

            orig_height, orig_width = image.shape[:2]
            best_patch_size = get_largest_power_of_2_window(image)

            padded_width = ((orig_width + best_patch_size - 1) // best_patch_size) * best_patch_size
            padded_height = ((orig_height + best_patch_size - 1) // best_patch_size) * best_patch_size

            padded_image = pad_image(image, padded_height, padded_width)
            padded_height, padded_width = padded_image.shape[:2]

            # Save patches
            crop_count = 0
            patch_coords = []
            for row in range(0, padded_height, best_patch_size):
                for col in range(0, padded_width, best_patch_size):
                    cropped_img = padded_image[row:row + best_patch_size, col:col + best_patch_size]
                    patch_name = f"{os.path.splitext(filename)[0]}_crop_{crop_count + 1}{os.path.splitext(filename)[1]}"
                    patch_path = os.path.join(temp_crop_folder, patch_name)
                    if filename.lower().endswith(('.tif', '.tiff')):
                        tiff.imwrite(patch_path, cropped_img)
                    else:
                        Image.fromarray(cropped_img).save(patch_path)
                    crop_count += 1

            # Predict patches
            for patch_filename in os.listdir(temp_crop_folder):
                if patch_filename.startswith('.') or not patch_filename.lower().endswith(valid_extensions):
                    continue

                patch_path = os.path.join(temp_crop_folder, patch_filename)
                if patch_filename.lower().endswith(('.tif', '.tiff')):
                    patch_img = tiff.imread(patch_path)
                else:
                    patch_img = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)

                if patch_img is None:
                    print(f"[WARNING] Could not read patch: {patch_filename}")
                    continue

                if patch_img.ndim == 2:
                    patch_img = np.expand_dims(patch_img, axis=-1)

                if patch_img.shape[2] != input_channels:
                    print(f"[WARNING] Skipping patch {patch_filename} due to mismatched channel count.")
                    continue

                patch_tensor = transform(patch_img).unsqueeze(0).float()

                with torch.no_grad():
                    output = model(patch_tensor)

                output_mask = output.squeeze().cpu().numpy()
                output_mask = (output_mask > 0.5).astype(np.uint8) * 255
                output_mask_resized = cv2.resize(output_mask, (best_patch_size, best_patch_size), interpolation=cv2.INTER_NEAREST)
                mask_path = os.path.join(temp_mask_folder, patch_filename)
                if patch_filename.lower().endswith(('.tif', '.tiff')):
                    tiff.imwrite(mask_path, output_mask_resized)
                else:
                    Image.fromarray(output_mask_resized).convert("L").save(mask_path)
                patch_coords.append((patch_filename, output_mask_resized))

            # Stitch back full mask
            full_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
            for patch_filename, output_mask in patch_coords:
                try:
                    crop_index = int(patch_filename.split("_crop_")[1].split('.')[0])
                    row = (crop_index - 1) // (padded_width // best_patch_size)
                    col = (crop_index - 1) % (padded_width // best_patch_size)
                except Exception as e:
                    print(f"[ERROR] Error parsing patch filename: {patch_filename}. Error: {e}")
                    continue

                full_mask[row * best_patch_size: (row + 1) * best_patch_size,
                          col * best_patch_size: (col + 1) * best_patch_size] = output_mask

            final_mask = full_mask[:orig_height, :orig_width]
            mask_output_path = os.path.join(fold_output_folder, filename)
            if filename.lower().endswith(('.tif', '.tiff')):
                tiff.imwrite(mask_output_path, final_mask)
            else:
                Image.fromarray(final_mask).convert("L").save(mask_output_path)

            # Clean temp folders
            safe_rmtree(temp_crop_folder)
            safe_rmtree(temp_mask_folder)
            os.makedirs(temp_crop_folder, exist_ok=True)
            os.makedirs(temp_mask_folder, exist_ok=True)

    print("[INFO] All fold models processed successfully!")
