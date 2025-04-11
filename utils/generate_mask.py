import os
import shutil
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.model.unet import AttentionUNet
from utils.cropping_image import get_largest_power_of_2_window
import tifffile as tiff

def process_images_across_folds():
    # Define paths
    data_path = os.path.join("utils/temp_files/Model_Training", "Test_Dataset")
    generated_path = os.path.join("utils/temp_files/Model_Validation")
    folder_path = os.path.join(data_path, "Test_Images")
    output_folder = os.path.join(generated_path, "Generated_Masks")

    # Temporary folders for processing
    temp_crop_folder = os.path.join(generated_path, "temp_crop")
    temp_mask_folder = os.path.join(generated_path, "temp_mask")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_crop_folder, exist_ok=True)
    os.makedirs(temp_mask_folder, exist_ok=True)

    # Define valid image extensions and transform
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    transform = transforms.Compose([transforms.ToTensor()])

    # Get list of images
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print("!!! No valid images found !!! Please check images in utils/temp_files/Model_Training/Test_Dataset/Test_Images")
        return

    # Find all fold model files
    model_dir = "utils/temp_files/output"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth") and "_fold_" in f]
    model_files.sort(key=lambda x: int(x.split("_fold_")[1].split(".")[0]))

    # Loop over each model
    for model_file in model_files:
        fold_num = model_file.split("_fold_")[1].split(".")[0]
        print(f"\n==============================")
        print(f"Processing with model: {model_file} (Fold {fold_num})")
        print(f"==============================")

        # Load model
        model_path = os.path.join(model_dir, model_file)
        model = AttentionUNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Output path for this fold
        fold_output_folder = os.path.join(output_folder, f"fold_{fold_num}")
        os.makedirs(fold_output_folder, exist_ok=True)

        # Process each image
        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}...")

            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to load image: {filename}. Skipping...")
                continue

            orig_height, orig_width, _ = image.shape
            best_patch_size = get_largest_power_of_2_window(image)

            pad_width = ((orig_width // best_patch_size) + 1) * best_patch_size - orig_width
            pad_height = ((orig_height // best_patch_size) + 1) * best_patch_size - orig_height
            padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded_height, padded_width, _ = padded_image.shape

            # Save patches to temp_crop
            crop_count = 0
            for row in range(0, padded_height, best_patch_size):
                for col in range(0, padded_width, best_patch_size):
                    cropped_img = padded_image[row:row + best_patch_size, col:col + best_patch_size]
                    patch_name = f"{os.path.splitext(filename)[0]}_crop_{crop_count + 1}{os.path.splitext(filename)[1]}"
                    patch_path = os.path.join(temp_crop_folder, patch_name)
                    cv2.imwrite(patch_path, cropped_img)
                    crop_count += 1

            # Inference on each patch
            patch_coords = []
            for patch_filename in os.listdir(temp_crop_folder):
                if patch_filename.startswith('.') or not patch_filename.lower().endswith(valid_extensions):
                    continue

                patch_path = os.path.join(temp_crop_folder, patch_filename)
                patch = Image.open(patch_path)
                patch_tensor = transform(patch).unsqueeze(0)

                with torch.no_grad():
                    output = model(patch_tensor)

                output_mask = output.squeeze().cpu().numpy()
                output_mask = (output_mask > 0.5).astype(np.uint8) * 255
                output_mask_resized = cv2.resize(output_mask, (best_patch_size, best_patch_size), interpolation=cv2.INTER_NEAREST)
                mask_path = os.path.join(temp_mask_folder, patch_filename)
                Image.fromarray(output_mask_resized).convert("L").save(mask_path)
                patch_coords.append((patch_filename, output_mask_resized))

            # Reconstruct full mask
            full_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
            for patch_filename, output_mask in patch_coords:
                try:
                    crop_index = int(patch_filename.split("_crop_")[1].split('.')[0])
                    row = (crop_index - 1) // (padded_width // best_patch_size)
                    col = (crop_index - 1) % (padded_width // best_patch_size)
                except Exception as e:
                    print(f"Error parsing patch filename: {patch_filename}. Error: {e}")
                    continue

                full_mask[row * best_patch_size: (row + 1) * best_patch_size,
                          col * best_patch_size: (col + 1) * best_patch_size] = output_mask

            final_mask = full_mask[:orig_height, :orig_width]
            mask_output_path = os.path.join(fold_output_folder, filename)
            if filename.lower().endswith(('.tif', '.tiff')):
                tiff.imwrite(mask_output_path, final_mask)
            else:
                Image.fromarray(final_mask).convert("L").save(mask_output_path)

            # Clean up temp folders
            shutil.rmtree(temp_crop_folder)
            shutil.rmtree(temp_mask_folder)
            os.makedirs(temp_crop_folder, exist_ok=True)
            os.makedirs(temp_mask_folder, exist_ok=True)

    print("\nAll fold models processed successfully!")

# Call the function to process images
process_images_across_folds()
