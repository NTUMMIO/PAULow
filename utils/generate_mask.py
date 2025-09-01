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
from utils.cropping_image import pad_image
import tifffile as tiff
from utils.datasetloader.dataset import SegmentationDataset

def safe_rmtree(folder_path):
    def remove_readonly(func, path, _):
        os.chmod(path, 0o777)
        func(path)
    if os.path.exists(folder_path):
        time.sleep(0.2)
        shutil.rmtree(folder_path, onerror=remove_readonly)

def generate_weight_map(patch_size, overlap=64):
    h, w = patch_size
    y = np.hanning(h)
    x = np.hanning(w)
    weight = np.outer(y, x)
    weight = weight.astype(np.float32)
    return weight

def process_images_across_folds():
    data_path = os.path.join("utils/temp_files/Model_Training", "Test_Dataset")
    generated_path = os.path.join("utils/temp_files/Model_Validation")
    folder_path = os.path.join(data_path, "Images")
    output_folder = os.path.join(generated_path, "Generated_Masks")

    os.makedirs(output_folder, exist_ok=True)

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

        dataset = SegmentationDataset(folder_path, folder_path, transform=transform)
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

            if filename.lower().endswith(('.tif', '.tiff')):
                image = tiff.imread(file_path)
            else:
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"[ERROR] Failed to load image: {filename}. Skipping...")
                continue

            orig_height, orig_width = image.shape[:2]
            patch_size = 128
            stride = 64
            weight_map = generate_weight_map((patch_size, patch_size), overlap=stride)

            padded_width = ((orig_width - 1) // stride + 1) * stride + patch_size - stride
            padded_height = ((orig_height - 1) // stride + 1) * stride + patch_size - stride
            padded_image = pad_image(image, padded_height, padded_width)

            accum_mask = np.zeros((padded_height, padded_width), dtype=np.float32)
            accum_weight = np.zeros((padded_height, padded_width), dtype=np.float32)

            for row in range(0, padded_height - patch_size + 1, stride):
                for col in range(0, padded_width - patch_size + 1, stride):
                    patch_img = padded_image[row:row + patch_size, col:col + patch_size]

                    if patch_img.ndim == 2:
                        patch_img = np.expand_dims(patch_img, axis=-1)
                    if patch_img.shape[2] != input_channels:
                        continue

                    patch_tensor = transform(patch_img).unsqueeze(0).float()

                    with torch.no_grad():
                        output = model(patch_tensor)

                    output_mask = output.squeeze().cpu().numpy()
                    output_mask = (output_mask > 0.5).astype(np.float32)

                    accum_mask[row:row + patch_size, col:col + patch_size] += output_mask * weight_map
                    accum_weight[row:row + patch_size, col:col + patch_size] += weight_map

            final_mask = np.divide(accum_mask, accum_weight, out=np.zeros_like(accum_mask), where=accum_weight > 0)
            final_mask = (final_mask > 0.5).astype(np.uint8) * 255
            final_mask = final_mask[:orig_height, :orig_width]

            mask_output_path = os.path.join(fold_output_folder, filename)
            if filename.lower().endswith(('.tif', '.tiff')):
                tiff.imwrite(mask_output_path, final_mask)
            else:
                Image.fromarray(final_mask).convert("L").save(mask_output_path)

    print("[INFO] All fold models processed successfully!")
