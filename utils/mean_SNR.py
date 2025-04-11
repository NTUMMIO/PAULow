import os
import numpy as np
import cv2
from glob import glob

def calculate_snr(image, mask):
    """
    Calculate SNR given a grayscale image and its binary mask.
    """
    signal_pixels = image[mask > 0]
    background_pixels = image[mask == 0]

    mean_signal = np.mean(signal_pixels)
    std_noise = np.std(background_pixels)

    return mean_signal / std_noise if std_noise > 0 else float('inf')


def calculate_mean_snr(masks_folder="utils/temp_files/Masks", images_folder="utils/temp_files/Images"):
    """
    Calculate mean SNR across all image-mask pairs in specified folders.
    
    Parameters:
        masks_folder (str): Folder containing binary masks.
        images_folder (str): Folder containing grayscale images.
        
    Returns:
        float or None: Mean SNR if valid pairs exist, else None.
    """
    # Search for mask files with valid extensions
    mask_files = []
    for ext in ["*.png", "*.tif", "*.tiff","*.jpg", "*.jpeg"]:
        mask_files.extend(glob(os.path.join(masks_folder, ext)))

    snr_values = []

    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # Convert MaskX.tif to imageX.tif
        mask_basename = os.path.basename(mask_file)
        image_filename = mask_basename.replace("Mask", "image").replace("MASK", "image")
        image_file = os.path.join(images_folder, image_filename)

        if os.path.exists(image_file):
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if image is not None and mask is not None:
                snr = calculate_snr(image, mask)
                snr_values.append(snr)
            else:
                print(f" Could not read image or mask: {image_file}")
        else:
            print(f" Image not found for mask: {mask_basename}")

    if snr_values:
        mean_snr = np.mean(snr_values)
        print(f" Mean SNR for all image-mask pairs: {mean_snr:.3f}")
        return mean_snr
    else:
        print(" No valid image-mask pairs found.")
        return None
    
calculate_mean_snr()