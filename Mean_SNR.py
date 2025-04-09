import os
import numpy as np
import cv2  # To read the images
from glob import glob

def calculate_snr(image, mask):
    """
    Calculate SNR given an image and its corresponding mask.
    
    Parameters:
        image (numpy array): The input grayscale image.
        mask (numpy array): The binary mask (1 for signal, 0 for background).
        
    Returns:
        float: SNR value.
    """
    signal_pixels = image[mask > 0]  # Extract signal region (1 in mask)
    background_pixels = image[mask == 0]  # Extract background region (0 in mask)

    mean_signal = np.mean(signal_pixels)
    std_noise = np.std(background_pixels)  # Standard deviation of background

    return mean_signal / std_noise if std_noise > 0 else float('inf')


def calculate_mean_snr(masks_folder, images_folder):
    """
    Calculate the mean SNR value for all images in a folder with their corresponding masks.
    
    Parameters:
        masks_folder (str): Path to the folder containing the masks.
        images_folder (str): Path to the folder containing the images.
        
    Returns:
        float: The mean SNR of all images in the folder.
    """
    # Get all mask image paths from the folder, considering .png, .tif, and .tiff extensions
    mask_files = glob(os.path.join(masks_folder, "*.png")) + \
                 glob(os.path.join(masks_folder, "*.tif")) + \
                 glob(os.path.join(masks_folder, "*.tiff"))
                 
    snr_values = []

    for mask_file in mask_files:
        # Load the mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        # Construct the corresponding image file path by converting the mask name to lowercase
        mask_basename = os.path.basename(mask_file)
        image_filename = mask_basename.replace("Mask", "image").lower()  # Convert "Mask1" to "image1"
        image_file = os.path.join(images_folder, image_filename)
        
        print(f"Looking for image file: {image_file}")  # Print the image path for debugging
        
        # Check if the image exists
        if os.path.exists(image_file):
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # Load corresponding image
            if image is not None and mask is not None:
                # Calculate the SNR for this image-mask pair
                snr = calculate_snr(image, mask)
                snr_values.append(snr)
            else:
                print(f"Warning: Image or mask not found for {mask_file}")
        else:
            print(f"Warning: No corresponding image for mask file {mask_file}")

    # Calculate the mean SNR
    if snr_values:
        mean_snr = np.mean(snr_values)
        print(f"Mean SNR value for all images: {mean_snr}")
    else:
        print("No valid images/masks found for SNR calculation.")


# Example usage
masks_folder = "Masks"  # Path to the folder containing masks
images_folder = "Images"  # Path to the folder containing images
calculate_mean_snr(masks_folder, images_folder)
