import os
import numpy as np
import cv2
import sys

def get_largest_power_of_2_window(image):
    """Find the largest square window with a power of 2 that fits inside the image."""
    if image is None:
        raise ValueError("[ERROR] Input image is None")
    
    largest_power_of_2 = 128  # Example fixed size, could be adjusted based on the image dimensions
    
    return largest_power_of_2

def pad_image(image, target_height, target_width):
    """Pad an image to match the target dimensions, preserving the number of channels."""
    height, width = image.shape[:2]
    pad_bottom = target_height - height
    pad_right = target_width - width

    # Get the number of channels
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    
    # Generate the padding value (0s for all channels)
    pad_value = (0,) * num_channels  # Creates a tuple of zeros matching the number of channels

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=pad_value)

    return padded_image

def crop_images():
    crop_size = 128
    overlap = 64
    stride = crop_size - overlap

    # Define paths
    model_training_folder = 'utils/temp_files/Model_Training'
    training_dataset_folder = os.path.join(model_training_folder, 'Training_Dataset')

    training_images_folder = os.path.join(training_dataset_folder, 'Training_Images')
    training_masks_folder = os.path.join(training_dataset_folder, 'Training_Masks')

    os.makedirs(training_images_folder, exist_ok=True)
    os.makedirs(training_masks_folder, exist_ok=True)

    image_folder = os.path.join(training_dataset_folder, 'Original_Training_Images')
    mask_folder = os.path.join(training_dataset_folder, 'Original_Training_Masks')

    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[Warning] Failed to load image '{img_file}'. Skipping...")
            continue
        if mask is None:
            print(f"[Warning] Failed to load mask '{mask_file}'. Skipping...")
            continue

        # Check if image and mask have the same size
        if img.shape[:2] != mask.shape[:2]:
            print(f"[ERROR] Input image and mask have different size: {img_file} and {mask_file}")
            sys.exit(1)

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
