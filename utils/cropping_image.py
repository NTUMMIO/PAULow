import os
import numpy as np
import cv2

def get_largest_power_of_2_window(image):
    """Find the largest square window with a power of 2 that fits inside the image."""
    if image is None:
        raise ValueError("!!! Error: Input image is None !!!")

    height, width = image.shape[:2]  # Get height and width
    max_size = min(width, height)  # The limiting factor

    # Find the largest power of 2 ≤ max_size
    largest_power_of_2 = 2 ** (max_size.bit_length() - 1)
    
    return largest_power_of_2

def pad_image(image, target_height, target_width):
    """Pad an image to match the target dimensions."""
    height, width = image.shape[:2]
    pad_bottom = target_height - height
    pad_right = target_width - width

    if len(image.shape) == 3:  # RGB image
        padded_image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:  # Grayscale mask
        padded_image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)

    return padded_image

def crop_images():
    # Define paths
    model_training_folder = 'Model_Training'
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

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure images loaded successfully
        if img is None:
            print(f"Warning: Failed to load image '{img_file}'. Skipping...")
            continue
        if mask is None:
            print(f"Warning: Failed to load mask '{mask_file}'. Skipping...")
            continue

        # Determine the maximum crop size
        crop_size = get_largest_power_of_2_window(img)

        # Compute padded size correctly
        padded_width = (img.shape[1] // crop_size) * crop_size if img.shape[1] % crop_size == 0 else ((img.shape[1] // crop_size) + 1) * crop_size
        padded_height = (img.shape[0] // crop_size) * crop_size if img.shape[0] % crop_size == 0 else ((img.shape[0] // crop_size) + 1) * crop_size

        # Pad images and masks
        padded_img = pad_image(img, padded_height, padded_width)
        padded_mask = pad_image(mask, padded_height, padded_width)

        # Compute grid size
        num_cols = padded_width // crop_size
        num_rows = padded_height // crop_size

        print(f" --> {img_file}: Padded to {padded_width}x{padded_height}, Cropping {num_rows}x{num_cols} patches.")

        # Crop systematically
        crop_count = 0
        img_extension = os.path.splitext(img_file)[1].lower()  # Get the extension of the input image
        mask_extension = os.path.splitext(mask_file)[1].lower()  # Get the extension of the mask image

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * crop_size
                y = row * crop_size

                cropped_img = padded_img[y:y + crop_size, x:x + crop_size]
                cropped_mask = padded_mask[y:y + crop_size, x:x + crop_size]

                # Save cropped images with the same extension as the original image
                img_name = f'{os.path.splitext(img_file)[0]}_crop_{crop_count + 1}{img_extension}'
                mask_name = f'{os.path.splitext(mask_file)[0]}_crop_{crop_count + 1}{mask_extension}'

                cv2.imwrite(os.path.join(training_images_folder, img_name), cropped_img)
                cv2.imwrite(os.path.join(training_masks_folder, mask_name), cropped_mask)

                print(f"Saved {img_name} and {mask_name}")
                crop_count += 1

    print("Cropping complete.")