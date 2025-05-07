import os
import numpy as np
import cv2

def get_largest_power_of_2_window(image):
    """Find the largest square window with a power of 2 that fits inside the image."""
    if image is None:
        raise ValueError("!!! Error: Input image is None !!!")
    
    largest_power_of_2 = 128  # Example fixed size, could be adjusted based on the image dimensions
    
    return largest_power_of_2

def pad_image(image, padded_height, padded_width, pad_value=0):
    """Pads the image to the target height and width using constant padding."""
    pad_bottom = max(0, padded_height - image.shape[0])
    pad_right = max(0, padded_width - image.shape[1])
    return cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=pad_value)

def crop_images():
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
            print(f"Warning: Failed to load image '{img_file}'. Skipping...")
            continue
        if mask is None:
            print(f"Warning: Failed to load mask '{mask_file}'. Skipping...")
            continue

        crop_size = get_largest_power_of_2_window(img)

        padded_width = ((img.shape[1] + crop_size - 1) // crop_size) * crop_size
        padded_height = ((img.shape[0] + crop_size - 1) // crop_size) * crop_size

        padded_img = pad_image(img, padded_height, padded_width)
        padded_mask = pad_image(mask, padded_height, padded_width)

        num_cols = padded_width // crop_size
        num_rows = padded_height // crop_size

        crop_count = 0
        img_extension = os.path.splitext(img_file)[1].lower()
        mask_extension = os.path.splitext(mask_file)[1].lower()

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * crop_size
                y = row * crop_size

                cropped_img = padded_img[y:y + crop_size, x:x + crop_size]
                cropped_mask = padded_mask[y:y + crop_size, x:x + crop_size]

                img_name = f'{os.path.splitext(img_file)[0]}_crop_{crop_count + 1}{img_extension}'
                mask_name = f'{os.path.splitext(mask_file)[0]}_crop_{crop_count + 1}{mask_extension}'

                cv2.imwrite(os.path.join(training_images_folder, img_name), cropped_img)
                cv2.imwrite(os.path.join(training_masks_folder, mask_name), cropped_mask)

                crop_count += 1

    print("[INFO] Cropping complete.")
