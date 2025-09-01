import os
import shutil
import random

def split_dataset(images_folder, masks_folder, train_output_folder, test_output_folder, split_ratio=0.8):
    """
    Split dataset into training and testing sets.

    Args:
        images_folder (str): Path to input images folder.
        masks_folder (str): Path to input masks folder.
        train_output_folder (str): Path to save training set.
        test_output_folder (str): Path to save test set.
        split_ratio (float): Ratio of training samples (default: 0.8).
    """

    # Create necessary directories
    original_training_images_folder = os.path.join(train_output_folder, 'Images')
    original_training_masks_folder = os.path.join(train_output_folder, 'Masks')
    test_images_folder = os.path.join(test_output_folder, 'Images')
    test_masks_folder = os.path.join(test_output_folder, 'Masks')

    os.makedirs(original_training_images_folder, exist_ok=True)
    os.makedirs(original_training_masks_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_masks_folder, exist_ok=True)

    # Get image and mask filenames
    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))

    # Make sure the number of images and masks are the same
    assert len(image_files) == len(mask_files), "[ERROR] Number of images and masks do not match!"

    # Shuffle data indices
    total_images = len(image_files)
    indices = list(range(total_images))
    random.shuffle(indices)

    # Train/Test split
    num_train_images = int(split_ratio * total_images)
    train_indices = indices[:num_train_images]
    test_indices = indices[num_train_images:]

    # Copy training files
    for i in train_indices:
        shutil.copy(os.path.join(images_folder, image_files[i]), os.path.join(original_training_images_folder, image_files[i]))
        shutil.copy(os.path.join(masks_folder, mask_files[i]), os.path.join(original_training_masks_folder, mask_files[i]))

    # Copy testing files
    for i in test_indices:
        shutil.copy(os.path.join(images_folder, image_files[i]), os.path.join(test_images_folder, image_files[i]))
        shutil.copy(os.path.join(masks_folder, mask_files[i]), os.path.join(test_masks_folder, mask_files[i]))

    print(f"[INFO] Training Samples: {len(train_indices)}, Testing Samples: {len(test_indices)}")
