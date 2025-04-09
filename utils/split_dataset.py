import os
import shutil
import random

def split_dataset():
    # Define paths
    model_training_folder = 'Model_Training'
    training_dataset_folder = os.path.join(model_training_folder, 'Training_Dataset')
    test_dataset_folder = os.path.join(model_training_folder, 'Test_Dataset')
    
    original_training_images_folder = os.path.join(training_dataset_folder, 'Original_Training_Images')
    original_training_masks_folder = os.path.join(training_dataset_folder, 'Original_Training_Masks')
    test_images_folder = os.path.join(test_dataset_folder, 'Test_Images')
    test_masks_folder = os.path.join(test_dataset_folder, 'Test_Masks')

    # Create necessary directories if they do not exist
    os.makedirs(original_training_images_folder, exist_ok=True)
    os.makedirs(original_training_masks_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_masks_folder, exist_ok=True)

    # Get image and mask filenames
    image_files = sorted(os.listdir('Images'))
    mask_files = sorted(os.listdir('Masks'))

    # Make sure the number of images and masks are the same
    assert len(image_files) == len(mask_files)

    total_images = len(image_files)

    # Training data (at least 10 images, preferably counting from image1)
    num_train_images = max(10, total_images - 10)  # Ensures at least 10 training images
    num_test_images = total_images - num_train_images  # Test images count

    # Split the images and masks
    for i in range(num_train_images):
        shutil.copy(os.path.join('Images', image_files[i]), os.path.join(original_training_images_folder, f'{image_files[i]}'))
        shutil.copy(os.path.join('Masks', mask_files[i]), os.path.join(original_training_masks_folder, f'{mask_files[i]}'))

    for i in range(num_train_images, total_images):
        shutil.copy(os.path.join('Images', image_files[i]), os.path.join(test_images_folder, f'{image_files[i]}'))
        shutil.copy(os.path.join('Masks', mask_files[i]), os.path.join(test_masks_folder, f'{mask_files[i]}'))

    print(f"Training data: {num_train_images} images, {num_test_images} test images")
