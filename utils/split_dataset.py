import os
import shutil
import random

def split_dataset():
    # Define paths
    model_training_folder = 'utils/temp_files/Model_Training'
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
    image_files = sorted(os.listdir('utils/temp_files/Images'))
    mask_files = sorted(os.listdir('utils/temp_files/Masks'))

    # Make sure the number of images and masks are the same
    assert len(image_files) == len(mask_files)

    # Shuffle data indices
    total_images = len(image_files)
    indices = list(range(total_images))
    random.shuffle(indices)

    # 80/20 split
    num_train_images = int(0.8 * total_images)
    train_indices = indices[:num_train_images]
    test_indices = indices[num_train_images:]

    # Split the images and masks
    for i in train_indices:
        shutil.copy(os.path.join('utils/temp_files/Images', image_files[i]), os.path.join(original_training_images_folder, image_files[i]))
        shutil.copy(os.path.join('utils/temp_files/Masks', mask_files[i]), os.path.join(original_training_masks_folder, mask_files[i]))

    for i in test_indices:
        shutil.copy(os.path.join('utils/temp_files/Images', image_files[i]), os.path.join(test_images_folder, image_files[i]))
        shutil.copy(os.path.join('utils/temp_files/Masks', mask_files[i]), os.path.join(test_masks_folder, mask_files[i]))

    print(f"Training data: {len(train_indices)} images, Testing data: {len(test_indices)} images")

split_dataset()