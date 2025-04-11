import os
import numpy as np
from PIL import Image

def normalize_image(image):
    """Normalize the image to [0, 255] based on the max pixel value."""
    image_array = np.array(image, dtype=np.float32)
    max_val = image_array.max()
    
    if max_val > 0:
        normalized_array = (image_array / max_val) * 255.0
    else:
        normalized_array = image_array
    
    return Image.fromarray(normalized_array.astype(np.uint8))

def process_image_file(image_path, output_folder, counter, is_mask=False):
    """Process and save an image or stack with proper extension and format."""
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Error: Could not open image {image_path}")
        return counter

    ext = os.path.splitext(image_path)[1].lower()  # Get the original extension

    # Process multi-page TIFFs
    if ext in [".tif", ".tiff"]:
        page = 0
        while True:
            try:
                base_name = f"Mask{counter}" if is_mask else f"image{counter}"
                output_path = os.path.join(output_folder, f"{base_name}{ext}")
                normalized_image = normalize_image(image)
                normalized_image.save(output_path)
                counter += 1
                page += 1
                image.seek(page)  # Try to go to the next frame
            except EOFError:
                break  # End of stack
            except Exception as e:
                print(f"Error while processing {image_path}: {e}")
                break
    else:
        # Process single image (PNG, JPG, JPEG, etc.)
        base_name = f"Mask{counter}" if is_mask else f"image{counter}"
        output_path = os.path.join(output_folder, f"{base_name}{ext}")
        normalized_image = normalize_image(image)
        normalized_image.save(output_path)
        counter += 1

    return counter

def process_all_images(input_folder="Train_Model/TRAINING_IMAGES", output_folder="utils/temp_files/Images"):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    if not files_to_process:
        print(f"No valid image files found in '{input_folder}' to process.")
        return

    counter = 1
    for filename in files_to_process:
        image_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")
        counter = process_image_file(image_path, output_folder, counter)

def process_all_masks(input_folder="Train_Model/TRAINING_MASKS", output_folder="utils/temp_files/Masks"):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    if not files_to_process:
        print(f"No valid mask files found in '{input_folder}' to process.")
        return

    counter = 1
    for filename in files_to_process:
        mask_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")
        counter = process_image_file(mask_path, output_folder, counter, is_mask=True)

process_all_images()
process_all_masks()
