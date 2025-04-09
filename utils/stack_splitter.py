import os
import numpy as np
from PIL import Image

def normalize_image(image):
    """Normalize the image to the range [0, 255] for saving."""
    image_array = np.array(image, dtype=np.float32)
    min_val, max_val = image_array.min(), image_array.max()
    
    if max_val > min_val:  # Avoid division by zero
        normalized_array = (image_array - min_val) / (max_val - min_val) * 255.0
    else:
        normalized_array = image_array  # If all values are the same, keep as is
    
    return Image.fromarray(normalized_array.astype(np.uint8))

def split_stacked_image_pillow(image_path, output_folder, page_counter, is_mask=False):
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Error: Could not open the image at {image_path}")
        return page_counter
    
    page = 0
    while True:
        try:
            if is_mask:
                output_path = os.path.join(output_folder, f"Mask{page_counter}.tiff")
            else:
                output_path = os.path.join(output_folder, f"image{page_counter}.tiff")
            
            # Normalize the image
            normalized_image = normalize_image(image)
            
            # Save the normalized image
            normalized_image.save(output_path)
            print(f"Saved: {output_path}")
            page_counter += 1
            page += 1
            
            image.seek(page)  # Move to the next page
        except EOFError:
            break  # End of TIFF stack
    
    return page_counter

def split_all_images(input_folder="stacked_images", output_folder="Images"):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)
    
    files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
    if not files_to_process:
        print(f"No TIFF files found in '{input_folder}' to process.")
        return
    
    page_counter = 1
    for filename in files_to_process:
        image_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")
        page_counter = split_stacked_image_pillow(image_path, output_folder, page_counter)

def split_all_masks(input_folder="stacked_masks", output_folder="Masks"):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)
    
    files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))]
    if not files_to_process:
        print(f"No TIFF files found in '{input_folder}' to process.")
        return
    
    page_counter = 1
    for filename in files_to_process:
        mask_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")
        page_counter = split_stacked_image_pillow(mask_path, output_folder, page_counter, is_mask=True)
