import os
import numpy as np
from PIL import Image
import re

def natural_sort_key(filename):
    """Sort filenames naturally (e.g., image2 before image10)."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]

def save_frame(im, output_folder, base_name_prefix, counter, ext):
    """Save frame without normalization."""
    output_name = f"{base_name_prefix}{counter}{ext}"
    output_path = os.path.join(output_folder, output_name)
    im.save(output_path)
    return counter + 1

def process_image_file(image_path, output_folder, counter, is_mask=False):
    ext = os.path.splitext(image_path)[1].lower()
    filename = os.path.basename(image_path)
    base_name_prefix = "mask" if is_mask else "image"

    print(f"--> Processing: {filename} ...")

    try:
        image = Image.open(image_path)
    except IOError:
        print(f"[ERROR] Could not open image {image_path}")
        return counter

    # Check if it's a multi-page / multi-frame image
    try:
        image.seek(1)
        is_stack = True
    except EOFError:
        is_stack = False
    image.seek(0)

    if is_stack:
        while True:
            try:
                counter = save_frame(image.copy(), output_folder, base_name_prefix, counter, ext)
                image.seek(image.tell() + 1)
            except EOFError:
                break
    else:
        counter = save_frame(image, output_folder, base_name_prefix, counter, ext)

    return counter

def process_all_images(input_folder, output_folder):
    print("[INFO] Processing Images")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)], key=natural_sort_key)

    if not files_to_process:
        print(f"[ERROR] No valid image files found in '{input_folder}'.")
        return

    counter = 1
    for filename in files_to_process:
        image_path = os.path.join(input_folder, filename)
        counter = process_image_file(image_path, output_folder, counter, is_mask=False)

    print("[INFO] Image Processing Complete")

def process_all_masks(input_folder, output_folder):
    print("[INFO] Processing Masks")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)], key=natural_sort_key)

    if not files_to_process:
        print(f"[ERROR] No valid mask files found in '{input_folder}'.")
        return

    counter = 1
    for filename in files_to_process:
        mask_path = os.path.join(input_folder, filename)
        counter = process_image_file(mask_path, output_folder, counter, is_mask=True)

    print("[INFO] Mask Processing Complete")
