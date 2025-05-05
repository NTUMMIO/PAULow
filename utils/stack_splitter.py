import os
import numpy as np
from PIL import Image

def normalize_image(image):
    """Normalize image to [0, 255] based on max pixel value, channel-aware."""
    image_array = np.array(image, dtype=np.float32)

    if image_array.ndim == 2:  # Grayscale
        max_val = image_array.max()
        if max_val > 0:
            image_array = (image_array / max_val) * 255.0
        return Image.fromarray(image_array.astype(np.uint8))

    elif image_array.ndim == 3:  # Multi-channel (e.g., RGB)
        max_val = image_array.max()
        if max_val > 0:
            image_array = (image_array / max_val) * 255.0
        return Image.fromarray(image_array.astype(np.uint8))

    else:
        raise ValueError("Unsupported image dimensions for normalization.")

def process_image_file(image_path, output_folder, counter, is_mask=False):
    """Process and save single/multi-frame, single/multi-channel TIFFs or standard images."""
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"[ERROR] Could not open image {image_path}")
        return counter

    ext = os.path.splitext(image_path)[1].lower()
    filename = os.path.basename(image_path)
    base_name_prefix = "Mask" if is_mask else "image"

    print(f"--> Processing: {filename} ...")

    def save_frame(im, counter):
        normalized = normalize_image(im)
        output_name = f"{base_name_prefix}{counter}{ext}"
        output_path = os.path.join(output_folder, output_name)
        normalized.save(output_path)
        return counter + 1

    # TIFF: check for stack and channels
    if ext in [".tif", ".tiff"]:
        try:
            image.seek(1)  # Try to go to next frame
            is_stack = True
        except EOFError:
            is_stack = False
        image.seek(0)  # Reset to first frame

        if is_stack:
            while True:
                try:
                    counter = save_frame(image, counter)
                    image.seek(image.tell() + 1)
                except EOFError:
                    break
                except Exception as e:
                    print(f"[ERROR] while processing frame in {image_path}: {e}")
                    break
        else:
            counter = save_frame(image, counter)

    else:
        # Non-TIFF single images
        counter = save_frame(image, counter)

    return counter

def process_all_images(input_folder="Train_Model/TRAINING_IMAGES", output_folder="utils/temp_files/Images"):
    print("\n[INFO] Processing Images")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    if not files_to_process:
        print(f"\n[ERROR] No valid image files found in '{input_folder}' to process.\n")
        return

    counter = 1
    for filename in files_to_process:
        image_path = os.path.join(input_folder, filename)
        counter = process_image_file(image_path, output_folder, counter)

    print("[INFO] Processing Complete")

def process_all_masks(input_folder="Train_Model/TRAINING_MASKS", output_folder="utils/temp_files/Masks"):
    print("[INFO] Processing Masks")
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    if not files_to_process:
        print(f"[ERROR] No valid mask files found in '{input_folder}' to process.")
        return

    counter = 1
    for filename in files_to_process:
        mask_path = os.path.join(input_folder, filename)
        counter = process_image_file(mask_path, output_folder, counter, is_mask=True)

    print("[INFO] Processing Complete")
