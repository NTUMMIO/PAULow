import os

def clear_images_in_folder(*folders):
    """
    Clear image and model files from given folders.
    Accepts relative or absolute paths.
    """
    # Allowed extensions
    exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif', '.pth')

    for folder in folders:
        # Expand to absolute path (works with relative input too)
        folder_path = os.path.abspath(folder)

        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    if file.lower().endswith(exts):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
            print(f"[INFO] Cleared images in: {folder_path}")
        else:
            print(f"[WARNING] Folder not found: {folder_path}")

    print("[INFO] Clearing Complete.")
