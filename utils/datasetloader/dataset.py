import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def read_image(path):
    image = Image.open(path)
    return np.array(image)

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # Update this line to include png, jpeg, jpg, tif, and tiff
        self.image_names = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpeg', '.jpg', '.tif', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        # Assuming the mask files are named like 'MaskX_crop_Y.tiff'
        mask_name = image_name.replace("image", "Mask")  # Replace 'image' with 'Mask' for mask file name
        mask_name = mask_name  # Ensure that the mask file is in the correct format

        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("L")       # Convert mask to single-channel grayscale

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # No unsqueeze(0) here

        return image, mask, image_name