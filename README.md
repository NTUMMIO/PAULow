# PAULow
PAULow: Patch-based Attention U-Net for Low-resource learning

Segmentation is an essential tool for cell biologists and involves isolating cells or cellular features from microscopy images. An automated segmentation pipeline with high precision and accuracy can significantly reduce manual labor and subjectivity. Frequently, researchers would seek for a validated model available online and fine-tune it to meet their segmentation requirements. However, the established fine-tuning approach may involve online training or computationally intensive offline training. To address this, we propose an offline training pipeline requiring only tens of samples that are morphologically distinct from pre-training data. Specifically, we employed a patch-based attention U-Net trained with a threshold-based custom loss function.

(Currently supporting: .png, .jpg, .jpeg, .tif, .tiff)
(Capable of handling stacked images)

## Installation
1. Download python version 3.12.4

```bash
https://www.python.org/downloads/release/python-3124/
```

2. Install Libraries on terminal

    **Windows** : Press Windows + R, Type cmd and hit Enter

    **macOS** : Press Command (⌘) + Space to open Spotlight Search, Type Terminal and press Enter

    **Linux** : Press Ctrl + Alt + T

    and run this command:

```bash
pip install numpy opencv-python matplotlib imageio tifffile pillow torch torchvision scikit-learn tqdm
```

3. Run directory_setup.py to set up folder structures

## Model Training
1. "COPY" images to Train_Model/TRAINING_IMAGES and masks to Train_Model/TRAINING_MASKS
2. Run train.model.py, wait until training is completed. Best model(Highest Dicescore) will be saved at: Use_Model/saved_models as a .pth file. 

## Use Saved Model
1. "COPY" images to Use_Model/INPUT_IMAGES
2. Run use_model.py and follow the prompts on terminal. Segmented Masks can be seen in Use_Model/OUTPUT_MASKS.

## **WARNING** 
   **To prevent data from different datasets from being mixed together, train_model.py will automatically delete all images in Train_Model/TRAINING_IMAGES and Train_Model/TRAINING_MASKS.**
   
   **Please copy your dataset into these folders before training to ensure that your original dataset remains intact.**
   
   **Similarly, use_model.py will delete all images in Use_Model/INPUT_IMAGES to avoid segmenting previously used inputs.**
