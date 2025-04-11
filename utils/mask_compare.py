import os
import shutil
import numpy as np
import imageio.v2 as imageio
import re  # For extracting fold number from filenames

def evaluate_and_save_best_model():
    # Set up paths
    data_path = os.path.join("utils/temp_files/Model_Training", "Test_Dataset")
    generated_path = os.path.join("utils/temp_files/Model_Validation")
    ground_truth_path = os.path.join(data_path, "Test_Masks")
    generated_mask_path = os.path.join(generated_path, "Generated_Masks")
    model_folds_path = os.path.join("utils/temp_files", "output")  # Path to store .pth files

    # Create the output directory for storing the best model
    best_model_path = os.path.join("Use_Model/saved_models")
    os.makedirs(best_model_path, exist_ok=True)

    # Allowed image formats
    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    # Collect ground truth and generated mask files
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_path) if f.lower().endswith(valid_extensions)])
    generated_mask_files = sorted([f for f in os.listdir(generated_mask_path) if f.lower().endswith(valid_extensions)])

    # Function to extract the numeric part of a filename (fold number)
    def extract_fold_number(filename):
        match = re.search(r'_fold_(\d+)', filename)  # Extract fold number
        return int(match.group(1)) if match else None

    # Initialize a dictionary to store Dice scores for each fold
    fold_dice_scores = {f"fold_{i}": [] for i in range(1, 6)}  # For storing Dice scores for each fold

    # Process images
    for gt_file in ground_truth_files:
        gt_number = extract_fold_number(gt_file)

        if gt_number is None:
            print(f"Could not extract fold number from: {gt_file}. Skipping...")
            continue

        generated_file = generated_mask_path + f"fold_{gt_number}"

        if not generated_file:
            print(f"No match found for: {gt_file} (expected number: {gt_number})")
            continue

        # Load images
        try:
            gt_mask = imageio.imread(os.path.join(ground_truth_path, gt_file))
            generated_mask = imageio.imread(os.path.join(generated_mask_path, generated_file))

            # Convert to grayscale if multi-channel
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask[:, :, 0]
            if len(generated_mask.shape) == 3:
                generated_mask = generated_mask[:, :, 0]

        except Exception as e:
            print(f"!!! Error loading {gt_file} or {generated_file}: {e} !!!")
            continue

        # Validate shape
        if gt_mask.shape != generated_mask.shape:
            print(f"Shape mismatch: {gt_file} and {generated_file}. Skipping...")
            continue

        # Compute Metrics
        TP = np.sum((gt_mask > 0) & (generated_mask > 0))  # True Positives
        FP = np.sum((gt_mask == 0) & (generated_mask > 0))  # False Positives
        FN = np.sum((gt_mask > 0) & (generated_mask == 0))  # False Negatives
        TN = np.sum((gt_mask == 0) & (generated_mask == 0))  # True Negatives

        dice_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        fold_dice_scores[f"fold_{gt_number}"].append(dice_score)  # Adding Dice score to the appropriate fold

    # Calculate and print the average Dice scores for each fold
    best_dice_score = -1
    best_fold = None
    best_model_file = None

    for fold, dice_scores in fold_dice_scores.items():
        avg_dice = np.mean(dice_scores)
        print(f"{fold} - Average Dice Score: {avg_dice:.4f}")
        
        # Track the highest Dice score
        if avg_dice > best_dice_score:
            best_dice_score = avg_dice
            best_fold = fold
            # Find the .pth file for the best fold (no need for model name)
            best_model_file = next((f for f in os.listdir(model_folds_path) if f.endswith(f"_fold_{best_fold.split('_')[1]}.pth")), None)

    if best_model_file:
        best_model_path = os.path.join(best_model_path, best_model_file.replace(f"_fold_{best_fold.split('_')[1]}", ""))
        shutil.copy(os.path.join(model_folds_path, best_model_file), best_model_path)
        print(f"Best model with highest Dice score saved as: {best_model_file.replace(f'_fold_{best_fold.split('_')[1]}', '')} in saved_model!")

    # Final message
    print(f"\n**Comparison complete. Best model based on Dice score saved!**")

evaluate_and_save_best_model()