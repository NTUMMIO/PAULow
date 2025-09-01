import os
import shutil
import numpy as np
import imageio.v2 as imageio

def evaluate_and_save_best_model():
    # Define paths
    gt_path = os.path.join("utils/temp_files/Model_Training", "Test_Dataset", "Masks")
    generated_root = os.path.join("utils/temp_files/Model_Validation/Generated_Masks")
    model_path = os.path.join("utils/temp_files", "output")
    save_path = os.path.join("Use_Model", "saved_models")
    os.makedirs(save_path, exist_ok=True)

    # Valid image formats
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    # Ground truth list
    gt_files = [f for f in os.listdir(gt_path) if f.lower().endswith(valid_exts)]

    # Store scores per fold
    dice_scores = {}
    for fold in range(1, 6):
        fold_name = f"fold_{fold}"
        gen_path = os.path.join(generated_root, fold_name)
        if not os.path.exists(gen_path):
            print(f"[ERROR] Model Generated Masks folder not found: {gen_path}")
            continue

        fold_dice = []
        gen_files = [f for f in os.listdir(gen_path) if f.lower().endswith(valid_exts)]

        for gen_file in gen_files:
            # Extract numeric part from filename (e.g., image16.tif -> 16)
            num = ''.join(filter(str.isdigit, gen_file))

            # Match GT file with same number
            matched_gt = next((g for g in gt_files if num in g), None)
            if not matched_gt:
                print(f"[ERROR] Ground truth not found for {gen_file}. Skipping.")
                continue

            try:
                gt_img = imageio.imread(os.path.join(gt_path, matched_gt))
                pred_img = imageio.imread(os.path.join(gen_path, gen_file))

                # Convert to grayscale if needed
                if len(gt_img.shape) == 3:
                    gt_img = gt_img[:, :, 0]
                if len(pred_img.shape) == 3:
                    pred_img = pred_img[:, :, 0]

                if gt_img.shape != pred_img.shape:
                    print(f"[ERROR] Shape mismatch: {matched_gt} vs {gen_file}. Skipping...")
                    continue

                # Compute Dice
                TP = np.sum((gt_img > 0) & (pred_img > 0))
                FP = np.sum((gt_img == 0) & (pred_img > 0))
                FN = np.sum((gt_img > 0) & (pred_img == 0))

                dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
                fold_dice.append(dice)

            except Exception as e:
                print(f"[ERROR] Error comparing {gen_file}: {e}")
                continue

        if fold_dice:
            mean_dice = np.mean(fold_dice)
            dice_scores[fold] = mean_dice
            print(f"[INFO] Fold {fold} - Avg Dice: {mean_dice:.4f}")
        else:
            print(f"[ERROR] No valid predictions in {fold_name}.")

    # Find best fold
    if not dice_scores:
        print("[ERROR] No valid Dice scores found. Exiting.")
        return

    best_fold = max(dice_scores, key=dice_scores.get)
    print(f"[INFO] Best Fold: fold_{best_fold} with Dice: {dice_scores[best_fold]:.4f}")

    # Find and copy model
    best_model_file = next(
        (f for f in os.listdir(model_path) if f.endswith(f"_fold_{best_fold}.pth")),
        None
    )

    if best_model_file:
        new_model_name = best_model_file.replace(f"_fold_{best_fold}", "")
        shutil.copy(os.path.join(model_path, best_model_file), os.path.join(save_path, new_model_name))
        print(f"[INFO] Best model saved as: {new_model_name} in {save_path}")
    else:
        print("[ERROR] No best model found to save!")

