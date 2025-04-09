import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import re  # For extracting numbers from filenames

# Set up paths
data_path = os.path.join("Model_Training", "Test_Dataset")
generated_path = os.path.join("Model_Validation")
ground_truth_path = os.path.join(data_path, "Test_Masks")
generated_mask_path = os.path.join(generated_path, "Generated_Masks")

# Create the output directory
output_path = os.path.join("Model_Validation", "Overlapped_Images")
os.makedirs(output_path, exist_ok=True)

# Allowed image formats
valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Collect ground truth and generated mask files
ground_truth_files = sorted([f for f in os.listdir(ground_truth_path) if f.lower().endswith(valid_extensions)])
sam_mask_files = sorted([f for f in os.listdir(generated_mask_path) if f.lower().endswith(valid_extensions)])

# Function to extract the numeric part of a filename
def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find first number in filename
    return int(match.group()) if match else None  # Return number if found, otherwise None

# Create a dictionary for generated masks with extracted numbers
sam_file_dict = {extract_number(f): f for f in sam_mask_files if extract_number(f) is not None}

# Initialize lists to store metric values
dice_scores, iou_scores, precision_scores, recall_scores, f1_scores, mcc_scores, accuracy_scores = [], [], [], [], [], [], []

# Process images
for gt_file in ground_truth_files:
    gt_number = extract_number(gt_file)

    if gt_number is None:
        print(f"Could not extract number from: {gt_file}. Skipping...")
        continue

    # Find the corresponding SAM mask by number
    sam_file = sam_file_dict.get(gt_number)

    if not sam_file:
        print(f"No match found for: {gt_file} (expected number: {gt_number})")
        continue

    # Load images
    try:
        gt_mask = imageio.imread(os.path.join(ground_truth_path, gt_file))
        sam_mask = imageio.imread(os.path.join(generated_mask_path, sam_file))

        # Convert to grayscale if multi-channel
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0]
        if len(sam_mask.shape) == 3:
            sam_mask = sam_mask[:, :, 0]

    except Exception as e:
        print(f"!!! Error loading {gt_file} or {sam_file}: {e} !!!")
        continue

    # Validate shape
    if gt_mask.shape != sam_mask.shape:
        print(f"Shape mismatch: {gt_file} and {sam_file}. Skipping...")
        continue

    # Generate overlay
    height, width = gt_mask.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # Color-coded overlay
    overlay[(gt_mask > 0) & (sam_mask > 0)] = [0, 255, 0]  # Green: Agreement (True Positive)
    overlay[(gt_mask > 0) & (sam_mask == 0)] = [255, 0, 0]  # Red: Ground Truth only (False Negative)
    overlay[(gt_mask == 0) & (sam_mask > 0)] = [0, 0, 255]  # Blue: Generated Mask only (False Positive)

    # Compute Metrics
    TP = np.sum((gt_mask > 0) & (sam_mask > 0))  # True Positives
    FP = np.sum((gt_mask == 0) & (sam_mask > 0))  # False Positives
    FN = np.sum((gt_mask > 0) & (sam_mask == 0))  # False Negatives
    TN = np.sum((gt_mask == 0) & (sam_mask == 0))  # True Negatives

    dice_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    iou_score = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # MCC Calculation with safeguard to avoid NaN (division by zero)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN) - (FP * FN)) / denominator if denominator != 0 else 0
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    # Store scores for averaging
    dice_scores.append(dice_score)
    iou_scores.append(iou_score)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1_score)
    accuracy_scores.append(accuracy)

    # Display Metrics
    metrics_text = f"Dice: {dice_score:.4f} | IoU: {iou_score:.4f} | Precision: {precision:.4f}\nRecall: {recall:.4f} | F1: {f1_score:.4f} | Accuracy: {accuracy:.4f}"

    # Plot overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Overlay for {gt_file}")
    plt.axis('off')

    # Add Metrics text
    plt.text(0.5, -0.05, metrics_text, ha='center', va='top', fontsize=12, color='black',
             transform=plt.gca().transAxes)

    # Save as PNG
    output_file = os.path.join(output_path, f"overlay_{gt_number}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    print(f" Saved overlay: {output_file} ")

    plt.close()  # Free memory

# Compute and print average metrics
num_images = len(dice_scores)
if num_images > 0:
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    print("\n📊 **Average Metrics Across All Images:**")
    print(f"🔹 Dice Score: {avg_dice:.4f}")
    print(f"🔹 IoU Score: {avg_iou:.4f}")
    print(f"🔹 Precision: {avg_precision:.4f}")
    print(f"🔹 Recall: {avg_recall:.4f}")
    print(f"🔹 F1 Score: {avg_f1:.4f}")
    print(f"🔹 Accuracy: {avg_accuracy:.4f}")

print(f"\n🎉 All overlays saved in: {output_path}")
