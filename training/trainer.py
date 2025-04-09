import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch.nn.functional as F
import os

# ===========================
# Loss Functions
# ===========================

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Ensure outputs are in [0,1]
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

class BCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)  # For binary segmentation, use sigmoid
        true_pos = torch.sum(inputs * targets)
        false_pos = torch.sum((1 - targets) * inputs)
        false_neg = torch.sum(targets * (1 - inputs))

        tversky_index = (true_pos + smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + smooth)
        return 1 - tversky_index

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)  # For binary segmentation, use sigmoid
        true_pos = torch.sum(inputs * targets)
        false_pos = torch.sum((1 - targets) * inputs)
        false_neg = torch.sum(targets * (1 - inputs))

        tversky_index = (true_pos + smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + smooth)
        focal_tversky = (1 - tversky_index) ** self.gamma
        return focal_tversky

def intersection_over_union(predicted, target, threshold=0.5):
    predicted = predicted > threshold
    target = target > threshold
    intersection = torch.sum(predicted & target).float()
    union = torch.sum(predicted | target).float()
    return (intersection / (union + 1e-6)).item()

# ===========================
# Dynamic Loss Function Class
# ===========================

class DynamicLoss(torch.nn.Module):
    def __init__(self, small_roi_thresh=0.0625, large_roi_thresh=0.0625):
        super(DynamicLoss, self).__init__()
        self.small_roi_thresh = small_roi_thresh
        self.large_roi_thresh = large_roi_thresh
        self.bce_loss = BCEWithLogitsLoss()
        self.tversky_loss = TverskyLoss()
        self.focal_tversky_loss = FocalTverskyLoss()

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        sum_area = torch.sum(prob > 0.5) / prob.numel()

        if sum_area == 0:  # Empty ROI
            return self.bce_loss(output, target)
        elif sum_area < self.small_roi_thresh:  # Small ROI
            return self.bce_loss(output, target) + self.tversky_loss(output, target)
        else:  # Large ROI
            return self.tversky_loss(output, target) + self.focal_tversky_loss(output, target)

# ===========================
# EarlyStopping
# ===========================

class EarlyStopping:
    def __init__(self, patience=5, delta=0, threshold=0.05):
        self.patience = patience
        self.delta = delta
        self.threshold = threshold
        self.best_loss = None
        self.epochs_without_improvement = 0
        self.stopped_early = False

    def check_early_stop(self, val_loss):
        if val_loss < self.threshold:
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                self.stopped_early = True
                print(f"Early stopping activated.")
                return True
        return False

# ===========================
# Save Sample Images
# ===========================

def save_sample_images(fold, epoch, images, masks, outputs):
    fold_dir = os.path.join("Sample_Images", f"fold{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    for i in range(min(3, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy().transpose(1, 2, 0)
        output = outputs[i].cpu().numpy().transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[1].imshow(mask)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(output)
        axes[2].set_title("Predicted Mask")

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/fold{fold + 1}_epoch{epoch + 1}_sample{i + 1}.png")
        plt.close()

def train_model(model, dataset, kf, batch_size, num_epochs, early_stopping, model_name=None):
    loss_fn = DynamicLoss(small_roi_thresh=0.0625, large_roi_thresh=0.0625)

    fold_train_losses = []
    fold_val_losses = []
    fold_val_ious = []
    fold_val_dices = []

    os.makedirs("Sample_Images", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nStarting Fold {fold + 1}")

        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()

        train_losses, val_losses, val_ious, val_dices = [], [], [], []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images, masks = images.float(), masks.float()
                optimizer.zero_grad()
                outputs = model(images)
                outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)

                loss = loss_fn(outputs_resized, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            ious, dices = [], []
            with torch.no_grad():
                for images, masks, _ in tqdm(val_loader, desc="Validation"):
                    images, masks = images.float(), masks.float()
                    outputs = model(images)
                    outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)
                    loss = loss_fn(outputs_resized, masks)
                    val_loss += loss.item()

                    fold_ious = [intersection_over_union(torch.sigmoid(output), mask) for output, mask in zip(outputs_resized, masks)]
                    fold_dices = [1 - dice_loss(output, mask).item() for output, mask in zip(outputs_resized, masks)]
                    ious.extend(fold_ious)
                    dices.extend(fold_dices)

                    if epoch % 5 == 0:
                        save_sample_images(fold, epoch, images, masks, torch.sigmoid(outputs_resized))

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

            mean_iou = np.mean(ious)
            mean_dice = np.mean(dices)
            val_ious.append(ious)
            val_dices.append(dices)

            if early_stopping.check_early_stop(val_loss):
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        # Save Model per Fold
        fold_model_path = os.path.join("output", f"{model_name}_fold_{fold + 1}.pth")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Saved model for Fold {fold + 1} at: {fold_model_path}")

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_val_ious.append(val_ious)
        fold_val_dices.append(val_dices)

    return fold_train_losses, fold_val_losses, fold_val_ious, fold_val_dices

# Function to plot training results
def plot_metrics(fold_train_losses, fold_val_losses, fold_val_dices, fold_val_ious):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training Loss
    for fold in range(len(fold_train_losses)):
        axes[0, 0].plot(fold_train_losses[fold], label=f"Train Loss - Fold {fold+1}")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Validation Loss
    for fold in range(len(fold_val_losses)):
        axes[0, 1].plot(fold_val_losses[fold], label=f"Val Loss - Fold {fold+1}")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Dice Score
    for fold in range(len(fold_val_dices)):
        axes[1, 0].plot(np.mean(fold_val_dices[fold], axis=0), label=f"Dice Score - Fold {fold+1}")
    axes[1, 0].set_title("Dice Score")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Dice Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # IoU Score
    for fold in range(len(fold_val_ious)):
        axes[1, 1].plot(np.mean(fold_val_ious[fold], axis=0), label=f"IoU - Fold {fold+1}")
    axes[1, 1].set_title("IoU Score")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
