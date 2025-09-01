import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import os

from utils.model.unet import AttentionUNet

# Loss Functions

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
        inputs = torch.sigmoid(inputs)
        true_pos = torch.sum(inputs * targets)
        false_pos = torch.sum((1 - targets) * inputs)
        false_neg = torch.sum(targets * (1 - inputs))

        tversky_index = (true_pos + smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + smooth
        )
        return 1 - tversky_index
class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        true_pos = torch.sum(inputs * targets)
        false_pos = torch.sum((1 - targets) * inputs)
        false_neg = torch.sum(targets * (1 - inputs))

        tversky_index = (true_pos + smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + smooth
        )
        return (1 - tversky_index) ** self.gamma

#Dynamic Loss Function
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

# Save Sample Images

def save_sample_images(fold, epoch, images, masks, outputs, model_name):
    fold_dir = os.path.join("Sample_Images", model_name, f"fold{fold + 1}")
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

# Training Loop

def train_model(model, dataset, kf, batch_size, num_epochs, learning_rate, model_name=None):
    """
    Train model with cross-validation.

    Args:
        model (nn.Module): Model class (not instance).
        dataset (Dataset): Torch dataset.
        kf (KFold): KFold cross-validator.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs.
        learning_rate (float, optional): Learning rate. Default = 1e-5.
        model_name (str, optional): Base name for saving model checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = DynamicLoss(small_roi_thresh=0.0625, large_roi_thresh=0.0625)

    fold_train_losses = []
    fold_val_losses = []

    os.makedirs("utils/temp_files/output", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        model_instance = model().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
        )

        # ✅ learning rate now comes from argument
        optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")

        with tqdm(total=num_epochs, desc=f"FOLD {fold + 1}", ncols=125) as fold_bar:
            for epoch in range(num_epochs):
                # Training
                model_instance.train()
                epoch_loss = 0
                for images, masks, _ in train_loader:
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    optimizer.zero_grad()
                    outputs = model_instance(images)
                    outputs_resized = F.interpolate(
                        outputs, size=masks.shape[2:], mode="bilinear", align_corners=True
                    )

                    loss = loss_fn(outputs_resized, masks)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                train_losses.append(epoch_loss)

                # Validation
                model_instance.eval()
                val_loss = 0
                with torch.no_grad():
                    for images, masks, _ in val_loader:
                        images = images.float().to(device)
                        masks = masks.float().to(device)

                        outputs = model_instance(images)
                        outputs_resized = F.interpolate(
                            outputs, size=masks.shape[2:], mode="bilinear", align_corners=True
                        )

                        loss = loss_fn(outputs_resized, masks)
                        val_loss += loss.item()

                        if epoch % 5 == 0:
                            save_sample_images(
                                fold, epoch, images.cpu(), masks.cpu(),
                                torch.sigmoid(outputs_resized).cpu(), model_name
                            )

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    fold_model_path = os.path.join(
                        "utils/temp_files/output", f"{model_name}_fold_{fold + 1}.pth"
                    )
                    torch.save(model_instance.state_dict(), fold_model_path)

                # Update progress bar
                fold_bar.set_postfix(
                    epoch=epoch + 1,
                    train_loss=f"{epoch_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )
                fold_bar.update(1)

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

    print("Cross-Validation Complete!\n")
    return fold_train_losses, fold_val_losses