import os
import torch

def save_model(model, path):
    """
    Save the model's state_dict to the given file path.
    
    Args:
    - model: The model to save.
    - path: The file path to save the model.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model's state_dict
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Dice Coefficient for evaluation
def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)
