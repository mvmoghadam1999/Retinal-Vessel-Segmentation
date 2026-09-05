from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms


def load_image_and_mask(image_path, mask_path, image_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image)
    mask_tensor = (transform(mask) > 0).float()
    return image, mask, image_tensor, mask_tensor


def predict_probability(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)[0, 0]
    return probs.cpu().numpy()


def find_best_threshold(
    pred_probs,
    ground_truth,
    thresholds=np.linspace(0.1, 0.9, 20),
):
    gt_bin = (ground_truth > 0).astype(np.uint8)
    best_threshold, best_dice = 0.5, 0.0

    for threshold in thresholds:
        pred_bin = (pred_probs > threshold).astype(np.uint8)
        dice = f1_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=0)

        if dice > best_dice:
            best_dice = dice
            best_threshold = float(threshold)

    return best_threshold, best_dice


def evaluate_checkpoint(
    model, checkpoint_path, image_tensor, ground_truth, device
):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))

    probabilities = predict_probability(model, image_tensor, device)
    threshold, dice = find_best_threshold(probabilities, ground_truth)

    binary_mask = ((probabilities > threshold).astype(np.uint8) * 255)

    return {
        "probabilities": probabilities,
        "threshold": threshold,
        "dice": dice,
        "binary_mask": binary_mask,
    }
