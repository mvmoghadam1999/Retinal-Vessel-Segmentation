import numpy as np
import torch
from tqdm import tqdm

from .metrics import binary_segmentation_metrics


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    values = {
        "Accuracy": [], "Precision": [], "Recall": [],
        "F1": [], "Dice": [], "IoU": [], "mAP": []
    }

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, masks).item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            batch = binary_segmentation_metrics(
                masks.cpu().numpy(),
                preds.cpu().numpy(),
                probs.cpu().numpy(),
            )
            for key, value in batch.items():
                values[key].append(value)

    if not len(loader):
        raise ValueError("DataLoader is empty.")

    result = {"Loss": total_loss / len(loader)}
    result.update({key: float(np.mean(vals)) for key, vals in values.items()})
    return result


def train(
    model, train_loader, val_loader, loss_fn, optimizer, device,
    epochs=50, checkpoint_path="weights/best_model.pth"
):
    history = {"train": [], "val": []}
    best_dice = -1.0

    for epoch in range(epochs):
        model.train()

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_loader, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"\nEpoch {epoch + 1}")
        print(
            f"Train Loss: {train_metrics['Loss']:.4f} | "
            f"Val Loss: {val_metrics['Loss']:.4f}"
        )
        for key in ["Accuracy", "Precision", "Recall", "F1", "Dice", "IoU", "mAP"]:
            print(
                f"{key}: Train {train_metrics[key]:.4f} | "
                f"Val {val_metrics[key]:.4f}"
            )

        if val_metrics["Dice"] > best_dice:
            best_dice = val_metrics["Dice"]
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Saved best model at epoch {epoch + 1} "
                f"with validation Dice: {best_dice:.4f}"
            )

    return history
