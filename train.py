import torch

from src.data import create_dataloaders
from src.losses import DiceBCELoss
from src.models import build_model
from src.training import train
from src.visualization import plot_training_history


IMAGE_DIR = "/path/to/images_masks/BV_segmentation_dataset/original"
MASK_DIR = "/path/to/images_masks/BV_segmentation_dataset/mask/blood-vessel"

IMAGE_SIZE = (256, 256)
TRAIN_RATIO = 0.80
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 4

ARCHITECTURE = "Unet"
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"

LEARNING_RATE = 1e-4
EPOCHS = 50
CHECKPOINT_PATH = "weights/best_Unet_resnet18.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        IMAGE_DIR, MASK_DIR,
        image_size=IMAGE_SIZE,
        train_ratio=TRAIN_RATIO,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        seed=42,
    )

    model = build_model(
        architecture=ARCHITECTURE,
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    ).to(device)

    history = train(
        model,
        train_loader,
        val_loader,
        DiceBCELoss(),
        torch.optim.Adam(model.parameters(), lr=LEARNING_RATE),
        device,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
    )

    plot_training_history(history)


if __name__ == "__main__":
    main()
