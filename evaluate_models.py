import torch

from src.inference import evaluate_checkpoint, load_image_and_mask
from src.models import build_model
from src.visualization import plot_model_comparison


IMAGE_PATH = "/path/to/DR-HAGIS-2_org.png"
MASK_PATH = "/path/to/DR-HAGIS-2.png"
IMAGE_SIZE = (256, 256)

MODEL_CONFIGS = [
    {
        "name": "DeepLabV3",
        "architecture": "DeepLabV3",
        "encoder_name": "resnet18",
        "checkpoint": "weights/best_weight_DeepLabV3_resize_39.pth",
    },
    {
        "name": "DeepLabV3+",
        "architecture": "DeepLabV3Plus",
        "encoder_name": "resnet34",
        "checkpoint": "weights/best_weight_DeepLabV3+_resize_27.pth",
    },
    {
        "name": "Unet",
        "architecture": "Unet",
        "encoder_name": "resnet18",
        "checkpoint": "weights/best_weight_Unet_new_resize_30.pth",
    },
    {
        "name": "MAnet",
        "architecture": "MAnet",
        "encoder_name": "resnet34",
        "checkpoint": "weights/best_weight_MAnet_res34_resize_31.pth",
    },
    {
        "name": "Unet++",
        "architecture": "UnetPlusPlus",
        "encoder_name": "resnet18",
        "checkpoint": "weights/best_weight_Unet++_maskresize_29.pth",
    },
]

OUTPUT_PATH = "outputs/model_predictions_segmentation.pdf"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image, mask, image_tensor, mask_tensor = load_image_and_mask(
        IMAGE_PATH, MASK_PATH, IMAGE_SIZE
    )
    ground_truth = mask_tensor.squeeze(0).numpy()

    predictions = {}

    for config in MODEL_CONFIGS:
        print(f"\nEvaluating {config['name']}...")

        model = build_model(
            architecture=config["architecture"],
            encoder_name=config["encoder_name"],
            encoder_weights=None,
            in_channels=3,
            classes=1,
        ).to(device)

        result = evaluate_checkpoint(
            model, config["checkpoint"],
            image_tensor, ground_truth, device
        )

        predictions[config["name"]] = result

        print(
            f"Best threshold: {result['threshold']:.3f} | "
            f"Dice: {result['dice']:.4f}"
        )

    plot_model_comparison(
        image, mask, predictions, OUTPUT_PATH
    )
    print(f"\nSaved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
