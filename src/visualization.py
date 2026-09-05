from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_history(history, output_dir="outputs/training_curves"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in [
        "Loss", "Accuracy", "F1", "Dice",
        "IoU", "Precision", "Recall", "mAP"
    ]:
        train_values = [x[metric] for x in history["train"]]
        val_values = [x[metric] for x in history["val"]]

        plt.figure(figsize=(8, 5))
        plt.plot(train_values, label=f"Train {metric}")
        plt.plot(val_values, label=f"Validation {metric}")
        plt.title(f"{metric} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric.lower()}_curve.png", dpi=200)
        plt.close()


def plot_model_comparison(
    image, ground_truth, predictions,
    output_path="outputs/model_predictions_segmentation.pdf"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(predictions.items())
    fig, axes = plt.subplots(
        1, 2 + len(items),
        figsize=(4 * (2 + len(items)), 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Input Image")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")

    for axis, (name, result) in zip(axes[2:], items):
        axis.imshow(result["binary_mask"], cmap="gray")
        axis.set_title(
            f"{name}\nThreshold={result['threshold']:.3f}\n"
            f"Dice={result['dice']:.4f}"
        )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
