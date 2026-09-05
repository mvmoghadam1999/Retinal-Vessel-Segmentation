# Retinal Vessel Segmentation

A modular deep-learning project for retinal blood-vessel segmentation using `segmentation-models-pytorch`.

## Workflows

- **Training:** paired retinal images/masks → 80/20 train/validation split → U-Net training → metrics → best checkpoint → curves.
- **Model comparison:** load multiple trained models → run inference on the same image → find the best threshold against the supplied ground truth → compare masks in one PDF.

## Features

- Binary retinal-vessel segmentation
- Reusable Dataset/DataLoader
- Reproducible 80/20 split
- Dice + BCE loss
- Accuracy, Precision, Recall, F1, Dice, IoU and Average Precision
- Best-checkpoint saving by validation Dice
- Training/validation curves
- One model factory for U-Net, U-Net++, MAnet, DeepLabV3 and DeepLabV3+
- One reusable inference and threshold-search pipeline
- Side-by-side model comparison
- Configurable paths instead of hard-coded dataset URLs

## Structure

```text
retinal-vessel-segmentation/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── training.py
│   ├── inference.py
│   └── visualization.py
├── train.py
├── evaluate_models.py
├── requirements.txt
├── .gitignore
├── weights/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── README.md
```

## Dataset

Dataset source:

```text
DATASET_LINK_PLACEHOLDER
```

Expected layout:

```text
images_masks/
└── BV_segmentation_dataset/
    ├── original/
    └── mask/
        └── blood-vessel/
```

The dataset and trained checkpoints are intentionally not included in GitHub.

## Installation

```bash
python -m venv .venv
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

`segmentation-models-pytorch` provides the segmentation architectures used here, including U-Net, U-Net++, MAnet, DeepLabV3 and DeepLabV3+. citeturn0search0turn0search2

## Training

Edit the paths in `train.py`:

```python
IMAGE_DIR = "/path/to/images_masks/BV_segmentation_dataset/original"
MASK_DIR = "/path/to/images_masks/BV_segmentation_dataset/mask/blood-vessel"
```

Run:

```bash
python train.py
```

Default configuration:

```text
Image size       : 256 x 256
Train split      : 80%
Validation split : 20%
Train batch size : 8
Validation batch : 4
Epochs           : 50
Optimizer        : Adam
Learning rate    : 1e-4
Model            : U-Net
Encoder          : ResNet18
```

A fixed random seed is used for the split. PyTorch's `random_split` creates non-overlapping subsets and supports a seeded generator for reproducibility. citeturn0search8

## Model Comparison

Put your trained checkpoints in `weights/` and edit `MODEL_CONFIGS` in `evaluate_models.py`.

Each model is configured once:

```python
{
    "name": "Unet++",
    "architecture": "UnetPlusPlus",
    "encoder_name": "resnet18",
    "checkpoint": "weights/your_checkpoint.pth",
}
```

Then:

```bash
python evaluate_models.py
```

The script reuses the same inference/threshold code for every configured model and saves:

```text
outputs/model_predictions_segmentation.pdf
```

## Threshold Search

`find_best_threshold()` is defined once in `src/inference.py`.

The default candidates are 20 thresholds from 0.10 to 0.90. The threshold maximizing Dice/F1 against the supplied ground-truth mask is selected.

For fair benchmarking on unseen data, threshold tuning should normally be performed on a validation/calibration set rather than on the final test sample itself.

## Metrics

- Loss
- Dice
- IoU / Jaccard
- Accuracy
- Precision
- Recall
- F1
- Average Precision (`mAP` in the original notebook terminology)

## Cleanup / Refactoring

The original notebook repeated the following for every model:

- imports
- transform creation
- device selection
- model construction
- checkpoint loading
- inference
- threshold search
- ground-truth preprocessing

These are now centralized.

Instead of six almost-identical blocks, the comparison workflow is:

```text
MODEL_CONFIGS
     ↓
build model
     ↓
load checkpoint
     ↓
predict probabilities
     ↓
find threshold
     ↓
create binary mask
     ↓
compare models
```

The original training model was U-Net while its checkpoint filename contained `DeepLabV3+`; the cleaned project uses architecture-consistent checkpoint naming.

The original MAnet block calculated a best threshold but then applied a hard-coded `0.05`; the cleaned workflow consistently applies the selected threshold.

Optional experimental losses (Focal, IoU and SSIM) are kept separate from the main training loss so the core pipeline remains clear.

## Future Improvements

- Add train-only augmentation
- Add a separate held-out test set
- Add early stopping and LR scheduling
- Add TensorBoard/W&B logging
- Add ROC and Precision-Recall curves
- Benchmark several encoders automatically
- Add morphological post-processing
- Add ensemble segmentation
- Evaluate a complete test directory rather than one image
- Add unit tests

## Technologies

Python · PyTorch · TorchVision · Segmentation Models PyTorch · NumPy · Pillow · OpenCV · scikit-learn · Matplotlib · tqdm

## License

Add the appropriate license for your project.

## Author

Your Name
