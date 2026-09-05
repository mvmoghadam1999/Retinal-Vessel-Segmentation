from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class RetinaDataset(Dataset):
    """Paired retinal RGB images and binary vessel masks."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.image_paths = sorted(p for p in self.image_dir.iterdir() if p.is_file())
        self.mask_paths = sorted(p for p in self.mask_dir.iterdir() if p.is_file())

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}"
            )

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        image = self.image_transform(image)
        mask = mask.resize(self.image_size, resample=Image.NEAREST)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0).float()

        return image, mask


def create_dataloaders(
    image_dir: str,
    mask_dir: str,
    image_size=(256, 256),
    train_ratio=0.8,
    train_batch_size=8,
    val_batch_size=4,
    seed=42,
):
    dataset = RetinaDataset(image_dir, mask_dir, image_size)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader
