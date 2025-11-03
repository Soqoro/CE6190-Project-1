from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def make_transforms(task: str, image_size: int):
    """
    Build train/val Albumentations pipelines.

    Args:
        task: one of {"medical", "scene", "parts"}
        image_size: final square size (crop/pad)
    Returns:
        (train_transform, val_transform)
    """
    task = task.lower()

    if task == "medical":
        # Moderate spatial + light blur; preserve structure
        train = A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.3), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, border_mode=0, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    elif task in {"scene", "parts"}:
        # VOC / Parts style: resize, pad, then random crop to image_size
        train = A.Compose([
            A.LongestMaxSize(max_size=int(image_size * 1.2)),
            A.PadIfNeeded(image_size, image_size, border_mode=0),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown task for transforms: {task}")

    val = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ])

    return train, val
