from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _rrc(image_size: int, scale=(0.7, 1.0), ratio=(0.9, 1.1)):
    try:
        return A.RandomResizedCrop(size=(image_size, image_size), scale=scale, ratio=ratio)
    except TypeError:
        return A.RandomResizedCrop(height=image_size, width=image_size, scale=scale, ratio=ratio)


def make_transforms(task: str, image_size: int):
    task = task.lower()
    image_size = int(image_size)

    if task == "medical":
        train = A.Compose([
            _rrc(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent={"x": 0.05, "y": 0.05}, rotate=(-10, 10), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    elif task in {"scene", "parts"}:
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
