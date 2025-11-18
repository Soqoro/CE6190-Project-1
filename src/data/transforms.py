from __future__ import annotations
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _rrc(size: int, *, scale=(0.5, 1.0), interpolation=cv2.INTER_LINEAR):
    """
    RandomResizedCrop:
      - Albumentations v2: size=(H,W)
      - Albumentations v1: height=H, width=W
    """
    try:
        # v2-style
        return A.RandomResizedCrop(size=(size, size), scale=scale, interpolation=interpolation)
    except Exception:
        # v1-style
        return A.RandomResizedCrop(height=size, width=size, scale=scale, interpolation=interpolation)

def _resize(size: int, *, interpolation=cv2.INTER_LINEAR):
    """
    Resize:
      - Albumentations v2: size=(H,W)
      - Albumentations v1: height=H, width=W
    """
    try:
        # v2-style
        return A.Resize(size=(size, size), interpolation=interpolation)
    except Exception:
        # v1-style
        return A.Resize(height=size, width=size, interpolation=interpolation)

def _scene_train(size: int, use_imagenet_norm: bool):
    ops = [
        _rrc(size, scale=(0.5, 1.0), interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
    ]
    if use_imagenet_norm:
        ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    ops.append(ToTensorV2())
    return A.Compose(ops)

def _scene_val(size: int, use_imagenet_norm: bool):
    ops = [
        _resize(size, interpolation=cv2.INTER_LINEAR),
    ]
    if use_imagenet_norm:
        ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    ops.append(ToTensorV2())
    return A.Compose(ops)

def _medical_train(size: int, use_imagenet_norm: bool):
    ops = [
        _rrc(size, scale=(0.8, 1.0), interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
    ]
    if use_imagenet_norm:
        ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    ops.append(ToTensorV2())
    return A.Compose(ops)

def _medical_val(size: int, use_imagenet_norm: bool):
    ops = [
        _resize(size, interpolation=cv2.INTER_LINEAR),
    ]
    if use_imagenet_norm:
        ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    ops.append(ToTensorV2())
    return A.Compose(ops)

def make_transforms(kind: str, size: int, *, use_imagenet_norm: bool = True):
    """
    kind: "scene" (VOC), "parts" (Pascal-Parts), "medical" (Kvasir)
    use_imagenet_norm: True when encoders are pretrained on ImageNet.
    """
    kind = kind.lower()
    if kind in ("scene", "parts"):
        return _scene_train(size, use_imagenet_norm), _scene_val(size, use_imagenet_norm)
    elif kind == "medical":
        return _medical_train(size, use_imagenet_norm), _medical_val(size, use_imagenet_norm)
    else:
        raise ValueError(f"Unknown transform kind: {kind}")
