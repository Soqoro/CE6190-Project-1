from __future__ import annotations
import torch.nn as nn
import segmentation_models_pytorch as smp


class SMPWrapper(nn.Module):
    """Thin wrapper around segmentation_models_pytorch (SMP) models.

    Families supported:
      - "unet"
      - "deeplabv3+" / "deeplabv3plus" / "deeplab"
      - "segformer"  (use encoders like: mit_b0|mit_b1|mit_b2|mit_b3|mit_b4|mit_b5)

    Notes
    -----
    - We force `activation=None` so the model returns **logits** (required by our losses).
    - For DeepLab, you can control output stride via `encoder_output_stride` (8 or 16) in kwargs.
    - For SegFormer, ensure the encoder name is one of the `mit_b*` variants.
    """

    def __init__(
        self,
        family: str,
        encoder_name: str,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Always return logits; ignore any user-provided activation
        if "activation" in kwargs:
            kwargs.pop("activation")

        common = dict(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # logits
        )
        common.update(kwargs)

        fam = family.lower().strip()
        if fam in ("unet", "u-net"):
            self.net = smp.Unet(**common)

        elif fam in ("deeplabv3+", "deeplabv3plus", "deeplab"):
            # `encoder_output_stride` can be passed via kwargs (8 or 16).
            self.net = smp.DeepLabV3Plus(**common)

        elif fam == "segformer":
            # Guardrail: typical encoders are mit_b0...mit_b5
            if not encoder_name.lower().startswith("mit_"):
                raise ValueError(
                    f"SegFormer encoders should be 'mit_b*' (got '{encoder_name}'). "
                    "Try one of: mit_b0|mit_b1|mit_b2|mit_b3|mit_b4|mit_b5."
                )
            self.net = smp.Segformer(**common)

        else:
            raise ValueError(
                f"Unsupported family: '{family}'. "
                "Use one of: 'unet', 'deeplabv3+', 'segformer'."
            )

    def forward(self, x):
        # Return logits
        return self.net(x)
