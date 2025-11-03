from __future__ import annotations
import os
from src.data.splits import make_percentage_split

root = "data/VOCdevkit/VOC2012"
with open(os.path.join(root, "ImageSets", "Segmentation", "train.txt")) as f:
    all_ids = [x.strip() for x in f.readlines()]

os.makedirs("splits/voc", exist_ok=True)
for seed in [0, 1, 2]:
    for pct in [1, 5, 10, 100]:
        make_percentage_split(all_ids, pct, seed, f"splits/voc/seed{seed}_{pct}.json")
print("VOC splits written to splits/voc/")
