from __future__ import annotations
import os
from src.data.splits import make_percentage_split

root = "data/PASCAL-Part"
with open(os.path.join(root, "ImageSets", "Part", "train.txt")) as f:
    all_ids = [x.strip() for x in f.readlines()]

os.makedirs("splits/parts", exist_ok=True)
for seed in [0, 1, 2]:
    for pct in [1, 5, 10, 100]:
        make_percentage_split(all_ids, pct, seed, f"splits/parts/seed{seed}_{pct}.json")
print("Parts splits written to splits/parts/")
