from __future__ import annotations
import os, json
from src.data.splits import make_percentage_split

root = "data/Kvasir-SEG/images"
all_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(root)
                  if f.lower().endswith((".jpg", ".png"))])
os.makedirs("splits/kvasir", exist_ok=True)
for seed in [0, 1, 2]:
    for pct in [1, 5, 10, 100]:
        make_percentage_split(all_ids, pct, seed, f"splits/kvasir/seed{seed}_{pct}.json")
print("Kvasir splits written to splits/kvasir/")
