from __future__ import annotations
import argparse, yaml, os
from copy import deepcopy
from src.engine.run import main as train_main

BASE = {
    ("kvasir","unet"):      "configs/kvasir/unet.yaml",
    ("kvasir","deeplab"):   "configs/kvasir/deeplab.yaml",
    ("kvasir","segformer"): "configs/kvasir/segformer.yaml",
    ("voc","unet"):         "configs/voc/unet.yaml",
    ("voc","deeplab"):      "configs/voc/deeplab.yaml",
    ("voc","segformer"):    "configs/voc/segformer.yaml",
    ("parts","unet"):       "configs/parts/unet.yaml",
    ("parts","deeplab"):    "configs/parts/deeplab.yaml",
    ("parts","segformer"):  "configs/parts/segformer.yaml",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["kvasir","voc","parts"])
    ap.add_argument("--model", required=True, choices=["unet","deeplab","segformer"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pct", type=int, default=10)
    args = ap.parse_args()

    cfg_path = BASE[(args.ds, args.model)]
    cfg = yaml.safe_load(open(cfg_path))

    if args.ds in ("kvasir","voc","parts"):
        cfg["seed"] = args.seed
        cfg["split_file"] = f"splits/{args.ds}/seed{args.seed}_{args.pct}.json"
 
    # unique out_dir
    cfg["out_dir"] = os.path.join(cfg["out_dir"], f"s{args.seed}_p{args.pct}")

    # Write a temp yaml and reuse the trainer
    tmp = ".tmp_run.yaml"
    with open(tmp, "w") as f: yaml.safe_dump(cfg, f)
    train_main(tmp)
    os.remove(tmp)

if __name__ == "__main__":
    main()
