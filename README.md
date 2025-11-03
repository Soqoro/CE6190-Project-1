# CE6190-Project-1

# Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Data Preparation:

```bash
# Split Kvasir Dataset 80/10/10 using symlinks
python scripts/split_kvasir_dirs.py \
  --src data/kvasir \
  --out data/kvasir_split \
  --train 0.8 --val 0.1 --test 0.1 \
  --seed 0 --mode symlink --write_json
```

Make splits once:

```bash
python scripts/make_kvasir_splits.py
python scripts/make_voc_splits.py
python scripts/make_parts_splits.py
```

Kvasir (binary; monitors val/dice):

```bash
python -m src.engine.run --cfg configs/kvasir/unet.yaml --seed 0
python -m src.engine.run --cfg configs/kvasir/deeplab.yaml --seed 0
python -m src.engine.run --cfg configs/kvasir/segformer.yaml --seed 0
```

VOC (multiclass; monitors val/miou):

```bash
python -m src.engine.run --cfg configs/voc/unet.yaml --seed 0
python -m src.engine.run --cfg configs/voc/deeplab.yaml --seed 0
python -m src.engine.run --cfg configs/voc/segformer.yaml --seed 0
```

PASCAL‑Parts (multiclass; monitors val/miou):

```bash
python -m src.engine.run --cfg configs/parts/unet.yaml --seed 0
python -m src.engine.run --cfg configs/parts/deeplab.yaml --seed 0
python -m src.engine.run --cfg configs/parts/segformer.yaml --seed 0
```

Switching % labels via split_file:

```bash
# Launcher picks base cfg, injects split_file + seed, and unique out_dir:
python -m scripts.launch --ds voc --model deeplab --seed 1 --pct 10
python -m scripts.launch --ds kvasir --model unet --seed 2 --pct 5
python -m scripts.launch --ds parts --model segformer --seed 0 --pct 100
```