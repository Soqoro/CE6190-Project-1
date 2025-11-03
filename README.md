# CE6190-Project-1

# Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare Datasets in the following structure:

```bash
data/
  voc/     {images,masks}
  kvasir/  {images,masks}
  parts/   {images,masks}
```

Make the dataset splits (80% train/10% val/10% test) for all datasets with 2 budgets (20%, 100%):

```bash
python scripts/dataset_split.py --data_root data --seed 0
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