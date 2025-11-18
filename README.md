# CE6190-Project-1

# Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Install datasets:

1. Kvasir-SEG: It has already been downloaded and placed in `data/kvasir/`.
2. PASCAL VOC 2012: Download from [here](https://github.com/dataset-ninja/pascal-voc-2012/blob/main/DOWNLOAD.md). Extract and place in `data/voc/`.
3. PASCAL-Person_parts: Download from [here](http://liangchiehchen.com/projects/DeepLab.html). Extract and place in `data/parts/`.
Prepare Datasets in the following structure:

```bash
data/
  voc/     {train,test,val} / {imgs,ann}
  kvasir/  {Annotations,ImageSets,JPEGImages,SegmentationClass}
  parts/   {JPEGImages,pascal_person_part_gt,splits}
```

For PASCAL-VOC, run the following to create masks:

```bash
python scripts/voc_json_to_masks.py --root data/voc --neutral_as_ignore
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

Evaluation Example:

```bash
python -m src.engine.eval_test --cfg configs/kvasir/unet.yaml --ckpt_policy best
```

Qualitative Results Visualization Example:

```bash
python -m src.engine.eval_test \
  --cfg configs/kvasir/unet.yaml --ckpt_policy best \
  --dump_dir runs/kvasir_unet_qual --save_preds --save_gts --topk 6 --save_color --save_overlay --save_orig
```   