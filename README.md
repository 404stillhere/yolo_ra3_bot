# RA3 YOLO training bundle

Minimal bundle for continuing RA3 YOLO training on a GPU server.

## Contents

- `scripts/voc_to_yolo_unified.py` builds YOLO `images/`, `labels/`, and `dataset.yaml` from PascalVOC XML.
- `scripts/train_ra3_allied.py` launches Ultralytics training.
- `data/datasets/ra3_unified/raw_frames/` contains JPG/XML source pairs.
- `data/datasets/ra3_unified/predefined_classes.txt` contains class names.
- `data/datasets/ra3_unified/manifest.csv` keeps stable train/val source split.
- `weights/ra3_unified_8n_more_japan_20260430_192240/best.pt` is the best current checkpoint.
- `weights/ra3_unified_8n_more_japan_20260430_192240/last.pt` is the last checkpoint from that run.

## Ubuntu setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git git-lfs
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ultralytics
```

## Train

```bash
source venv/bin/activate
python scripts/voc_to_yolo_unified.py
python scripts/train_ra3_allied.py \
  --data data/datasets/ra3_unified/dataset.yaml \
  --model weights/ra3_unified_8n_more_japan_20260430_192240/best.pt \
  --name ra3_unified_cloud_v100 \
  --epochs 200 \
  --imgsz 640 \
  --batch 32 \
  --device 0
```
