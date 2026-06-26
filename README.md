# medai-pneumonia-xray

Transfer-learning baseline for **pneumonia detection in chest X-rays** using
PyTorch and a ResNet-18 backbone. Intended as a clean starting point for
research portfolios and teaching, not a clinical tool.

> **Disclaimer.** This repository is for educational and research use only.
> It must not be used for diagnosis, triage, or any clinical decision.

## What it does

- Loads chest X-ray images organized in `train/`, `val/`, `test/` folders
  (one subfolder per class: `NORMAL/`, `PNEUMONIA/`).
- Fine-tunes a pretrained ResNet-18 on the binary task.
- Reports accuracy, ROC-AUC, sensitivity, specificity, and a confusion matrix.
- Saves the best checkpoint to `runs/<run-name>/best.pt`.

## Dataset

This repo expects the **Chest X-Ray Images (Pneumonia)** dataset structure
from Kermany et al., available on Kaggle:

```
data/chest_xray/
  train/{NORMAL,PNEUMONIA}/*.jpeg
  val/{NORMAL,PNEUMONIA}/*.jpeg
  test/{NORMAL,PNEUMONIA}/*.jpeg
```

Download it from Kaggle and unzip into `data/`. The folder is gitignored.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python -m src.train --data-dir data/chest_xray --epochs 5 --batch-size 32
python -m src.predict --checkpoint runs/latest/best.pt --image path/to/xray.jpeg
```

## Project layout

```
src/
  data.py      # dataloaders + augmentations
  model.py     # ResNet-18 with a 2-class head
  train.py     # training loop + metric logging
  predict.py   # single-image inference CLI
tests/
  test_model.py
```

## Caveats

- Class imbalance in the standard split is severe. The training script uses a
  weighted sampler; do not rely on raw accuracy.
- Validation split in the public dataset is tiny (16 images). Use a stratified
  re-split for any serious evaluation.
- For real research, evaluate on an external dataset (CheXpert, NIH ChestX-ray14).

## License

MIT
