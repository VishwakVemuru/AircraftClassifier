# Aircraft Classifier — Desktop App

A simple app that guesses whether a photo shows an F-16, a Su-30, a
Boeing 737, or an Airbus A320, using a MobileNetV2 model trained via
transfer learning on a small custom dataset.

## How to run it (Windows)
1. Unzip this folder anywhere.
2. Double-click **`run.bat`**.
   - First run: it'll check for Python, create a small private environment,
     and install what it needs (torch, torchvision, pillow). This can take
     a few minutes — you only do this once.
   - After that, it just opens the app.
3. Click **Choose Photo**, pick a picture, click **Classify**.

If it says Python isn't found: install it from
https://www.python.org/downloads/ — when installing, make sure to check
**"Add Python to PATH"** — then double-click `run.bat` again.

## How to run it (Mac/Linux)
```
chmod +x run.sh
./run.sh
```

## What's in this folder
- `run.bat` / `run.sh` — one-click launchers (install deps, start the app)
- `predict_gui.py` — the app itself
- `aircraft_classifier.pt` — the trained model
- `mobilenetv2_def.py` — model architecture (required to load the checkpoint)
- `predict.py` — command-line version, if you'd rather script it:
  `python predict.py photo.jpg`
- `train.py`, `split_data.py`, `evaluate.py`, `data/` — the full training
  pipeline, in case you want to retrain later with more photos

## Accuracy, honestly
- Boeing 737 vs Airbus A320: reliable
- F-16 vs Su-30: weaker, they get mixed up with each other sometimes

This is a dataset-size issue, not an app issue — only 15-17 photos per
class. If you gather more fighter jet photos (varied angles: top-down,
head-on, side profile — not just similar airshow shots), drop them into
`data/f-16/` and `data/su-30/`, then run `split_data.py` followed by
`train.py` to retrain with the bigger dataset.
