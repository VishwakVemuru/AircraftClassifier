# Aircraft Classifier

**Drop in a photo, find out if it is an F-16, Su-30, Boeing 737 or Airbus A320.** A small desktop app powered by a MobileNetV2 model, trained with transfer learning on a custom photo set. Includes the full training pipeline so you can retrain it on your own aircraft.

## Quick start (Windows)
1. Download the latest zip from [Releases](https://github.com/VishwakVemuru/AircraftClassifier/releases) (or clone the repo).
2. Open the `AircraftClassifier` folder and double-click **`run.bat`**. The first run installs Python packages (torch, torchvision, pillow), which takes a few minutes.
3. Click **Choose Photo**, pick an image, click **Classify**.

Python 3 is required and must be on your PATH. On Mac or Linux run `./run.sh`. Full instructions: [AircraftClassifier/README.md](AircraftClassifier/README.md).

## Command line
```
python predict.py photo.jpg
```

## Accuracy, honestly
- Boeing 737 vs Airbus A320: reliable
- F-16 vs Su-30: weaker, they sometimes get mixed up

It was trained on a small dataset, so treat it as a learning project rather than a production classifier.

## What is inside
| File | Purpose |
|---|---|
| `predict_gui.py` | the desktop app |
| `predict.py` | command-line classifier |
| `train.py`, `split_data.py`, `evaluate.py` | training and evaluation pipeline |
| `mobilenetv2_def.py` | model architecture |
| `aircraft_classifier.pt` | the trained model |
| `data/` | the training photos |

> **Photos:** the training photos in `data/` are gathered from the web and belong to their respective owners. They are included for educational, non-commercial use only. The MIT licence covers the code, not the photos. If you own a photo and want it removed, open an issue and it will be taken down.

## Related
[AeroMatch](https://github.com/VishwakVemuru/AeroMatch) | [AeroCatalogue](https://github.com/VishwakVemuru/AeroCatalogue) | [Airfoil Bench](https://github.com/VishwakVemuru/Airfoil_Bench)

If this is useful, a star helps others find it.

Built by [Vishwak Vemuru](https://github.com/VishwakVemuru). MIT licence.
