"""
Usage:
    python3 predict.py path/to/image.jpg

Loads aircraft_classifier.pt and prints the predicted class + confidence
for every class, for a single input image.
"""
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from mobilenetv2_def import mobilenetv2

if len(sys.argv) != 2:
    print("Usage: python3 predict.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

ckpt = torch.load("aircraft_classifier.pt", map_location="cpu", weights_only=False)
class_names = ckpt["class_names"]
img_size = ckpt["img_size"]
mean, std = ckpt["mean"], ckpt["std"]

model = mobilenetv2(num_classes=1000)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

ranked = sorted(zip(class_names, probs.tolist()), key=lambda p: -p[1])
print(f"\nPredictions for {image_path}:")
for name, p in ranked:
    print(f"  {name:8s} {p*100:5.1f}%")
print(f"\n=> Best guess: {ranked[0][0]} ({ranked[0][1]*100:.1f}% confidence)")
