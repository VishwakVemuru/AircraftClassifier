import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mobilenetv2_def import mobilenetv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("aircraft_classifier.pt", map_location="cpu", weights_only=False)
class_names = ckpt["class_names"]
img_size = ckpt["img_size"]
mean, std = ckpt["mean"], ckpt["std"]

model = mobilenetv2(num_classes=1000)
model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_ds = datasets.ImageFolder("split/val", transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Sanity check: ImageFolder class order must match training order
assert val_ds.classes == class_names, f"Class order mismatch! {val_ds.classes} vs {class_names}"

n = len(class_names)
confusion = [[0] * n for _ in range(n)]
results = []

with torch.no_grad():
    idx = 0
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu()
        confs = probs.max(dim=1).values.cpu()

        for i in range(len(labels)):
            true_c = labels[i].item()
            pred_c = preds[i].item()
            confusion[true_c][pred_c] += 1
            fname = val_ds.samples[idx][0]
            results.append({
                "file": fname,
                "true": class_names[true_c],
                "pred": class_names[pred_c],
                "confidence": round(confs[i].item(), 3),
                "correct": true_c == pred_c,
            })
            idx += 1

correct_total = sum(r["correct"] for r in results)
print(f"Overall val accuracy: {correct_total}/{len(results)} = {correct_total/len(results):.1%}\n")

print("Confusion matrix (rows=true, cols=predicted):")
header = "true\\pred".ljust(10) + "".join(c[:8].ljust(9) for c in class_names)
print(header)
for i, row in enumerate(confusion):
    print(class_names[i][:8].ljust(10) + "".join(str(v).ljust(9) for v in row))

print("\nPer-image predictions:")
for r in results:
    mark = "correct" if r["correct"] else "WRONG"
    print(f"  [{mark:7s}] {r['file'].split('/')[-1]:20s} true={r['true']:6s} pred={r['pred']:6s} conf={r['confidence']}")
