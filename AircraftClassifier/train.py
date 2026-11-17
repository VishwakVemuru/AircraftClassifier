import os, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mobilenetv2_def import mobilenetv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "split"
BATCH_SIZE = 8
IMG_SIZE = 224
NUM_EPOCHS_HEAD = 15      # phase 1: train only the new classifier head
NUM_EPOCHS_FINETUNE = 10  # phase 2: unfreeze last block, fine-tune with low LR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Data: heavy augmentation on train (dataset is tiny), light transform on val
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

class_names = train_ds.classes
print("Classes (index order):", class_names)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Model: pretrained MobileNetV2 backbone (ImageNet), new head for 4 classes
# ---------------------------------------------------------------------------
model = mobilenetv2(num_classes=1000)
state_dict = torch.load("mobilenetv2_pretrained.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=True)
print("Loaded ImageNet-pretrained MobileNetV2 backbone.")

num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(class_names))
model = model.to(DEVICE)

# Freeze everything except the new classifier head for phase 1
for name, param in model.named_parameters():
    param.requires_grad = "classifier" in name

criterion = nn.CrossEntropyLoss()


def run_epoch(loader, model, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_phase(model, epochs, lr, phase_name):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    history = []
    print(f"\n--- {phase_name} ({epochs} epochs, lr={lr}) ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_acc = run_epoch(val_loader, model, criterion, optimizer=None)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"Epoch {epoch:2d}/{epochs} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f}")
    return history


if __name__ == "__main__":
    t0 = time.time()

    # Phase 1: train just the new head, backbone frozen
    history1 = train_phase(model, NUM_EPOCHS_HEAD, lr=1e-3, phase_name="Phase 1: head only")

    # Phase 2: unfreeze the last inverted-residual block + conv/classifier, fine-tune at low LR
    for name, param in model.named_parameters():
        if name.startswith("features.16") or name.startswith("features.17") \
           or name.startswith("conv.") or name.startswith("classifier"):
            param.requires_grad = True

    history2 = train_phase(model, NUM_EPOCHS_FINETUNE, lr=1e-4, phase_name="Phase 2: fine-tune last block")

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f}s")

    # Save model + metadata
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }, "aircraft_classifier.pt")
    print("Saved trained model to aircraft_classifier.pt")

    with open("training_history.json", "w") as f:
        json.dump({"phase1": history1, "phase2": history2}, f, indent=2)
    print("Saved training history to training_history.json")
