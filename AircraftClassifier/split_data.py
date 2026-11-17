import os, shutil, random

random.seed(42)

SRC = "data"
OUT = "split"
VAL_PER_CLASS = 3  # small dataset -> small fixed holdout per class

classes = sorted([d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))])
print("Classes found:", classes)

if os.path.exists(OUT):
    shutil.rmtree(OUT)

for split in ["train", "val"]:
    for c in classes:
        os.makedirs(os.path.join(OUT, split, c), exist_ok=True)

for c in classes:
    files = [f for f in os.listdir(os.path.join(SRC, c)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    val_files = files[:VAL_PER_CLASS]
    train_files = files[VAL_PER_CLASS:]

    for f in val_files:
        shutil.copy(os.path.join(SRC, c, f), os.path.join(OUT, "val", c, f))
    for f in train_files:
        shutil.copy(os.path.join(SRC, c, f), os.path.join(OUT, "train", c, f))

    print(f"{c}: {len(train_files)} train, {len(val_files)} val")

print("Done splitting.")
