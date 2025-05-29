# src/train_regressor.py

import os
import csv
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from src.congestion_regressor import CongestionRegressor  

# ——— PROJECT ROOT & PATHS —————————————————————————————————————————
ROOT              = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMES_DIR        = os.path.join(ROOT, "frames")
FRAME_LABELS_CSV  = os.path.join(ROOT, "frame_labels.csv")

# ——— CONFIG ——————————————————————————————————————————
BATCH_SIZE       = 16
NUM_EPOCHS       = 10
LR               = 1e-3
NUM_WORKERS      = 0      # use 0 on Windows to avoid spawn issues
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ——— DATASET —————————————————————————————————————————
class FrameDataset(Dataset):
    def __init__(self, frames_dir, labels_csv, transform):
        self.tf = transform
        self.items = []

        # 1) load labels into a dict: frame_idx → label
        labels = {}
        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(os.path.splitext(os.path.basename(row["frame_path"]))[0])
                labels[idx] = float(row["label"])

        # 2) for each labeled frame, include it if the file exists
        for idx, label in sorted(labels.items()):
            filename = f"{idx:06d}.jpg"
            path = os.path.join(frames_dir, filename)
            if os.path.isfile(path):
                self.items.append((path, label))
            else:
                print(f"⚠️  Missing frame image {path} — skipping")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        x   = self.tf(img)
        y   = torch.tensor(label, dtype=torch.float32)
        return x, y

# ——— TRANSFORMS —————————————————————————————————————————
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]),
])

def run_train_regressor():
    # 1) Prepare dataset & loader
    dataset = FrameDataset(
        frames_dir=FRAMES_DIR,
        labels_csv=FRAME_LABELS_CSV,
        transform=transform
    )
    count = len(dataset)
    print(f"Found {count} frames in '{FRAMES_DIR}'")
    if count == 0:
        raise RuntimeError("No training frames found! Did you run pseudo‐label generator?")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # 2) Model, optimizer, loss
    model     = CongestionRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 3) Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(x_batch)
            loss  = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        epoch_loss = running_loss / count
        print(f"Epoch {epoch}/{NUM_EPOCHS} — MSE: {epoch_loss:.4f}")

    # 4) Save weights
    weights_dir = os.path.join(ROOT, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    out_path = os.path.join(weights_dir, "congestion_regressor.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Training complete. Model saved to: {out_path}")

if __name__ == "__main__":
    # Windows mp support
    from multiprocessing import freeze_support
    freeze_support()
    run_train_regressor()
