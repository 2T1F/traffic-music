# src/train_regressor.py

import os
import csv
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.congestion_regressor import CongestionRegressor

# ——— PROJECT ROOT & PATHS —————————————————————————————————————————
ROOT              = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMES_DIR        = os.path.join(ROOT, "frames")
FRAME_LABELS_CSV  = os.path.join(ROOT, "frame_labels.csv")

# ——— CONFIG ——————————————————————————————————————————
BATCH_SIZE       = 4       # keep small if GPU VRAM is limited
NUM_EPOCHS_HEAD  = 10      # increase head‐only epochs
NUM_EPOCHS_FT    = 5       # increase fine‐tune epochs
LR_HEAD          = 1e-3
LR_FINE_TUNE     = 1e-4
WEIGHT_DECAY     = 1e-4    # small weight decay for regularization
NUM_WORKERS      = 0       # set to >0 on Linux/Mac if possible
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS          = 10       # reduce to 5 if dataset is small
BACKBONE_NAME    = "resnet18"   # options: "resnet18", "resnet34", "efficientnet_b0"
DROPOUT_P        = 0.5     # dropout probability in regression head
METRIC_BUCKETS   = 10      # number of bins for stratification

# ——— DATASET —————————————————————————————————————————
class FrameDataset(Dataset):
    def __init__(self, frames_dir, labels_csv, transform):
        self.tf = transform
        self.items = []

        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["frame_path"]            # e.g. "frames/videoA/000010.jpg"
                label    = float(row["label"])

                abs_path = os.path.join(
                    os.path.dirname(frames_dir),   # project root
                    rel_path                       # "frames/videoA/000010.jpg"
                )
                if os.path.isfile(abs_path):
                    self.items.append((abs_path, label))
                else:
                    print(f"⚠️  Missing image file at {abs_path} — skipping")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        x   = self.tf(img)
        y   = torch.tensor(label, dtype=torch.float32)
        return x, y

    def get_all_labels(self):
        """Return a numpy array of all labels (useful for stratification)."""
        return np.array([label for (_, label) in self.items], dtype=np.float32)


# ——— TRANSFORMS (with data augmentation) —————————————————————————————————————————
transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ——— TRAINING & VALIDATION FUNCTIONS —————————————————————————————————————————
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.autocast():
            preds = model(xb)
            loss = criterion(preds, yb)

        scaler.scale(loss).backward()
        # Gradient clipping (optional) to stabilize training:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    val_mae  = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
            val_mae  += torch.abs(preds - yb).sum().item()

    val_loss /= len(loader.dataset)
    val_mae  /= len(loader.dataset)
    return val_loss, val_mae


def run_train_regressor():
    # 1) Load full dataset
    full_dataset = FrameDataset(
        frames_dir=FRAMES_DIR,
        labels_csv=FRAME_LABELS_CSV,
        transform=transform
    )
    N = len(full_dataset)
    print(f"Total frames available for training: {N}")
    if N == 0:
        raise RuntimeError("No training frames found! Did you run pseudo-label generator?")

    # 2) Prepare stratification bins if needed
    all_labels = full_dataset.get_all_labels()  # shape: (N,)
    # Digitize labels into METRIC_BUCKETS bins [0,1]
    bins = np.linspace(0.0, 1.0, METRIC_BUCKETS + 1)
    label_bins = np.digitize(all_labels, bins)  # values ∈ [1..METRIC_BUCKETS]
    # Use StratifiedKFold so each fold has roughly same distribution of bins
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_val_losses = []
    fold_val_maes   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), label_bins), start=1):
        print(f"\n>>> Fold {fold}/{K_FOLDS} <<<")
        train_subset = Subset(full_dataset, train_idx)
        val_subset   = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # 3) Instantiate model (initially freeze backbone; train head only)
        model = CongestionRegressor(
            backbone_name=BACKBONE_NAME,
            fine_tune=False,
            dropout_p=DROPOUT_P
        ).to(DEVICE)

        # Optimizer: only head parameters for now
        optimizer = torch.optim.Adam(
            model.backbone.fc.parameters(),
            lr=LR_HEAD,
            weight_decay=WEIGHT_DECAY
        )
        criterion = torch.nn.SmoothL1Loss()
        scaler    = torch.cuda.GradScaler()

        # Learning‐rate scheduler for head stage
        scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        best_val_loss = float("inf")
        best_model_state = None

        # 3a) Train head only
        for epoch in range(1, NUM_EPOCHS_HEAD + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_mae = validate_one_epoch(model, val_loader, criterion)

            # Step scheduler
            scheduler_head.step(val_loss)

            print(f"Fold {fold}, [HEAD] Epoch {epoch}/{NUM_EPOCHS_HEAD} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

            # Checkpoint best model for this “head” stage
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch
                }

        # 3b) Fine‐tune layer4 + head
        print("  --- Fine‐tuning layer4 + head for a few more epochs ---")
        model.set_fine_tune(True)
        # Re‐create optimizer to include all trainable params (layer4 + head)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_FINE_TUNE,
            weight_decay=WEIGHT_DECAY
        )
        scaler = torch.cuda.GradScaler()
        scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        for epoch in range(1, NUM_EPOCHS_FT + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_mae = validate_one_epoch(model, val_loader, criterion)
            scheduler_ft.step(val_loss)

            print(f"Fold {fold}, [FT] Epoch {epoch}/{NUM_EPOCHS_FT} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

            # Update checkpoint if this is best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': NUM_EPOCHS_HEAD + epoch
                }

        # Save best model of this fold to disk
        fold_weights_dir = os.path.join(ROOT, "weights", f"fold_{fold}")
        os.makedirs(fold_weights_dir, exist_ok=True)
        best_path = os.path.join(fold_weights_dir, "best_model.pth")
        torch.save(best_model_state, best_path)
        print(f"  ✅ Saved best model for fold {fold} (val_loss={best_val_loss:.4f}) at: {best_path}")

        fold_val_losses.append(best_val_loss)
        fold_val_maes.append(val_mae)

        torch.cuda.empty_cache()

    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
    avg_val_mae  = sum(fold_val_maes)   / len(fold_val_maes)
    print(f"\n=== Average validation loss over {K_FOLDS} folds: {avg_val_loss:.4f} ===")
    print(f"=== Average validation MAE over {K_FOLDS} folds: {avg_val_mae:.4f} ===")

    # 4) Retrain on entire dataset using the best hyperparameters found
    print("\nRetraining on entire dataset (HEAD + FT) to produce final model...")
    model = CongestionRegressor(
        backbone_name=BACKBONE_NAME,
        fine_tune=False,
        dropout_p=DROPOUT_P
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.backbone.fc.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.SmoothL1Loss()
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    scaler = torch.cuda.GradScaler()
    scheduler_head_full = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 4a) Train head on full data
    for epoch in range(1, NUM_EPOCHS_HEAD + 1):
        train_loss = train_one_epoch(model, full_loader, optimizer, criterion, scaler)
        # We’ll just monitor loss on the same training set here; no separate val:
        scheduler_head_full.step(train_loss)
        print(f"[Full HEAD] Epoch {epoch}/{NUM_EPOCHS_HEAD} | Loss: {train_loss:.4f}")

    # 4b) Fine‐tune layer4 + head on full data
    print("  --- Full Data: fine‐tuning layer4 + head ---")
    model.set_fine_tune(True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FINE_TUNE,
        weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.GradScaler()
    scheduler_ft_full = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    for epoch in range(1, NUM_EPOCHS_FT + 1):
        train_loss = train_one_epoch(model, full_loader, optimizer, criterion, scaler)
        scheduler_ft_full.step(train_loss)
        print(f"[Full FT] Epoch {epoch}/{NUM_EPOCHS_FT} | Loss: {train_loss:.4f}")

    # 5) Save final weights
    weights_dir = os.path.join(ROOT, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    final_path = os.path.join(weights_dir, "congestion_regressor_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_train_regressor()