# train_cvae.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from model_cvae5 import ConditionalVAE5

################################################################################
# 1) Hyperparameters (potato PC mode)
################################################################################
LATENT_DIM    = 32       # smaller latent vector
N_MELS        = 40       # must match preprocess
N_FEATS       = 5
SR            = 8000     # must match preprocess
DURATION      = 10.0     # seconds
HOP_LENGTH    = 256
T_FRAMES      = int(SR * DURATION / HOP_LENGTH)  # ≈ 312 frames

BATCH_SIZE    = 4        # smallish batch
LEARNING_RATE = 1e-4     # keep small
NUM_EPOCHS    = 15       # fewer epochs to save time

# Force CPU only (pin_memory and CUDA disabled)
DEVICE = torch.device("cpu")

################################################################################
# 2) Dataset: loads mel + conditioning vector
################################################################################
class TrafficMusicDataset(Dataset):
    def __init__(self, csv_file):
        """
        csv_file must have:
          track_id,acousticness,danceability,energy,instrumentalness,liveness,mel_path
        """
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["mel_path"])
        self.records = df.to_dict(orient="records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        mel = np.load(rec["mel_path"], allow_pickle=False)     # (40, 312)
        mel = torch.from_numpy(mel).float()                     # (40,312)
        mel = mel.unsqueeze(0)                                  # → (1,40,312)

        cond = torch.tensor([
            rec["acousticness"],
            rec["danceability"],
            rec["energy"],
            rec["instrumentalness"],
            rec["liveness"]
        ], dtype=torch.float32)                                 # (5,)
        return mel, cond

################################################################################
# 3) CVAE loss: reconstruction (MSE) + KL divergence
################################################################################
def cvae_loss(mel_hat, mel, mu, logvar):
    B = mel.size(0)
    rec_loss = nn.functional.mse_loss(mel_hat, mel, reduction="sum") / B
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return rec_loss + kld, rec_loss.detach(), kld.detach()

################################################################################
# 4) Train/validation split helper
################################################################################
def train_val_split(csv_path, val_frac=0.2, seed=42):
    df_full = pd.read_csv(csv_path)
    df_full = df_full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df_full) * val_frac)
    df_val = df_full.iloc[:n_val]
    df_trn = df_full.iloc[n_val:]
    return df_trn, df_val

################################################################################
# 5) Main: training loop
################################################################################
if __name__ == "__main__":
    # (A) Paths: adjust if needed
    csv_full = os.path.join(os.pardir, "matched_pruned_for_training_with_mels.csv")
    ckpt_dir = os.path.join(os.pardir, "checkpoints_cvae5_potato")
    os.makedirs(ckpt_dir, exist_ok=True)

    # (B) Split into train/val
    df_trn, df_val = train_val_split(csv_full, val_frac=0.15)
    trn_csv = os.path.join(os.pardir, "train_temp_potato.csv")
    val_csv = os.path.join(os.pardir, "val_temp_potato.csv")
    df_trn.to_csv(trn_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    print(f"Training: {len(df_trn)} examples   Validation: {len(df_val)} examples\n")

    # (C) Datasets & DataLoaders (no pin_memory, num_workers=0)
    train_ds = TrafficMusicDataset(trn_csv)
    val_ds   = TrafficMusicDataset(val_csv)
    trn_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    # (D) Instantiate model
    model = ConditionalVAE5(
        latent_dim=LATENT_DIM,
        n_mels=N_MELS,
        n_feats=N_FEATS,
        T=T_FRAMES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (E) Sanity check on the first batch
    for batch_idx, (mel, cond) in enumerate(trn_loader):
        print("▶▶▶ First‐batch sanity checks:")
        print(" mel     → min, max, mean, any NaN:",
              mel.min().item(), mel.max().item(), mel.mean().item(), torch.isnan(mel).any().item())
        print(" cond    → min, max, mean, any NaN:",
              cond.min().item(), cond.max().item(), cond.mean().item(), torch.isnan(cond).any().item())
        print(" shapes  → mel:", mel.shape, "cond:", cond.shape)
        print("=======================================\n")
        break  # only do this once

    # (F) Training loop
    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_trn_loss = 0.0
        total_trn_rec  = 0.0
        total_trn_kld  = 0.0

        for mel, cond in trn_loader:
            mel  = mel.to(DEVICE)  # (B,1,40,312)
            cond = cond.to(DEVICE) # (B,5)

            optimizer.zero_grad()
            mel_hat, mu, logvar = model(mel, cond)
            loss, rec, kld = cvae_loss(mel_hat, mel, mu, logvar)
            loss.backward()
            optimizer.step()

            total_trn_loss += loss.item()
            total_trn_rec  += rec.item()
            total_trn_kld  += kld.item()

        avg_trn_loss = total_trn_loss / len(train_ds)
        avg_trn_rec  = total_trn_rec  / len(train_ds)
        avg_trn_kld  = total_trn_kld  / len(train_ds)

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        total_val_rec  = 0.0
        total_val_kld  = 0.0
        with torch.no_grad():
            for mel, cond in val_loader:
                mel  = mel.to(DEVICE)
                cond = cond.to(DEVICE)
                mel_hat, mu, logvar = model(mel, cond)
                loss, rec, kld = cvae_loss(mel_hat, mel, mu, logvar)
                total_val_loss += loss.item()
                total_val_rec  += rec.item()
                total_val_kld  += kld.item()

        avg_val_loss = total_val_loss / len(val_ds)
        avg_val_rec  = total_val_rec  / len(val_ds)
        avg_val_kld  = total_val_kld  / len(val_ds)

        print(
            f"Epoch {epoch:02d}  "
            f"Train Loss={avg_trn_loss:.3f} Rec={avg_trn_rec:.3f} KLD={avg_trn_kld:.3f} |  "
            f"Val   Loss={avg_val_loss:.3f} Rec={avg_val_rec:.3f} KLD={avg_val_kld:.3f}"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(ckpt_dir, f"cvae5_epoch{epoch:02d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ New best val loss. Checkpoint saved to:\n      {ckpt_path}")

    print("Training complete.")
