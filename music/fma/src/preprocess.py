# preprocess.py

import os
import librosa
import numpy as np
import pandas as pd

################################################################################
#  SETTINGS: adjust these paths to match your project layout
################################################################################
RAW_AUDIO_DIR = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\instrumentals"
FEATURES_CSV  = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_pruned_for_training.csv"
MEL_DIR       = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\mels"
MATCHED_CSV   = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_pruned_for_training_with_mels.csv"

# Even on a potato PC, these can be smallish
SR         = 8000     # downsample from 16 kHz→8 kHz to reduce mel size & speed up
DURATION   = 10.0     # only keep 10 s instead of 30 s (lowers T_FRAMES by ×3)
N_MELS     = 40       # half as many mel bins (40 instead of 80)
HOP_LENGTH = 256      
N_FFT      = 512      # smaller FFT
################################################################################

# Calculated once
T_FRAMES = int(SR * DURATION / HOP_LENGTH)  # ≈ (8000*10)/256 ≈ 312 frames
os.makedirs(MEL_DIR, exist_ok=True)

#  1) Load features CSV
df_feat = pd.read_csv(FEATURES_CSV, dtype={
    "track_id": int,
    "acousticness": float,
    "danceability": float,
    "energy": float,
    "instrumentalness": float,
    "liveness": float
})
required = {"track_id", "acousticness", "danceability", "energy", "instrumentalness", "liveness"}
if not required.issubset(df_feat.columns):
    missing = required - set(df_feat.columns)
    raise ValueError(f"Missing columns in {FEATURES_CSV}: {missing}")

#  2) Build lookup: stripped‐ID → filename
lookup = {}
for fname in os.listdir(RAW_AUDIO_DIR):
    base, ext = os.path.splitext(fname)
    if ext.lower() not in [".mp3", ".wav"]:
        continue
    prefix = base.split("_", 1)[0]
    try:
        track_int = int(prefix.lstrip("0") or "0")
    except ValueError:
        continue
    lookup[track_int] = fname

#  3) Compute & save mel, skip all-zero results
records = []
for idx, row in df_feat.iterrows():
    track_id = int(row["track_id"])
    if track_id not in lookup:
        print(f"Warning: no raw audio file for track_id={track_id}; skipping.")
        continue

    raw_fname = lookup[track_id]
    raw_path  = os.path.join(RAW_AUDIO_DIR, raw_fname)

    # Load at 8 kHz (SR)
    y, _ = librosa.load(raw_path, sr=SR)

    # Trim/pad to exactly DURATION (10 s)
    target_len = int(SR * DURATION)
    if y.shape[0] < target_len:
        y = librosa.util.fix_length(y, size=target_len)
    else:
        y = y[:target_len]

    # Compute mel‐spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    actual_T = S.shape[1]

    # Pad/truncate to exactly (N_MELS × T_FRAMES)
    if actual_T > T_FRAMES:
        S = S[:, :T_FRAMES]
    elif actual_T < T_FRAMES:
        pad_width = T_FRAMES - actual_T
        pad_array = np.full((N_MELS, pad_width), 1e-8, dtype=S.dtype)
        S = np.concatenate([S, pad_array], axis=1)

    # Convert power→dB and normalize to [0,1]
    log_S = librosa.power_to_db(S, ref=1.0)        # in [−80,0]
    mel01 = (log_S + 80.0) / 80.0
    mel01 = np.clip(mel01, 0.0, 1.0).astype(np.float32)

    # Skip if mel is all zeros (i.e. silent file)
    if np.allclose(mel01, 0.0, atol=1e-6):
        print(f"Skipping track_id={track_id} (all-zero mel).")
        continue

    # Save as “mels/{track_id}.npy”
    mel_path = os.path.join(MEL_DIR, f"{track_id}.npy")
    np.save(mel_path, mel01)

    # Append to records
    records.append({
        "track_id":         track_id,
        "acousticness":     row["acousticness"],
        "danceability":     row["danceability"],
        "energy":           row["energy"],
        "instrumentalness": row["instrumentalness"],
        "liveness":         row["liveness"],
        "mel_path":         mel_path.replace("\\", "/")
    })

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx+1}/{len(df_feat)} tracks…")

#  4) Write out matched CSV
df_out = pd.DataFrame(records)
df_out = df_out.sort_values("track_id").reset_index(drop=True)
df_out.to_csv(MATCHED_CSV, index=False)

print(f"\nWrote {len(df_out)} rows to {MATCHED_CSV}")
print("Preprocessing complete.")
