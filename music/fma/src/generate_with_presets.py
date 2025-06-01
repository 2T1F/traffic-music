# generate_with_presets.py

import os
import torch
import librosa
import soundfile as sf

from src.model_cvae5 import ConditionalVAE5  # make sure model_cvae5.py lives in the same folder

# ──────────────────────────────────────────────────────────────────────────────
# 1) Define your 5 presets (each a 5‐D tensor)
presets = {
    'calm_classical':   torch.tensor([1.0, 0.0, 0.1, 1.0, 0.0]),   # Pure classical: acoustic, instrumental, very low energy/liveness
    'ambient_drone':    torch.tensor([0.9, 0.0, 0.0, 1.0, 0.1]),   # Ambient pad/drone: minimal rhythm, high instrumental
    'chill_hiphop':     torch.tensor([0.5, 0.7, 0.6, 0.3, 0.4]),   # Mellow hip‐hop: mid energy, mid liveness
    'pop_electronic':   torch.tensor([0.1, 0.9, 0.9, 0.0, 0.8]),   # High‐energy EDM/pop: heavy beat, high danceability & liveness
    'upbeat_folk':      torch.tensor([0.8, 0.5, 0.5, 0.7, 0.2]),   # Acoustic folk groove: medium energy, acoustic instruments
}

# 2) Map congestion c ∈ [0,1] → weights over those presets
def traffic_to_weights(c: float) -> dict:
    """
    Given congestion c ∈ [0,1], return a dict mapping each preset name → weight ∈ [0,1].
    Weights sum to 1 and vary piecewise:
      c < 0.20     → mostly pop_electronic (+ small upbeat_folk)
      0.20 ≤ c < 0.40 → mostly upbeat_folk → introduce chill_hiphop
      0.40 ≤ c < 0.60 → mostly chill_hiphop → introduce ambient_drone
      0.60 ≤ c < 0.80 → mostly ambient_drone → introduce calm_classical
      c ≥ 0.80     → mix ambient_drone & calm_classical → mostly calm_classical at c = 1.00
    """
    c = max(0.0, min(1.0, c))

    if c < 0.20:
        # Very low congestion: favor pop_electronic (free‐flow roads) + slight upbeat_folk
        return {
            'pop_electronic': 0.9,
            'upbeat_folk':    0.1,
            'chill_hiphop':   0.0,
            'ambient_drone':  0.0,
            'calm_classical': 0.0,
        }

    elif c < 0.40:
        # Low‐to‐moderate congestion: base is upbeat_folk → introduce chill_hiphop as c→0.40
        w = (c - 0.20) / 0.20  # normalized over [0.20,0.40]
        return {
            'pop_electronic': 0.0,
            'upbeat_folk':    1.0 - w * 0.5,   # at c=0.20 → 1.0; at c=0.40 → 0.5
            'chill_hiphop':   w * 0.5,         # at c=0.20 → 0.0; at c=0.40 → 0.5
            'ambient_drone':  0.0,
            'calm_classical': 0.0,
        }

    elif c < 0.60:
        # Moderate congestion: base is chill_hiphop → introduce ambient_drone as c→0.60
        w = (c - 0.40) / 0.20  # normalized over [0.40,0.60]
        return {
            'pop_electronic': 0.0,
            'upbeat_folk':    0.0,
            'chill_hiphop':   1.0 - w * 0.5,   # at c=0.40 → 1.0; at c=0.60 → 0.5
            'ambient_drone':  w * 0.5,         # at c=0.40 → 0.0; at c=0.60 → 0.5
            'calm_classical': 0.0,
        }

    elif c < 0.80:
        # High congestion: base is ambient_drone → introduce calm_classical as c→0.80
        w = (c - 0.60) / 0.20  # normalized over [0.60,0.80]
        return {
            'pop_electronic': 0.0,
            'upbeat_folk':    0.0,
            'chill_hiphop':   0.0,
            'ambient_drone':  1.0 - w * 0.5,   # at c=0.60 → 1.0; at c=0.80 → 0.5
            'calm_classical': w * 0.5,         # at c=0.60 → 0.0; at c=0.80 → 0.5
        }

    else:
        # Very high congestion: mix ambient_drone & calm_classical → mostly calm_classical at c=1.00
        w = (c - 0.80) / 0.20  # normalized over [0.80,1.00]
        return {
            'pop_electronic': 0.0,
            'upbeat_folk':    0.0,
            'chill_hiphop':   0.0,
            'ambient_drone':  1.0 - w,  # at c=0.80 → 1.0; at c=1.00 → 0.0
            'calm_classical': w,        # at c=0.80 → 0.0; at c=1.00 → 1.0
        }

# 3) Utility: compute a single 5-D cond_vec from c
def compute_cond_vec(c: float) -> torch.Tensor:
    """
    Given congestion c ∈ [0,1], return a 5-D tensor in [0,1]^5
    that is the convex combination of the five preset vectors.
    """
    weights = traffic_to_weights(c)
    cond = torch.zeros(5, dtype=torch.float32)
    for name, w in weights.items():
        cond += w * presets[name]
    return cond  # shape (5,)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Model hyperparameters (must match what you used during training)
LATENT_DIM = 128
N_MELS     = 80
N_FEATS    = 5
SR         = 16000
DURATION   = 30.0
T_FRAMES   = int(SR * DURATION / 256)  # e.g. ≈1875 frames

# 5) Path to your trained CVAE5 checkpoint
CHECKPOINT_PATH = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\checkpoints_cvae5\\cvae5_epoch30.pt"

# 6) Load model & weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalVAE5(latent_dim=LATENT_DIM,
                       n_mels=N_MELS,
                       n_feats=N_FEATS,
                       T=T_FRAMES).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# 7) Mel → waveform (Griffin-Lim) helper
def mel_to_audio(mel_tensor: torch.Tensor) -> torch.Tensor:
    """
    mel_tensor: shape (1,80,T) in [0,1].
    Returns a numpy waveform of length SR * DURATION.
    """
    # 1) Scale [0,1] → [−80,0] dB
    log_S = mel_tensor.squeeze(0).cpu().numpy() * 80.0 - 80.0
    S = librosa.db_to_power(log_S, ref=1.0)

    # 2) Invert mel back to audio with Griffin-Lim
    wav = librosa.feature.inverse.mel_to_audio(
        S,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_iter=32,
        length=int(SR * DURATION),
    )
    return wav

# ──────────────────────────────────────────────────────────────────────────────
# 8) Main: generate one 30 s clip for each sample congestion value
if __name__ == "__main__":
    out_folder = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\generated_presets"
    os.makedirs(out_folder, exist_ok=True)

    # Example congestion levels to generate
    congestion_values = [0.00, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00]

    for c in congestion_values:
        # a) Compute a single 5-D cond_vec and move to the correct device
        cond_vec = compute_cond_vec(c).unsqueeze(0).to(device)  # shape (1,5)

        # b) Sample z ∼ N(0,I)
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, device=device)
            # c) Decode: mel_hat has shape (1,1,80,T_FRAMES)
            mel_hat = model.decoder(z, cond_vec)

        # d) Convert mel_hat → waveform
        wav = mel_to_audio(mel_hat.squeeze(0))

        # e) Write out as a .wav file
        out_filename = f"c{c:.2f}.wav"
        out_path = os.path.join(out_folder, out_filename)
        sf.write(out_path, wav, samplerate=SR)
        print(f"Saved: congestion={c:.2f} → {out_path}")
