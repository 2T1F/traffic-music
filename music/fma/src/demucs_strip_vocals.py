#!/usr/bin/env python3
"""
demucs_strip_vocals.py

Usage:
    python demucs_strip_vocals.py <input_audio.mp3> <output_instrumental.wav>

Loads the pretrained Demucs “htdemucs” model and writes out only the accompaniment stems.
"""

import sys
import os
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply     import apply_model

def load_model(device):
    # Load the Hybrid Transformer Demucs model
    model = get_model(name="htdemucs")
    model.to(device)
    model.eval()
    return model

def strip_vocals(model, src_path, dst_path, device):
    # 1) Read audio (numpy float64): shape = (n,) or (n, channels)
    wav_np, sr = sf.read(src_path)

    # 2) Convert to torch.FloatTensor with shape (1, channels, samples)
    if wav_np.ndim == 1:
        wav = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,n)
    else:
        wav = torch.from_numpy(wav_np.T).float().unsqueeze(0)             # (1,c,n)
    wav = wav.to(device)

    # 3) Separate stems
    sources = apply_model(model, wav, device=device, split=True)

    # 4) Ensure we have a list of tensors
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    # 5) Concatenate along the “stem” dimension: results in (n_stems, channels, samples)
    stems = torch.cat(sources, dim=0)

    # 6) Sum all non-vocal stems (indices 1..end) → (channels, samples)
    accompaniment = stems[1:].sum(dim=0)

    # 7) Convert back to numpy (samples, channels)
    out_np = accompaniment.cpu().numpy().T

    # 8) Write the instrumental WAV
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, out_np, sr)
    print(f"✔ Instrumental saved to: {dst_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python demucs_strip_vocals.py <input_audio> <output_instrumental.wav>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    if not os.path.isfile(src):
        print(f"Error: input file not found: {src}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Demucs ‘htdemucs’ model on {device}…")
    model = load_model(device)

    print(f"Processing: {src}")
    strip_vocals(model, src, dst, device)

if __name__ == "__main__":
    main()
