import os
import sys
import subprocess
import shutil
import glob
# Directories
INPUT_DIR  = "music\\fma\\small\\fma_small"
OUTPUT_DIR = "music\\fma\\instrumentals"
MODEL_NAME = "htdemucs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_mp3s = glob.glob(os.path.join(INPUT_DIR, "**", "*.mp3"), recursive=True)

# ─── 2) Filter: skip any track whose *_ins.mp3 already exists ───────────────────
pending = []
for mp3 in all_mp3s:
    track_name = os.path.splitext(os.path.basename(mp3))[0]
    out_file   = os.path.join(OUTPUT_DIR, f"{track_name}_ins.mp3")
    if not os.path.exists(out_file):
        pending.append(mp3)

print(f"Total pending: {len(pending)} tracks")

# ─── 3) Sort pending in descending order by their parent folder name ─────────────
# Extract the immediate parent folder (e.g., "155", "154", etc.), convert to int, sort descending
def folder_key(path):
    parent = os.path.basename(os.path.dirname(path))
    try:
        return int(parent)
    except ValueError:
        # If folder names aren’t strictly integers, fall back to lex order
        return parent

pending.sort(key=folder_key, reverse=True)

# ─── 4) Process each in descending order ────────────────────────────────────────
for mp3 in pending:
    track_name      = os.path.splitext(os.path.basename(mp3))[0]
    model_track_dir = os.path.join(OUTPUT_DIR, MODEL_NAME, track_name)

    # Skip again if someone crashed and wrote partial output
    final_ins = os.path.join(OUTPUT_DIR, f"{track_name}_ins.mp3")
    if os.path.exists(final_ins):
        print(f"Skipping {track_name}: already exists")
        continue

    print(f"Processing: {mp3}")
    try:
        # 1) Run Demucs for this single file
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "--two-stems=vocals",
            "--mp3",
            "--device", "cuda",
            "-j", "4",
          
            "--out", OUTPUT_DIR,
            mp3
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ! Demucs failed for {track_name}, skipping. ({e})")
        continue

    # 2) Delete the unwanted vocals.mp3
    vocals_path = os.path.join(model_track_dir, "vocals.mp3")
    if os.path.exists(vocals_path):
        os.remove(vocals_path)

    # 3) Move & rename the instrumental stem
    instrumental_src = os.path.join(model_track_dir, "no_vocals.mp3")
    if os.path.exists(instrumental_src):
        instrumental_dst = os.path.join(OUTPUT_DIR, f"{track_name}_ins.mp3")
        shutil.move(instrumental_src, instrumental_dst)
        print(f"  • Saved instrumental as: {instrumental_dst}")
    else:
        print(f"  ! no_vocals.mp3 not found for {track_name}")

print("✅ All done.")
