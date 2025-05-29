# src/pseudo_label_generator.py

import os
import csv
import shutil
import cv2
import numpy as np
from collections import defaultdict

# ——— PROJECT ROOT & CONFIG —————————————————————————————————————————
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_PATH  = os.path.join(ROOT, "training-videos", "test3.mp4")
TRACK_CSV   = os.path.join(ROOT, "tracking_speed.csv")
FRAMES_DIR  = os.path.join(ROOT, "frames")
LABELS_CSV  = os.path.join(ROOT, "frame_labels.csv")

# sampling & labeling weights
SAMPLE_RATE = 5    # take one frame every 5 frames
ALPHA       = 0.5  # weight density vs. inverse speed


def compute_pseudo_labels(track_csv):
    """
    Reads tracking_speed.csv and for each frame t computes:
      density_t = (# unique IDs in t) / max_over_t
      speed_inv_t = 1 - (avg_speed_t / max_avg_speed_over_t)
    Then returns labels[t] = clip(ALPHA*density_t + (1-ALPHA)*speed_inv_t, 0,1).
    """
    counts     = defaultdict(set)
    speeds_sum = defaultdict(float)
    speeds_cnt = defaultdict(int)

    with open(track_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t   = int(row["frame"])
            tid = int(row["id"])
            spd = float(row["speed_kmh"])
            counts[t].add(tid)
            speeds_sum[t]  += spd
            speeds_cnt[t]  += 1

    frame_idxs    = sorted(counts.keys())
    max_count     = max((len(counts[t]) for t in frame_idxs), default=1)
    max_avg_speed = max(
        (speeds_sum[t] / speeds_cnt[t] if speeds_cnt[t] else 0.0)
        for t in frame_idxs
    ) or 1.0

    labels = {}
    for t in frame_idxs:
        cnt     = len(counts[t])
        avg_spd = (speeds_sum[t] / speeds_cnt[t]) if speeds_cnt[t] else 0.0

        density   = cnt / max_count
        speed_inv = 1.0 - (avg_spd / max_avg_speed)

        raw = ALPHA * density + (1 - ALPHA) * speed_inv
        labels[t] = float(np.clip(raw, 0.0, 1.0))

    return labels


def sample_and_save_frames(video_path, labels_dict, out_dir, sample_rate):
    """
    Samples every `sample_rate`-th frame from video_path, saves to out_dir,
    and returns two parallel lists: frame_paths (abs) and frame_labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_paths, frame_labels = [], []

    for t in range(0, total_frames, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, frame = cap.read()
        if not ret:
            continue
        label = labels_dict.get(t, 0.0)
        fname = f"{t:06d}.jpg"
        path  = os.path.join(out_dir, fname)
        cv2.imwrite(path, frame)
        frame_paths.append(path)
        frame_labels.append(label)

    cap.release()
    return frame_paths, frame_labels


def write_frame_labels_csv(frame_paths, frame_labels, out_csv):
    """
    Writes a CSV with columns [frame_path, label], using paths relative to ROOT.
    """
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "label"])
        for abs_path, lbl in zip(frame_paths, frame_labels):
            rel = os.path.relpath(abs_path, ROOT).replace("\\", "/")
            writer.writerow([rel, lbl])


def run_pseudo_label_generator():
    # 1) Clean frames directory
    if os.path.exists(FRAMES_DIR):
        print(f"Removing old frames folder: {FRAMES_DIR}")
        shutil.rmtree(FRAMES_DIR)

    # 2) Compute pseudo‐labels
    print("Generating pseudo‐labels from:", TRACK_CSV)
    labels_dict = compute_pseudo_labels(TRACK_CSV)
    print(f"Computed labels for {len(labels_dict)} frames.")

    # 3) Sample frames & save
    print("Sampling and saving frames from:", VIDEO_PATH)
    frame_paths, frame_labels = sample_and_save_frames(
        VIDEO_PATH, labels_dict, FRAMES_DIR, SAMPLE_RATE
    )
    print(f"Saved {len(frame_paths)} frames to '{FRAMES_DIR}/'")

    # 4) Write out frame_labels.csv
    print("Writing frame_labels.csv to:", LABELS_CSV)
    write_frame_labels_csv(frame_paths, frame_labels, LABELS_CSV)
    print("Done. Frame–label CSV:", LABELS_CSV)


if __name__ == "__main__":
    run_pseudo_label_generator()
