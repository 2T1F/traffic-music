# src/pipeline.py

import sys
import os
from src.object_identifier import run_pipeline_obj_id, ROOT

from src.train_regressor import run_train_regressor

def main():
    VIDEO_DIR = os.path.join(ROOT, "training-videos", "train")

    if not os.path.isdir(VIDEO_DIR):
        raise RuntimeError(f"VIDEO_DIR does not exist: {VIDEO_DIR}")

    # 1) List all video files in VIDEO_DIR
    video_files = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if os.path.isfile(os.path.join(VIDEO_DIR, f))
           and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])
    if not video_files:
        raise RuntimeError(f"No video files found in '{VIDEO_DIR}'")
    # STEP 1: Object detection & tracking → per-video CSV + annotated AVI
    print("=== STEP 1: Object detection & tracking → per-video CSV + annotated AVI ===")
    for vf in video_files:
        # if not vf.endswith("mkv"):
        #     continue
        basename   = os.path.splitext(vf)[0]
        video_path = os.path.join(VIDEO_DIR, vf)
        print(f"Processing video: {video_path}")

        # Where to write this video’s CSV and annotated AVI:
        output_csv    = os.path.join(ROOT, "output/csvs/", f"{basename}_tracking_speed.csv")

        output_video_folder_path = os.path.join(ROOT, "output/videos")

        # Call the single‐video processor:
        run_pipeline_obj_id(video_path,vf,output_csv,output_video_folder_path)

    """
    # STEP 2: Pseudo-label generation → frames/ + frame_labels.csv
    print("\n=== STEP 2: Pseudo-label generation → frames/ + frame_labels.csv ===")
    run_pseudo_label_generator()

    # STEP 3: Train congestion regressor (with K-Fold CV + optional fine-tuning)
    print("\n=== STEP 3: Train congestion regressor (K-Fold CV + fine-tune) ===")
    run_train_regressor()

    print("\nAll done! Your final model weights are in weights/congestion_regressor_final.pth")
    """

if __name__ == "__main__":
    sys.exit(main())
