import os
import random
import shutil

# ——— CONFIG —————————————————————————————————————————
ROOT            = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_DIR       = os.path.join(ROOT, "training-videos\\bddk1\\videos")       # original directory with all videos
TEST_DIR        = os.path.join(ROOT, "training-videos\\test")  # where 20% of videos will be moved
TRAIN_DIR       = os.path.join(ROOT, "training-videos\\train") # where the remaining 80% will go
TEST_RATIO      = 0.20                                        # fraction to move into test set
RANDOM_SEED     = 42

def split_videos_to_train_test():
    """
    Randomly selects TEST_RATIO fraction of video filenames from VIDEO_DIR,
    moves those into TEST_DIR, and moves the rest into TRAIN_DIR.
    """
    # 1) Create train/test directories (clear them first if they exist)
    for d in (TEST_DIR, TRAIN_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 2) List all video files in VIDEO_DIR
    all_videos = [
        f for f in os.listdir(VIDEO_DIR)
        if os.path.isfile(os.path.join(VIDEO_DIR, f)) 
           and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    print(f"Found {len(all_videos)} total video files in '{VIDEO_DIR}'.")

    # 3) Shuffle and pick TEST_RATIO fraction for test set
    random.seed(RANDOM_SEED)
    random.shuffle(all_videos)
    num_test = int(len(all_videos) * TEST_RATIO)
    test_videos = set(all_videos[:num_test])
    train_videos = set(all_videos[num_test:])

    print(f"-> {len(test_videos)} videos will be moved to TEST_DIR ({TEST_DIR}).")
    print(f"-> {len(train_videos)} videos will be moved to TRAIN_DIR ({TRAIN_DIR}).")

    # 4) Move files accordingly
    for fname in all_videos:
        src_path = os.path.join(VIDEO_DIR, fname)
        if fname in test_videos:
            dst_path = os.path.join(TEST_DIR, fname)
        else:
            dst_path = os.path.join(TRAIN_DIR, fname)

        shutil.move(src_path, dst_path)

    print("Finished splitting videos into train/test directories.")

if __name__ == "__main__":
    split_videos_to_train_test()
