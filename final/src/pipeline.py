# src/pipeline.py

import sys
from src.object_identifier    import run_object_identifier
from src.pseudo_label_generator import run_pseudo_label_generator
from src.train_regressor        import run_train_regressor

def main():
    print("=== STEP 1: Object detection & tracking → tracking_speed.csv ===")
    run_object_identifier()

    print("\n=== STEP 2: Pseudo‐label generation → frames/ + frame_labels.csv ===")
    run_pseudo_label_generator()

    print("\n=== STEP 3: Train congestion regressor ===")
    run_train_regressor()

    print("\nAll done! Your model weights are in weights/")

if __name__ == "__main__":
    sys.exit(main())
