# src/object_identifier.py

import os
import csv
import cv2
from ultralytics.solutions.speed_estimation import SpeedEstimator
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import Colors

# ——— PROJECT ROOT & PATHS —————————————————————————————————————————
ROOT             = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_PATH       = os.path.join(ROOT, "training-videos", "test3.mp4")
TRACKER_CFG      = os.path.join(ROOT, "tracker_config", "byte_tracker.yaml")
OUTPUT_CSV       = os.path.join(ROOT, "tracking_speed.csv")
METER_PER_PIXEL  = 0.05    # scene calibration (meters per pixel)

# initialize color palette
COLORS = Colors()

class IDSpeedEstimator(SpeedEstimator):
    """
    Extends SpeedEstimator to draw CLASS, CONFIDENCE, TRACK ID, and SPEED
    in a single label per detection box.
    """
    def process(self, im0):
        # 1. Run detection + tracking + speed logic
        self.frame_count += 1
        self.extract_tracks(im0)

        # 2. Prepare annotator
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        # 3. Iterate over each box
        for box, cls, conf, tid in zip(
            self.boxes, self.clss, self.confs, self.track_ids
        ):
            coords = box.cpu().numpy().ravel().astype(int)
            x1, y1, x2, y2 = coords

            # Build label
            label = self.adjust_box_label(cls, conf, tid)
            if tid in self.spd:
                label += f" {self.spd[tid]:.1f} km/h"

            # Draw
            annotator.box_label(
                (x1, y1, x2, y2),
                label=label,
                color=COLORS(tid, bgr=True)
            )

        # 4. Finalize and display
        annotated = annotator.result()
        self.display_output(annotated)
        return SolutionResults(plot_im=annotated, total_tracks=len(self.track_ids))


def run_object_identifier():
    """
    1) Opens the video, instantiates the IDSpeedEstimator,
    2) Processes each frame, annotates it, logs to CSV,
    3) Writes out an annotated video and tracking_speed.csv.
    """
    # 1) Grab video metadata
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2) Prepare output video writer
    out_vid = cv2.VideoWriter(
        os.path.join(ROOT, "annotated_ids_speed.avi"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # 3) Instantiate estimator
    estimator = IDSpeedEstimator(
        show=False,                  # set True if you want live display
        model="yolo11n.pt",          # your model file
        tracker=TRACKER_CFG,
        fps=fps,
        meter_per_pixel=METER_PER_PIXEL,
        max_speed=200,
        max_hist=5,
        classes=[2],                 # only “car”
        line_width=2,
    )

    # 4) Open CSV for logging
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame","id","x1","y1","x2","y2","conf","speed_kmh"])

        cap = cv2.VideoCapture(VIDEO_PATH)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # annotate & compute speeds
            results = estimator.process(frame)

            # log each detection
            for box, tid, conf in zip(
                estimator.boxes, estimator.track_ids, estimator.confs
            ):
                coords = box.cpu().numpy().ravel().astype(int)
                x1, y1, x2, y2 = coords
                speed = estimator.spd.get(tid, 0.0)
                writer.writerow([
                    frame_idx,
                    tid,
                    x1, y1, x2, y2,
                    float(conf),
                    round(speed, 2)
                ])

            # write out annotated frame
            out_vid.write(results.plot_im)
            frame_idx += 1

        cap.release()
        out_vid.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames. Results in:\n"
          f" • {OUTPUT_CSV}\n"
          f" • {os.path.join(ROOT,'annotated_ids_speed.avi')}")


if __name__ == "__main__":
    run_object_identifier()
