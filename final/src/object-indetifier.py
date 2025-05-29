import csv
import cv2
from ultralytics.solutions.speed_estimation import SpeedEstimator
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import Colors

# ——— CONFIG ——————————————————————————————————————————
VIDEO_PATH       = "training-videos/test3.mp4"
TRACKER_CFG      = "C:/Users/OZBERK/Desktop/final/tracker_config/byte_tracker.yaml"
OUTPUT_CSV       = "tracking_speed.csv"
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
        self.extract_tracks(im0)  # tracker path was set in constructor

        # 2. Prepare annotator
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        # 3. Iterate over each box (now a list of Tensors)
        for box, cls, conf, tid in zip(
            self.boxes, self.clss, self.confs, self.track_ids
        ):
            # box is a torch.Tensor of shape (4,)
            coords = box.cpu().numpy().ravel()
            x1, y1, x2, y2 = coords.astype(int)

            # Build label: "class_name conf% ID:tid"
            label = self.adjust_box_label(cls, conf, tid)
            # Append speed once available
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


if __name__ == "__main__":
    # get video metadata
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # optional: writer for annotated output
    writer_vid = cv2.VideoWriter(
        "annotated_ids_speed.avi",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # instantiate with your tracker config
    estimator = IDSpeedEstimator(
        show=True,
        model="yolo11n.pt",
        tracker=TRACKER_CFG,
        fps=fps,
        meter_per_pixel=METER_PER_PIXEL,
        max_speed=200,
        max_hist=5,
        classes=[2],   # only “car”
        line_width=2,
    )

    # open CSV log
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame","id","x1","y1","x2","y2","conf","speed_kmh"])

        cap = cv2.VideoCapture(VIDEO_PATH)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # process & annotate frame
            results = estimator.process(frame)

            # log detections to CSV
            for box, tid, conf in zip(
                estimator.boxes, estimator.track_ids, estimator.confs
            ):
                coords = box.cpu().numpy().ravel()
                x1, y1, x2, y2 = coords.astype(int)
                speed = estimator.spd.get(tid, 0.0)
                writer.writerow([
                    frame_idx,
                    tid,
                    x1, y1, x2, y2,
                    float(conf),
                    round(speed, 2)
                ])

            # save annotated frame
            writer_vid.write(results.plot_im)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

        cap.release()
        writer_vid.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_idx} frames. Results saved to {OUTPUT_CSV} and annotated video.")