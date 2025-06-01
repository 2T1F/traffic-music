# src/object_identifier.py

import os
import csv
import cv2
import numpy as np
from ultralytics import solutions
from ultralytics.utils.plotting import Colors

# ——— PROJECT ROOT & PATHS —————————————————————————————————————————
ROOT             = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_DIR        = os.path.join(ROOT, "training-videos/train")
TRACKER_CFG      = os.path.join(ROOT, "tracker_config", "byte_tracker.yaml")
METER_PER_PIXEL  = 0.05    # scene calibration (meters per pixel)

# ===== SPEED ESTIMATION IMPROVEMENTS =====
# Camera motion compensation settings
CAMERA_MOTION_THRESHOLD = 3.0  # pixels - detect if camera is moving
STATIONARY_SPEED_THRESHOLD = 1.0  # km/h - below this is considered stationary
MIN_MOVEMENT_PIXELS = 10  # minimum pixel movement to calculate speed

# initialize color palette
COLORS = Colors()

class MotionCompensatedSpeedEstimator:
    """
    Enhanced speed estimator with camera motion compensation
    """
    def __init__(self, **kwargs):
        self.speed_estimator = solutions.SpeedEstimator(**kwargs)
        self.prev_frame = None
        self.camera_motion = (0, 0)  # Store camera motion vector
        self.stationary_objects = {}  # Track potentially stationary objects
        
    def detect_camera_motion(self, current_frame):
        """
        Detect camera motion using optical flow on background features
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return (0, 0)
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in previous frame
        corners = cv2.goodFeaturesToTrack(
            self.prev_frame, 
            maxCorners=100, 
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        if corners is not None:
            # Calculate optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_gray, corners, None, **lk_params)
            
            # Filter good points
            good_old = corners[status == 1]
            good_new = new_corners[status == 1]
            
            if len(good_old) > 10:  # Need enough points for reliable estimation
                # Calculate average motion vector
                motion_vectors = good_new - good_old
                avg_motion = np.mean(motion_vectors, axis=0)
                self.camera_motion = tuple(avg_motion)
            else:
                self.camera_motion = (0, 0)
        else:
            self.camera_motion = (0, 0)
            
        self.prev_frame = current_gray
        return self.camera_motion
    
    def compensate_speed(self, track_id, raw_speed, box_center):
        """
        Compensate speed based on camera motion and object behavior
        """
        # Get camera motion magnitude
        camera_speed_px = np.sqrt(self.camera_motion[0]**2 + self.camera_motion[1]**2)
        camera_speed_kmh = camera_speed_px * METER_PER_PIXEL * 3.6  # Convert to km/h
        
        # If camera is moving significantly, adjust speeds
        if camera_speed_kmh > CAMERA_MOTION_THRESHOLD:
            # For objects moving in same direction as camera, subtract camera speed
            # This is a simple approximation - more complex scenarios need perspective correction
            
            # Track object consistency for stationary detection
            if track_id not in self.stationary_objects:
                self.stationary_objects[track_id] = []
            
            self.stationary_objects[track_id].append(raw_speed)
            
            # Keep only recent speed history
            if len(self.stationary_objects[track_id]) > 10:
                self.stationary_objects[track_id] = self.stationary_objects[track_id][-10:]
            
            # Check if object appears stationary relative to background
            recent_speeds = self.stationary_objects[track_id]
            if len(recent_speeds) >= 5:
                avg_speed = np.mean(recent_speeds)
                speed_variance = np.var(recent_speeds)
                
                # If speed is consistent and low variance, likely stationary relative to ground
                if speed_variance < 2.0 and abs(avg_speed - camera_speed_kmh) < STATIONARY_SPEED_THRESHOLD:
                    return 0.0  # Mark as stationary
        
        # Apply basic camera motion compensation
        compensated_speed = max(0, raw_speed - camera_speed_kmh * 0.3)  # Partial compensation
        
        return compensated_speed

def draw_custom_box(frame, x1, y1, x2, y2, track_id, confidence, speed, color):
    """
    Draw custom bounding box with ID, confidence, and speed annotations in one line
    """
    # Draw main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Create single combined label with all info
    combined_text = f"ID:{track_id} {confidence:.2f} {speed:.1f} km/h"
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)
    
    # Calculate text size for the combined label
    (text_w, text_h), _ = cv2.getTextSize(combined_text, font, font_scale, font_thickness)
    
    # Position label above the box
    label_y_start = y1 - 10
    
    # Draw single background rectangle for the entire label
    cv2.rectangle(frame, (x1, label_y_start - text_h - 10), 
                 (x1 + text_w + 15, label_y_start), color, -1)
    
    # Draw border around the label
    cv2.rectangle(frame, (x1, label_y_start - text_h - 10), 
                 (x1 + text_w + 15, label_y_start), (0, 0, 0), 2)
    
    # Draw the combined text
    cv2.putText(frame, combined_text, (x1 + 7, label_y_start - 7), 
               font, font_scale, text_color, font_thickness)

class CustomSpeedEstimator:
    """
    Custom wrapper with motion compensation and improved speed calculation
    """
    def __init__(self, **kwargs):
        # Initialize the base SpeedEstimator
        self.speed_estimator = solutions.SpeedEstimator(**kwargs)
        self.motion_compensator = MotionCompensatedSpeedEstimator(**kwargs)
        
    def process_frame(self, frame):
        """
        Process frame with motion compensation and custom annotations
        """
        # Get original frame copy for custom annotations
        original_frame = frame.copy()
        
        # Detect camera motion
        camera_motion = self.motion_compensator.detect_camera_motion(frame)
        
        # Process with speed estimator
        results = self.speed_estimator(frame)
        
        # Check if we have detections
        if (hasattr(self.speed_estimator, 'boxes') and 
            hasattr(self.speed_estimator, 'track_ids') and 
            len(self.speed_estimator.boxes) > 0):
            
            boxes = self.speed_estimator.boxes
            track_ids = self.speed_estimator.track_ids
            confs = self.speed_estimator.confs
            raw_speeds = self.speed_estimator.spd  # Raw speed dictionary
            
            # Draw custom annotations with compensated speeds
            for box, track_id, conf in zip(boxes, track_ids, confs):
                # Get box coordinates and center
                coords = box.cpu().numpy().ravel().astype(int)
                x1, y1, x2, y2 = coords
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Get raw speed and apply compensation
                raw_speed = raw_speeds.get(track_id, 0.0)
                compensated_speed = self.motion_compensator.compensate_speed(
                    track_id, raw_speed, box_center)
                
                # Get color for this track ID
                color = COLORS(track_id, bgr=True)
                
                # Draw custom box with compensated speed
                draw_custom_box(original_frame, x1, y1, x2, y2, track_id, 
                              float(conf), compensated_speed, color)
        
        # Update results with our custom annotated frame
        results.plot_im = original_frame
        return results
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying speed estimator"""
        return getattr(self.speed_estimator, name)


# def run_object_identifier():
#     """
#     Main function to process video with custom annotations
#     """
#     # 1) Get video properties
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()

#     print(f"Processing video: {VIDEO_PATH}")
#     print(f"Resolution: {w}x{h}, FPS: {fps}, Total frames: {total_frames}")

#     # 2) Setup output video
#     output_path = os.path.join(ROOT, "annotated_ids_speed.avi")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out_vid = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

#     # 3) Initialize custom speed estimator with better parameters
#     estimator = CustomSpeedEstimator(
#         show=False,                  # We'll handle display ourselves
#         model="yolo11n.pt",
#         meter_per_pixel=METER_PER_PIXEL,
#         max_speed=150,               # Reduced max speed for more realistic results
#         max_hist=15,                 # More frames for very stable speed calculation
#         classes=[2],                 # Cars only - add [0,2,3,5,7] for more vehicle types
#         line_width=3,
#         # Adjusted parameters for better detection
#         conf=0.4,                    # Lower confidence threshold for more detections
#         iou=0.7,                     # Better IoU for tracking
#     )
    
#     # 4) Process video and log to CSV
#     with open(OUTPUT_CSV, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "confidence", "speed_kmh"])

#         cap = cv2.VideoCapture(VIDEO_PATH)
#         frame_idx = 0

#         print("Starting video processing...")
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("End of video reached.")
#                 break

#             # Process frame with custom annotations
#             results = estimator.process_frame(frame)

#             # Log detections to CSV
#             if (hasattr(estimator.speed_estimator, 'boxes') and 
#                 len(estimator.speed_estimator.boxes) > 0):
                
#                 for box, track_id, conf in zip(
#                     estimator.speed_estimator.boxes,
#                     estimator.speed_estimator.track_ids,
#                     estimator.speed_estimator.confs
#                 ):
#                     coords = box.cpu().numpy().ravel().astype(int)
#                     x1, y1, x2, y2 = coords
#                     speed = estimator.speed_estimator.spd.get(track_id, 0.0)
                    
#                     writer.writerow([
#                         frame_idx, track_id, 2,  # class 2 = car
#                         x1, y1, x2, y2,
#                         float(conf), round(speed, 2)
#                     ])

#             # Write annotated frame to output video
#             out_vid.write(results.plot_im)
            
#             # Optional: Display frame (comment out if running headless)
#             cv2.imshow('Speed Estimation', results.plot_im)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Processing interrupted by user.")
#                 break
            
#             frame_idx += 1
            
#             # Progress update
#             if frame_idx % 30 == 0:
#                 progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
#                 print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")

#         cap.release()
#         out_vid.release()
#         cv2.destroyAllWindows()

#     print(f"\n✅ Processing complete!")
#     print(f"📁 Output files:")
#     print(f"   • Video: {output_path}")
#     print(f"   • CSV: {OUTPUT_CSV}")
#     print(f"📊 Processed {frame_idx} frames total")


def run_pipeline_obj_id(vid_path : str, vid_name : str, output_csv_path : str, output_video_folder : str):
    """
    Main function to process video with custom annotations
    """
    # 1) Get video properties
    output_csv = os.path.join(VIDEO_DIR, f"{vid_name}.csv")
    cap = cv2.VideoCapture(vid_path)
    assert cap.isOpened(), f"Cannot open {vid_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Processing video: {vid_path}")
    print(f"Resolution: {w}x{h}, FPS: {fps}, Total frames: {total_frames}")

    out_video_filename = f"{vid_name}_annotated_ids_speed.mp4"
    # 2) Setup output video
    output_path = os.path.join(output_video_folder, out_video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

    # 3) Initialize custom speed estimator with better parameters
    estimator = CustomSpeedEstimator(
        show=False,                  # We'll handle display ourselves
        model="yolo11n.pt",
        meter_per_pixel=METER_PER_PIXEL,
        max_speed=150,               # Reduced max speed for more realistic results
        max_hist=15,                 # More frames for very stable speed calculation
        classes=[2],                 # Cars only - add [0,2,3,5,7] for more vehicle types
        line_width=3,
        # Adjusted parameters for better detection
        conf=0.4,                    # Lower confidence threshold for more detections
        iou=0.7,                     # Better IoU for tracking
    )
    
    # 4) Process video and log to CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "confidence", "speed_kmh"])

        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0

        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            # Process frame with custom annotations
            results = estimator.process_frame(frame)

            # Log detections to CSV
            if (hasattr(estimator.speed_estimator, 'boxes') and 
                len(estimator.speed_estimator.boxes) > 0):
                
                for box, track_id, conf in zip(
                    estimator.speed_estimator.boxes,
                    estimator.speed_estimator.track_ids,
                    estimator.speed_estimator.confs
                ):
                    coords = box.cpu().numpy().ravel().astype(int)
                    x1, y1, x2, y2 = coords
                    speed = estimator.speed_estimator.spd.get(track_id, 0.0)
                    
                    writer.writerow([
                        frame_idx, track_id, 2,  # class 2 = car
                        x1, y1, x2, y2,
                        float(conf), round(speed, 2)
                    ])

            # Write annotated frame to output video
            out_vid.write(results.plot_im)
            
            # Optional: Display frame (comment out if running headless)
            cv2.imshow('Speed Estimation', results.plot_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing interrupted by user.")
                break
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")

        cap.release()
        out_vid.release()
        cv2.destroyAllWindows()

    print(f"\n✅ Processing complete!")
    print(f"📁 Output files:")
    print(f"   • Video: {output_path}")
    print(f"   • CSV: {output_csv}")
    print(f"📊 Processed {frame_idx} frames total")