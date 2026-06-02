import cv2
import numpy as np
import json
import os

class Detector:
    def __init__(self, global_settings: dict, environment_profile: dict):
        """
        Initializes the stage 1 extraction pipeline using configuration settings.
        """
        # 1. Ingest Configuration Constraints
        self.target_width = global_settings["target_width"]
        self.bg_history = global_settings["bg_history"]
        self.bg_threshold = global_settings["bg_threshold"]
        
        self.min_area = environment_profile["min_area"]
        self.max_area = environment_profile["max_area"]
        
        # 2. Instantiate OpenCV's MOG2 Background Subtractor
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history, 
            varThreshold=self.bg_threshold, 
            detectShadows=True
        )
        
        # 3. Create a Morphological Structural Kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Processes a single frame: downscales, subtracts background, cleans noise,
        and returns both the resized frame matrix and a vector of valid object bounding boxes.
        """
        # 1. Resize Frame for performance parity and stable geometric math
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        target_height = int(self.target_width * aspect_ratio)
        resized_frame = cv2.resize(frame, (self.target_width, target_height))
        
        # 2. Apply Background Subtraction
        fg_mask = self.subtractor.apply(resized_frame)
        
        # 3. Filter out shadows and convert to strict binary
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological Closing (Dilation followed by Erosion)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # 5. Extract Outer Contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_bounding_boxes = []
        
        # 6. Apply Domain-Specific Spatial Constraints
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                valid_bounding_boxes.append((x, y, w, h))
                
        # Returns resized_frame along with boxes to support drawing rectangles downstream
        return resized_frame, valid_bounding_boxes
    
class CentroidTracker:
    def __init__(self, global_settings: dict):
        """
        Initializes the Stage 2 identity memory and state tracking system.
        """
        self.memory_limit = global_settings["memory_limit"]
        self.max_distance = global_settings["spatial_proximity_threshold"]
        
        self.next_object_id = 0    
        self.objects = {}          
        self.disappeared = {}      
        
    def register(self, centroid: tuple):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects: list) -> dict:
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] >= self.memory_limit:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
            return self.objects
        
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] >= self.memory_limit:
                self.deregister(object_id)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects
    
def run_engine(domain_key: str, video_source):
    # 1. Load decoupled configuration layers
    with open("engine_settings.json", "r") as f:
        global_settings = json.load(f)["global_settings"]
        
    with open("environment_profiles.json", "r") as f:
        profile = json.load(f)[domain_key]
        
    # Extract Gatekeeper metrics labels and thresholds
    threshold_y = profile["threshold_y"]
    label_inbound = profile["labels"][0]
    label_outbound = profile["labels"][1]

    # UPGRADE: Ingest the buffer offset from config with a fallback safety default
    buffer_offset = global_settings.get("buffer_offset", 15)  
    line_top = threshold_y - buffer_offset
    line_bottom = threshold_y + buffer_offset

    # 2. Instantiate pipeline engines
    detector = Detector(global_settings, profile)
    tracker = CentroidTracker(global_settings)
    
    # State tracking metrics
    track_state_history = {} 
    counts = {label_inbound: 0, label_outbound: 0}

    # Open video capture stream
    cap = cv2.VideoCapture(video_source)
    print(f"\n[INFO] Initialized GGE with Domain: {domain_key.upper()}")
    print("[INFO] Press 'q' with the video window selected to stop execution.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process through Stage 1 & Scale
        processed_frame, rects = detector.process_frame(frame)
        
        # UPGRADE: Draw full visual bounding box rectangles over raw objects
        for (x, y, w, h) in rects:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # Process through Stage 2
        tracked_objects = tracker.update(rects)

        # UPGRADE: Render double-line hysteresis gates instead of a single baseline wire
        cv2.line(processed_frame, (0, line_top), (processed_frame.shape[1], line_top), (0, 165, 255), 1)     # Orange Top
        cv2.line(processed_frame, (0, line_bottom), (processed_frame.shape[1], line_bottom), (0, 255, 255), 1) # Yellow Bottom
        cv2.putText(processed_frame, "GATEKEEPER BUFFER ZONE", (10, line_top - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Evaluate Upgraded Stage 3: Hysteresis Tracking Math Loop
        for object_id, centroid in tracked_objects.items():
            cX, cY = centroid
            
            # Record track entry orientations
            if object_id not in track_state_history:
                if cY < line_top:
                    track_state_history[object_id] = "entered_above"
                elif cY > line_bottom:
                    track_state_history[object_id] = "entered_below"
                else:
                    track_state_history[object_id] = "initialized_inside_buffer"
            else:
                current_state = track_state_history[object_id]
                
                # Check Inbound Trigger: Commenced above buffer, completely broke through bottom line
                if current_state == "entered_above" and cY >= line_bottom:
                    counts[label_inbound] += 1
                    track_state_history[object_id] = "counted" # Spend the state token
                    
                # Check Outbound Trigger: Commenced below buffer, completely broke through top line
                elif current_state == "entered_below" and cY <= line_top:
                    counts[label_outbound] += 1
                    track_state_history[object_id] = "counted" # Spend the state token

            # Render UI overlays per active element
            cv2.circle(processed_frame, (cX, cY), 4, (0, 255, 0), -1)
            cv2.putText(processed_frame, f"ID {object_id}", (cX - 10, cY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Clean old IDs from track history if they were wiped by tracker memory limits
        for missing_id in list(track_state_history.keys()):
            if missing_id not in tracked_objects:
                del track_state_history[missing_id]

        # Draw dynamic statistics scorecard dashboard
        cv2.rectangle(processed_frame, (10, 10), (240, 75), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"{label_inbound}: {counts[label_inbound]}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"{label_outbound}: {counts[label_outbound]}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the live window
        cv2.imshow("Geometric Gatekeeper Engine (GGE)", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Terminal Summary Report Print block
    print("\n" + "=" * 40)
    print("         FINAL COUNTS SUMMARY REPORT")
    print("=" * 40)
    print(f" Profile Domain : {domain_key.upper()}")
    print(f" {label_inbound.ljust(15)}: {counts[label_inbound]}")
    print(f" {label_outbound.ljust(15)}: {counts[label_outbound]}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    print("=" * 50)
    print("   GEOMETRIC GATEKEEPER ENGINE TERMINAL TESTER")
    print("=" * 50)
    
    # 1. Interactive Domain Profile Selection
    try:
        with open("environment_profiles.json", "r") as f:
            available_profiles = list(json.load(f).keys())
    except FileNotFoundError:
        available_profiles = ["highway_traffic", "warehouse_safety"]
        
    print(f"Available Environment Profiles: {available_profiles}")
    selected_domain = input(f"Enter profile name (default: '{available_profiles[0]}'): ").strip()
    if not selected_domain or selected_domain not in available_profiles:
        selected_domain = available_profiles[0]
        
    # 2. Interactive Video Source Selection
    print("\nVideo Source Options:")
    print("  - Type '0' to launch your primary live hardware webcam.")
    print("  - Type a video file name (e.g., 'sample_video.mp4').")
    raw_source = input("Enter video source selection: ").strip()
    
    if not raw_source:
        raw_source = "sample_video.mp4"
        
    if raw_source.isdigit():
        video_source_selection = int(raw_source)
    else:
        video_source_selection = raw_source

    # 3. Fire up the execution engine
    run_engine(selected_domain, video_source_selection)