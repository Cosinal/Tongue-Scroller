#!/usr/bin/env python3
"""
tonguescroll.py
Desktop prototype for hands-free scrolling by sticking out tongue.
Detects HEAD SHAKING with tongue out to show GIF overlay.
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Constants
CALIBRATION_FRAMES = 30
MOUTH_RATIO_MULTIPLIER = 1.6
TONGUE_COLOR_THRESHOLD = 0.30
HEAD_SHAKE_WINDOW = 1.0  # Time window to detect shake
HEAD_SHAKE_THRESHOLD = 3  # Number of direction changes needed
HEAD_SHAKE_MOVEMENT_THRESHOLD = 0.03  # Minimum head movement (% of screen width)

# GIF file path
GIF_PATH = "flight-reacts.gif"


class GIFPlayer:
    """Class to load and play GIF frames."""
    
    def __init__(self, gif_path):
        self.frames = []
        self.current_frame = 0
        self.frame_delay = 0.05
        self.last_frame_time = time.time()
        self.load_gif(gif_path)
    
    def load_gif(self, gif_path):
        """Load all frames from GIF file."""
        paths_to_try = [
            gif_path,
            os.path.join(os.path.dirname(__file__), gif_path),
            "/mnt/user-data/uploads/flight-reacts.gif",
            "C:\\Users\\j.shaw\\sus\\flight-reacts.gif"
        ]
        
        gif_loaded = False
        for path in paths_to_try:
            if os.path.exists(path):
                print(f"Trying to load GIF from: {path}")
                cap = cv2.VideoCapture(path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self.frames.append(frame)
                
                cap.release()
                
                if len(self.frames) > 0:
                    print(f"✓ Successfully loaded GIF with {len(self.frames)} frames")
                    gif_loaded = True
                    break
        
        if not gif_loaded:
            print(f"WARNING: Could not load GIF!")
            print("Continuing without GIF overlay...")
    
    def get_next_frame(self):
        """Get next frame in GIF sequence."""
        if len(self.frames) == 0:
            return None
        
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_delay:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.last_frame_time = current_time
        
        return self.frames[self.current_frame]
    
    def reset(self):
        """Reset to first frame."""
        self.current_frame = 0


class HeadShakeDetector:
    """Detect head shaking (side-to-side movement)."""
    
    def __init__(self, window_size=1.0, threshold=3, movement_threshold=0.03):
        self.window_size = window_size
        self.threshold = threshold
        self.movement_threshold = movement_threshold
        self.positions = deque()  # Store (timestamp, x_position)
        self.direction_changes = 0
        self.last_direction = None
    
    def update(self, head_x_position, timestamp):
        """
        Update with new head position and check for shaking.
        
        Args:
            head_x_position: normalized x position of head center (0-1)
            timestamp: current time in seconds
        
        Returns:
            True if head shaking detected, False otherwise
        """
        self.positions.append((timestamp, head_x_position))
        
        # Remove old positions outside time window
        while self.positions and timestamp - self.positions[0][0] > self.window_size:
            self.positions.popleft()
        
        if len(self.positions) < 3:
            return False
        
        # Check for direction changes
        recent = list(self.positions)
        
        for i in range(len(recent) - 1):
            prev_x = recent[i][1]
            curr_x = recent[i + 1][1]
            movement = curr_x - prev_x
            
            # Detect significant movement
            if abs(movement) > self.movement_threshold:
                current_direction = "right" if movement > 0 else "left"
                
                # Count direction change
                if self.last_direction is not None and self.last_direction != current_direction:
                    self.direction_changes += 1
                    print(f"🔄 Head direction change! Total: {self.direction_changes}")
                
                self.last_direction = current_direction
        
        # Check if shaking
        is_shaking = self.direction_changes >= self.threshold
        
        if is_shaking:
            print(f"🎉 HEAD SHAKE DETECTED! (changes: {self.direction_changes})")
        
        return is_shaking
    
    def reset(self):
        """Reset shake detection."""
        self.positions.clear()
        self.last_direction = None
        self.direction_changes = 0


def get_mouth_and_head_metrics(landmarks, frame):
    """
    Calculate mouth opening, tongue color, and head position.
    
    Args:
        landmarks: MediaPipe face landmarks
        frame: current video frame (BGR)
    
    Returns:
        (mouth_ratio, tongue_color_score, head_x_position)
    """
    h, w, _ = frame.shape
    
    # Mouth landmarks
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    
    # Head position (use nose tip)
    nose_tip = landmarks[1]
    head_x_position = nose_tip.x
    
    upper_y = upper_lip.y * h
    lower_y = lower_lip.y * h
    left_x = left_corner.x * w
    right_x = right_corner.x * w
    
    # Calculate mouth opening ratio
    vertical_dist = abs(lower_y - upper_y)
    horizontal_dist = abs(right_x - left_x)
    
    if horizontal_dist < 1:
        mouth_ratio = 0
    else:
        mouth_ratio = vertical_dist / horizontal_dist
    
    # ROI for tongue color detection
    center_x_px = int((left_x + right_x) / 2)
    center_y_px = int((upper_y + lower_y) / 2)
    
    roi_size_x = int(horizontal_dist * 0.6)
    roi_size_y = int(vertical_dist * 0.7)
    
    x1 = max(0, center_x_px - roi_size_x // 2)
    x2 = min(w, center_x_px + roi_size_x // 2)
    y1 = max(0, int(center_y_px - roi_size_y * 0.2))
    y2 = min(h, int(center_y_px + roi_size_y * 0.8))
    
    mouth_roi = frame[y1:y2, x1:x2]
    
    tongue_color_score = 0
    
    if mouth_roi.size > 0:
        hsv_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
        
        # Detect red/pink tongue color
        lower_red1 = np.array([0, 40, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 60])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        tongue_color_score = np.count_nonzero(mask) / mask.size
    
    return mouth_ratio, tongue_color_score, head_x_position


def detect_tongue(mouth_ratio, tongue_color_score, baseline_ratio):
    """Detect if tongue is sticking out."""
    mouth_open_enough = mouth_ratio > baseline_ratio * MOUTH_RATIO_MULTIPLIER
    tongue_color_strong = tongue_color_score > TONGUE_COLOR_THRESHOLD
    return mouth_open_enough and tongue_color_strong


def calibrate_baseline(cap, face_mesh):
    """Calibrate baseline mouth opening ratio."""
    print("Calibrating... Please keep your mouth closed and face neutral.")
    print(f"Capturing {CALIBRATION_FRAMES} frames...")
    
    mouth_ratios = []
    
    for i in range(CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mouth_ratio, _, _ = get_mouth_and_head_metrics(landmarks, frame)
            mouth_ratios.append(mouth_ratio)
            
            cv2.putText(frame, f"Calibrating: {i+1}/{CALIBRATION_FRAMES}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Tongue Scroll', frame)
            cv2.waitKey(1)
    
    if len(mouth_ratios) == 0:
        print("Warning: No face detected during calibration. Using default baseline.")
        return 0.1
    
    baseline = np.mean(mouth_ratios)
    print(f"Calibration complete! Baseline mouth ratio: {baseline:.4f}")
    print("\n=== Tongue Scroll Active ===")
    print("Tongue OUT → Next TikTok")
    print("Tongue OUT + SHAKE YOUR HEAD → Show GIF")
    print("Press 'q' to quit\n")
    return baseline


def main():
    """Main loop for tongue and head shake detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    gif_player = GIFPlayer(GIF_PATH)
    
    head_shake_detector = HeadShakeDetector(
        window_size=HEAD_SHAKE_WINDOW,
        threshold=HEAD_SHAKE_THRESHOLD,
        movement_threshold=HEAD_SHAKE_MOVEMENT_THRESHOLD
    )
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        baseline_mouth_ratio = calibrate_baseline(cap, face_mesh)
        
        previous_tongue_detected = False
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        shaking = False
        last_scroll_time = 0
        scroll_cooldown = 0.3  # Prevent rapid scrolling
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            current_tongue_detected = False
            current_time = time.time()
            display_frame = frame.copy()
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                mouth_ratio, tongue_color_score, head_x = get_mouth_and_head_metrics(landmarks, frame)
                
                # Detect tongue
                current_tongue_detected = detect_tongue(
                    mouth_ratio, tongue_color_score, baseline_mouth_ratio
                )
                
                # If tongue is out, track head movement for shake detection
                if current_tongue_detected:
                    shaking = head_shake_detector.update(head_x, current_time)
                else:
                    head_shake_detector.reset()
                    shaking = False
                
                # SCROLL: Only if tongue out AND not shaking AND cooldown passed
                if (current_tongue_detected and 
                    not previous_tongue_detected and 
                    not shaking and 
                    current_time - last_scroll_time > scroll_cooldown):
                    pyautogui.press('down')
                    last_scroll_time = current_time
                    print(f"⬇️ NEXT VIDEO!")
                
                previous_tongue_detected = current_tongue_detected
            else:
                head_shake_detector.reset()
                shaking = False
                previous_tongue_detected = False
            
            # Replace video with GIF if shaking with tongue out
            if shaking and current_tongue_detected:
                gif_frame = gif_player.get_next_frame()
                if gif_frame is not None:
                    h, w = frame.shape[:2]
                    gif_frame_resized = cv2.resize(gif_frame, (w, h))
                    display_frame = gif_frame_resized
            else:
                gif_player.reset()
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
            
            # Show FPS
            cv2.putText(display_frame, f"FPS: {fps:.0f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show status
            if shaking and current_tongue_detected:
                cv2.putText(display_frame, "HEAD SHAKING!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Tongue Scroll', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()