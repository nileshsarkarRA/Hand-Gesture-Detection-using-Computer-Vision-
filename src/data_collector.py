#!/usr/bin/env python3
"""
GestureFlow Data Collector
Record hand gesture samples for custom ML model training
"""

import cv2
import mediapipe as mp
import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gesture classes
GESTURES = {
    '1': 'idle',
    '2': 'left_click',
    '3': 'right_click',
    '4': 'double_click',
    '5': 'drag',
    '6': 'scroll',
    '7': 'cursor_move'
}

class DataCollector:
    def __init__(self, output_dir='gestures_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each gesture
        for gesture in GESTURES.values():
            (self.output_dir / gesture).mkdir(exist_ok=True)
        
        # Initialize MediaPipe
        model_path = "assets/hand_landmarker.task"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Recording state
        self.recording = False
        self.current_gesture = None
        self.recorded_frames = []
        self.sample_counts = {g: 0 for g in GESTURES.values()}
        
        logger.info("Data Collector initialized")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def landmarks_to_dict(self, hand_landmarks, handedness):
        """Convert landmarks to dictionary format"""
        if not hand_landmarks:
            return None
        
        data = {
            'handedness': handedness,
            'landmarks': []
        }
        
        for lm in hand_landmarks:
            data['landmarks'].append({
                'x': float(lm.x),
                'y': float(lm.y),
                'z': float(lm.z)
            })
        
        return data
    
    def save_sample(self):
        """Save recorded frames to JSON file"""
        if not self.recorded_frames or not self.current_gesture:
            logger.warning("No frames to save")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"seq_{timestamp}.json"
        filepath = self.output_dir / self.current_gesture / filename
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.recorded_frames, f, indent=2)
        
        self.sample_counts[self.current_gesture] += 1
        logger.info(f"✅ Saved {len(self.recorded_frames)} frames to {filepath}")
        logger.info(f"   Total {self.current_gesture} samples: {self.sample_counts[self.current_gesture]}")
        
        # Reset
        self.recorded_frames = []
        self.recording = False
    
    def draw_ui(self, frame):
        """Draw user interface on frame"""
        h, w = frame.shape[:2]
        
        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "GestureFlow Data Collector", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Instructions
        y = 70
        instructions = [
            "Press 1-7 to select gesture, SPACE to record, S to save, Q to quit",
            "",
            "Gestures: 1=Idle 2=LeftClick 3=RightClick 4=DoubleClick",
            "          5=Drag 6=Scroll 7=CursorMove"
        ]
        
        for line in instructions:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, (200, 200, 200), 1)
            y += 25
        
        # Current status
        if self.current_gesture:
            status = f"Current: {self.current_gesture.upper()}"
            color = (0, 0, 255) if self.recording else (0, 255, 255)
            cv2.putText(frame, status, (20, h-60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
        
        if self.recording:
            cv2.putText(frame, f"RECORDING... ({len(self.recorded_frames)} frames)",
                       (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Recording indicator
            cv2.circle(frame, (w-30, 30), 10, (0, 0, 255), -1)
        
        # Sample counts
        count_y = h - 150
        cv2.putText(frame, "Sample Counts:", (w-200, count_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        count_y += 20
        for gesture, count in self.sample_counts.items():
            cv2.putText(frame, f"{gesture}: {count}", (w-200, count_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            count_y += 18
    
    def run(self):
        """Main collection loop"""
        logger.info("\n" + "="*60)
        logger.info("GESTUREFLOW DATA COLLECTOR")
        logger.info("="*60)
        logger.info("\nControls:")
        logger.info("  1-7: Select gesture type")
        logger.info("  SPACE: Start/stop recording")
        logger.info("  S: Save current recording")
        logger.info("  Q: Quit")
        logger.info("\nTarget: 50+ samples per gesture\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            results = self.detector.detect(mp_image)
            
            # Draw hand landmarks
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            
            # Record frame if recording
            if self.recording and results.hand_landmarks:
                frame_data = {
                    'timestamp': time.time(),
                    'hands': []
                }
                
                for i, hand_lm in enumerate(results.hand_landmarks):
                    handedness = results.handedness[i][0].category_name if results.handedness else 'Unknown'
                    hand_dict = self.landmarks_to_dict(hand_lm, handedness)
                    if hand_dict:
                        frame_data['hands'].append(hand_dict)
                
                self.recorded_frames.append(frame_data)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Display
            cv2.imshow('Data Collector', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key in [ord(k) for k in GESTURES.keys()]:
                gesture_key = chr(key)
                self.current_gesture = GESTURES[gesture_key]
                logger.info(f"Selected gesture: {self.current_gesture}")
            elif key == ord(' '):  # Space to toggle recording
                if self.current_gesture is None:
                    logger.warning("Select a gesture first (press 1-7)")
                else:
                    self.recording = not self.recording
                    if self.recording:
                        self.recorded_frames = []
                        logger.info(f"🔴 Recording {self.current_gesture}...")
                    else:
                        logger.info(f"⏸️  Recording paused")
            elif key == ord('s'):  # Save
                self.save_sample()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        logger.info("\n" + "="*60)
        logger.info("Collection Summary:")
        for gesture, count in self.sample_counts.items():
            logger.info(f"  {gesture}: {count} samples")
        logger.info("="*60)

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
