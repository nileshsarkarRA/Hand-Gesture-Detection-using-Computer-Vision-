import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
import logging
from typing import Optional, Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure proper path for internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import ThreadedCamera
from src.engine import GestureEngine
from src.ui import HUD
from src.config import Config

def draw_landmarks(image, landmarks, color=(0, 255, 120)):
    """Simple replacement for mediapipe drawing_utils."""
    h, w, _ = image.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, color, -1)
    
    # Basic connections (not all, but enough to see)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (5, 6), (6, 7), (7, 8),         # Index
        (9, 10), (10, 11), (11, 12),    # Middle
        (13, 14), (14, 15), (15, 16),   # Ring
        (17, 18), (18, 19), (19, 20),   # Pinky
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # Palm
    ]
    for start, end in connections:
        p1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
        p2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
        cv2.line(image, p1, p2, (220, 220, 220), 1)

def main():
    """
    AuraHand - High-accuracy Hand Gesture OS Control.
    Refactored for performance and modularity.
    """
    # Validate configuration
    try:
        Config.validate()
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Initialize MediaPipe Task-based Hand Landmarker
    try:
        model_path = os.path.join("assets", "hand_landmarker.task")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}. Please run download_model.py first.")
            return

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=Config.MAX_HANDS,
            min_hand_detection_confidence=Config.MEDIAPIPE_CONFIDENCE,
            min_hand_presence_confidence=Config.MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence=Config.MEDIAPIPE_CONFIDENCE
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe HandLandmarker: {e}")
        return
    
    # Initialize Core Modules
    try:
        cam = ThreadedCamera(width=Config.WIDTH, height=Config.HEIGHT, fps=Config.FPS)
        cam.start()
        time.sleep(0.5)  # Wait for camera to stabilize
        if not cam.is_running():
            logger.error("Camera failed to start")
            return
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return
    
    engine = GestureEngine()
    
    # UI State
    show_video = False
    p_time = 0
    fps = 0
    
    logger.info("="*50)
    logger.info("  AURAHAND: GESTURE CONTROL OS")
    logger.info("="*50)
    logger.info("Press [V] to toggle video feed")
    logger.info("Press [Q] to quit safely")
    logger.info("Move mouse to corner to emergency stop")
    logger.info("-"*50)
    
    frame_count = 0
    try:
        while True:
            success, raw_frame = cam.read()
            if not success:
                logger.warning("Failed to read frame")
                continue
            
            frame_count += 1

            # Processing pipeline: Flip -> RGB -> Mediapipe
            frame = cv2.flip(raw_frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to Mediapipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)
            
            right_lm = None
            left_lm = None
            
            # Parse hand landmarks from results
            if results.hand_landmarks:
                for i, lm_list in enumerate(results.hand_landmarks):
                    # Category name is "Left" or "Right"
                    lbl = results.handedness[i][0].category_name
                    if lbl == "Right":
                        right_lm = lm_list
                    else:
                        left_lm = lm_list

            # Analyze gestures and execute OS commands
            # engine.process_gestures returns a tuple (main_text, sub_text, color, progress)
            t1, t2, col, prog = engine.process_gestures(right_lm, left_lm)
            
            # Build Dashboard Output
            if not show_video:
                # Clean Slate UI
                disp = np.zeros((480, 640, 3), dtype=np.uint8)
                # Sophisticated Grid Background
                for i in range(0, 480, 80):
                    cv2.line(disp, (0, i), (640, i), (20, 20, 20), 1)
                for j in range(0, 640, 80):
                    cv2.line(disp, (j, 0), (j, 480), (20, 20, 20), 1)
            else:
                disp = frame.copy()
                # Use custom drawing since mp.solutions is unavailable
                if results.hand_landmarks:
                    for lm_list in results.hand_landmarks:
                        draw_landmarks(disp, lm_list)

            # FPS Calculation for monitoring
            c_time = time.time()
            fps = 1 / (c_time - p_time) if p_time != 0 else 0
            p_time = c_time
            
            # Draw the Dynamic Island HUD
            HUD.render_island(disp, t1, t2, col, progress=prog, fps=fps, show_video=show_video)
            
            # Display the result
            cv2.imshow("AuraHand HUD", disp)
            
            # Check for keyboard interruptions
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit command received")
                break
            if key == ord('v'):
                show_video = not show_video
                logger.info(f"Video feed {'enabled' if show_video else 'disabled'}")
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Shutting down AuraHand...")
        try:
            cam.release()
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames total")
        logger.info("Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
