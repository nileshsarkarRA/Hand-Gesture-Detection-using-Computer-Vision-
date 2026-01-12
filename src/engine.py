import math
import time
import logging
from typing import Optional, Tuple, List
import pyautogui
from src.filters import OneEuroFilter
from src.config import Config

logger = logging.getLogger(__name__)

class GestureEngine:
    """
    GestureFlow Gesture Engine by Nilesh Sarkar.
    Advanced hand tracking and gesture recognition logic.
    """
    def __init__(self):
        self.scr_w, self.scr_h = pyautogui.size()
        # Smoothing filters for cursor
        self.filter_x = OneEuroFilter(mincutoff=0.01, beta=Config.SMOOTHING_BETA)
        self.filter_y = OneEuroFilter(mincutoff=0.01, beta=Config.SMOOTHING_BETA)
        
        # State Management: CALIBRATION -> ACTIVE
        self.state = "CALIBRATION"  # Start directly with calibration
        self.last_click = 0
        self.is_dragging = False
        
        # Hand Data (Landmarks)
        self.right_lm = None
        self.left_lm = None
        
        # Enhanced Calibration System
        self.calib_start = 0
        self.calib_ref_r = None
        self.calib_ref_l = None
        self.calib_time = Config.CALIB_TIME
        self.calib_samples = []  # Store samples for quality check
        self.calib_lighting_ok = False
        self.calib_hand_quality_ok = False
        
        # Scroll & Zoom helpers
        self.prev_zoom_dist = None
        self.prev_scroll_y = None
        
        # Sustained pinch detection for click stabilization
        self.left_pinch_start = 0   # Track when left pinch started
        self.right_pinch_start = 0  # Track when right pinch started
        
        logger.info("GestureFlow Engine initialized - Starting calibration mode")

    def update_data(self, right_lm, left_lm):
        self.right_lm = right_lm
        self.left_lm = left_lm

    def get_finger_status(self, lm, label):
        """Analyzes finger states: [thumb, index, middle, ring, pinky]."""
        if not lm: return [0]*5, [0]*5, False
        
        fingers = [0]*5
        pinches = [0]*5
        thumb_tip = (lm[4].x, lm[4].y)
        
        if label == "Right":
            fingers[0] = 1 if lm[4].x < lm[2].x else 0
        else:
            fingers[0] = 1 if lm[4].x > lm[2].x else 0
            
        for i, tip_id in enumerate([8, 12, 16, 20]):
            pip_id = tip_id - 2
            fingers[i+1] = 1 if lm[tip_id].y < lm[pip_id].y else 0
            
            tip = (lm[tip_id].x, lm[tip_id].y)
            dist = math.hypot(tip[0]-thumb_tip[0], tip[1]-thumb_tip[1])
            pinches[i+1] = 1 if dist < Config.PINCH_THRESH else 0
            
        fist = sum(fingers[1:]) == 0
        return fingers, pinches, fist

    def process_movement(self, lm):
        """Maps hand knuckle to screen coordinates with smoothing."""
        node = lm[9]
        margin = 0.15
        x = (node.x - margin) / (1 - 2*margin)
        y = (node.y - margin) / (1 - 2*margin)
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        x = (x - 0.5) * Config.TRACKING_SCALE + 0.5
        y = (y - 0.5) * Config.TRACKING_SCALE + 0.5
        sx = self.filter_x.filter(max(0, min(1, x)) * self.scr_w)
        sy = self.filter_y.filter(max(0, min(1, y)) * self.scr_h)
        pyautogui.moveTo(sx, sy, _pause=False)
        return int(sx), int(sy)

    def check_lighting_quality(self, frame):
        """Analyze frame brightness to ensure good lighting conditions."""
        if frame is None:
            return False, 0
        
        # Convert to grayscale and calculate mean brightness
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        
        # Optimal range: 80-180 (0-255 scale)
        # Too dark < 60, Too bright > 200
        if brightness < 60:
            return False, brightness  # Too dark
        elif brightness > 200:
            return False, brightness  # Too bright
        elif 80 <= brightness <= 180:
            return True, brightness   # Optimal
        else:
            return True, brightness   # Acceptable

    def check_hand_quality(self, right_lm, left_lm, handedness_data=None):
        """Check if both hands are properly detected with good confidence."""
        if not right_lm or not left_lm:
            return False, "Missing hands"
        
        # Check if we have all 21 landmarks for each hand
        if len(right_lm) < 21 or len(left_lm) < 21:
            return False, "Incomplete hand data"
        
        # Check landmark visibility (z-depth shouldn't be too far)
        right_depths = [lm.z for lm in right_lm]
        left_depths = [lm.z for lm in left_lm]
        
        # If hands are too far away (z > 0.1), tracking may be poor
        if max(right_depths) > 0.15 or max(left_depths) > 0.15:
            return False, "Hands too far from camera"
        
        return True, "Good quality"

    def run_enhanced_calibration(self, right_lm, left_lm, frame=None):
        """
        Enhanced calibration mode with environmental quality checks.
        - Checks lighting conditions
        - Validates hand detection quality
        - Ensures tracking stability
        - Multi-point validation for accuracy
        """
        
        # Step 1: Check lighting quality
        if frame is not None:
            lighting_ok, brightness = self.check_lighting_quality(frame)
            self.calib_lighting_ok = lighting_ok
            
            if not lighting_ok:
                if brightness < 60:
                    return "Poor Lighting", f"Too dark ({int(brightness)}/255) - Add more light", (0, 100, 255), 0.0
                elif brightness > 200:
                    return "Poor Lighting", f"Too bright ({int(brightness)}/255) - Reduce light", (0, 100, 255), 0.0
        
        # Step 2: Check hand presence and quality
        hand_quality_ok, quality_msg = self.check_hand_quality(right_lm, left_lm)
        self.calib_hand_quality_ok = hand_quality_ok
        
        if not hand_quality_ok:
            self.calib_start = 0
            return "Position Both Hands", quality_msg, Config.UI_WARN, 0.0
        
        # Step 3: Calculate current hand positions
        curr_r = (right_lm[9].x, right_lm[9].y, right_lm[9].z)
        curr_l = (left_lm[9].x, left_lm[9].y, left_lm[9].z)
        
        # Step 4: Initialize calibration reference
        if self.calib_start == 0:
            self.calib_start = time.time()
            self.calib_ref_r = curr_r
            self.calib_ref_l = curr_l
            self.calib_samples = []
            return "Calibrating", "Hold BOTH hands steady...", Config.UI_INFO, 0.1
        
        # Step 5: Check stability (minimal hand movement)
        dist_r = math.hypot(curr_r[0]-self.calib_ref_r[0], curr_r[1]-self.calib_ref_r[1])
        dist_l = math.hypot(curr_l[0]-self.calib_ref_l[0], curr_l[1]-self.calib_ref_l[1])
        
        # If hands moved too much, restart calibration
        if dist_r > Config.CALIB_DRIFT_MAX or dist_l > Config.CALIB_DRIFT_MAX:
            self.calib_start = time.time()
            self.calib_ref_r = curr_r
            self.calib_ref_l = curr_l
            self.calib_samples = []
            return "Hand Movement Detected", "Keep hands very still", (0, 0, 255), 0.0
        
        # Step 6: Collect calibration samples for quality validation
        elapsed = time.time() - self.calib_start
        self.calib_samples.append({
            'right': curr_r,
            'left': curr_l,
            'time': elapsed
        })
        
        prog = min(elapsed / Config.CALIB_TIME, 1.0)
        
        # Step 7: Complete calibration with quality validation
        if elapsed > Config.CALIB_TIME:
            # Validate collected samples for consistency
            if len(self.calib_samples) >= 10:  # Minimum samples
                # Calculate variance in samples
                r_variance = sum([
                    math.hypot(s['right'][0] - curr_r[0], s['right'][1] - curr_r[1])
                    for s in self.calib_samples[-10:]
                ]) / 10
                
                l_variance = sum([
                    math.hypot(s['left'][0] - curr_l[0], s['left'][1] - curr_l[1])
                    for s in self.calib_samples[-10:]
                ]) / 10
                
                # If variance is too high, tracking is unstable
                if r_variance > 0.02 or l_variance > 0.02:
                    self.calib_start = 0
                    self.calib_samples = []
                    return "Unstable Tracking", "Try better lighting or camera position", (0, 100, 255), 0.0
            
            # Calibration successful!
            self.state = "ACTIVE"
            logger.info(f"Calibration complete - Lighting: {brightness:.1f}, Samples: {len(self.calib_samples)}")
            return "System Ready!", "GestureFlow Calibrated Successfully", Config.UI_ACCENT, 1.0
        
        # Step 8: Display calibration progress
        return "Calibrating", f"Optimizing tracking... {int(prog*100)}%", Config.UI_INFO, prog

    def process_gestures(self, right_lm, left_lm, frame=None):
        """Main gesture processing with frame for lighting analysis."""
        if self.state == "CALIBRATION":
            return self.run_enhanced_calibration(right_lm, left_lm, frame)
        
        now = time.time()
        main_txt, sub_txt, col, prog = "Active", "GestureFlow Ready", Config.UI_ACCENT, 0.0
        r_fing, r_pinch, r_fist = self.get_finger_status(right_lm, "Right")
        l_fing, l_pinch, l_fist = self.get_finger_status(left_lm, "Left")
        
        # 1. Double Click - Both hands index pinch together
        if r_pinch[1] and l_pinch[1]:
            if now - self.last_click > Config.CLICK_COOLDOWN:
                pyautogui.doubleClick()
                self.last_click = now
                # Reset pinch timers
                self.left_pinch_start = 0
                self.right_pinch_start = 0
            return "Double Click", "Both Hands", (255, 255, 255), 0.0

        # 2. Advanced Scroll (left fist + right hand vertical movement)
        if l_fist and right_lm:
            ry = right_lm[9].y
            if self.prev_scroll_y is not None:
                dy = (self.prev_scroll_y - ry)
                if abs(dy) > 0.002:
                    pyautogui.scroll(int(dy * Config.SCROLL_SPEED * 10))
            self.prev_scroll_y = ry
            return "Scrolling", "Release left fist to stop", Config.UI_INFO, 0.0
        else:
            self.prev_scroll_y = None

        # 3. Cursor Movement & Drag (Right Hand)
        if right_lm:
            # Drag mode - right fist
            if r_fist:
                if not self.is_dragging:
                    pyautogui.mouseDown()
                    self.is_dragging = True
                self.process_movement(right_lm)
                return "Dragging", "Hold FIST to carry", Config.UI_WARN, 0.0
            else:
                # Release drag if was dragging
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                
                # Normal cursor movement
                self.process_movement(right_lm)
                
                # RIGHT CLICK - Right hand index + thumb pinch (sustained)
                if now - self.last_click > Config.CLICK_COOLDOWN:
                    if r_pinch[1]:  # Right hand index pinch detected
                        # Start tracking pinch time
                        if self.right_pinch_start == 0:
                            self.right_pinch_start = now
                        
                        # Check if pinch held long enough for stability
                        pinch_duration = now - self.right_pinch_start
                        if pinch_duration >= Config.PINCH_HOLD_TIME:
                            pyautogui.rightClick()
                            self.last_click = now
                            self.right_pinch_start = 0
                            return "Right Click", "Right Hand", Config.UI_ACCENT, 0.0
                        else:
                            # Show holding progress
                            prog = min(pinch_duration / Config.PINCH_HOLD_TIME, 1.0)
                            return "Right Click", f"Hold... {int(prog*100)}%", Config.UI_INFO, prog
                    else:
                        # Pinch released, reset timer
                        self.right_pinch_start = 0
        
        # 4. LEFT CLICK - Left hand index + thumb pinch (sustained)
        if left_lm and not l_fist:  # Only if not already in fist mode
            if now - self.last_click > Config.CLICK_COOLDOWN:
                if l_pinch[1]:  # Left hand index pinch detected
                    # Start tracking pinch time
                    if self.left_pinch_start == 0:
                        self.left_pinch_start = now
                    
                    # Check if pinch held long enough for stability
                    pinch_duration = now - self.left_pinch_start
                    if pinch_duration >= Config.PINCH_HOLD_TIME:
                        pyautogui.click()
                        self.last_click = now
                        self.left_pinch_start = 0
                        return "Left Click", "Left Hand", Config.UI_ACCENT, 0.0
                    else:
                        # Show holding progress
                        prog = min(pinch_duration / Config.PINCH_HOLD_TIME, 1.0)
                        return "Left Click", f"Hold... {int(prog*100)}%", Config.UI_INFO, prog
                else:
                    # Pinch released, reset timer
                    self.left_pinch_start = 0
                    
        return main_txt, sub_txt, col, prog
