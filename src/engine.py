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
    AuraHand Gesture Engine by Nilesh Sarkar.
    Advanced hand tracking and gesture recognition logic.
    """
    def __init__(self):
        self.scr_w, self.scr_h = pyautogui.size()
        # Smoothing filters for cursor
        self.filter_x = OneEuroFilter(mincutoff=0.01, beta=Config.SMOOTHING_BETA)
        self.filter_y = OneEuroFilter(mincutoff=0.01, beta=Config.SMOOTHING_BETA)
        
        # State Management: WALKTHROUGH -> CALIBRATION -> ACTIVE
        self.state = "WALKTHROUGH" 
        self.tutorial_step = 0
        self.tutorial_start = time.time()
        self.last_click = 0
        self.is_dragging = False
        
        # Hand Data (Landmarks)
        self.right_lm = None
        self.left_lm = None
        
        # Calibration helpers
        self.calib_start = 0
        self.calib_ref_r = None
        self.calib_ref_l = None
        self.calib_time = Config.CALIB_TIME
        
        # Scroll & Zoom helpers
        self.prev_zoom_dist = None
        self.prev_scroll_y = None

        # Tutorial Definitions
        self.tutorials = [
            ("Welcome to AuraHand", "Created by Nilesh Sarkar. Relax and watch."),
            ("Right hand controls", "Move your right hand to move the mouse cursor."),
            ("Pinch to Click", "Pinch right Index and Thumb to Left Click."),
            ("Fist to Drag", "Make a FIST with right hand to grab windows."),
            ("Drag Windows", "Hold the FIST and move to drag stuff accurately."),
            ("Left Hand Scroll", "Left FIST + Right hand Y-move to scroll pages."),
            ("Let's Start", "Prepare for 2-Hand Calibration...")
        ]

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

    def run_walkthrough(self):
        elapsed = time.time() - self.tutorial_start
        if elapsed > 4.0:
            self.tutorial_step += 1
            self.tutorial_start = time.time()
            if self.tutorial_step >= len(self.tutorials):
                self.state = "CALIBRATION"
                return self.run_calibration(self.right_lm, self.left_lm)
        
        main_txt, sub_txt = self.tutorials[self.tutorial_step]
        prog = min(elapsed / 4.0, 1.0)
        return main_txt, sub_txt, Config.UI_INFO, prog

    def run_calibration(self, right_lm, left_lm):
        if not right_lm or not left_lm:
            self.calib_start = 0
            return "Place Both Hands", "Show both hands to begin", Config.UI_WARN, 0.0
        
        curr_r = (right_lm[9].x, right_lm[9].y)
        curr_l = (left_lm[9].x, left_lm[9].y)
        
        if self.calib_start == 0:
            self.calib_start = time.time()
            self.calib_ref_r = curr_r
            self.calib_ref_l = curr_l
            return "Calibrating", "Hold BOTH steady...", Config.UI_INFO, 0.1
            
        dist_r = math.hypot(curr_r[0]-self.calib_ref_r[0], curr_r[1]-self.calib_ref_r[1])
        dist_l = math.hypot(curr_l[0]-self.calib_ref_l[0], curr_l[1]-self.calib_ref_l[1])
        
        if dist_r > Config.CALIB_DRIFT_MAX or dist_l > Config.CALIB_DRIFT_MAX: 
            self.calib_start = time.time()
            self.calib_ref_r = curr_r
            self.calib_ref_l = curr_l
            return "Hand Movement!", "Keep hands very still", (0, 0, 255), 0.0
            
        elapsed = time.time() - self.calib_start
        prog = min(elapsed / Config.CALIB_TIME, 1.0)
        
        if elapsed > Config.CALIB_TIME:
            self.state = "ACTIVE" 
            return "Ready!", "System Optimized - Nilesh Sarkar", Config.UI_ACCENT, 1.0
            
        return "Calibrating", f"Locking sensors... {int(prog*100)}%", Config.UI_INFO, prog

    def process_gestures(self, right_lm, left_lm):
        if self.state == "WALKTHROUGH":
            return self.run_walkthrough()
        if self.state == "CALIBRATION":
            return self.run_calibration(right_lm, left_lm)
        now = time.time()
        main_txt, sub_txt, col, prog = "Active", "Nilesh Sarkar Edition", Config.UI_ACCENT, 0.0
        r_fing, r_pinch, r_fist = self.get_finger_status(right_lm, "Right")
        l_fing, l_pinch, l_fist = self.get_finger_status(left_lm, "Left")
        
        # 1. Double Click
        if r_pinch[1] and l_pinch[1]:
            if now - self.last_click > Config.CLICK_COOLDOWN:
                pyautogui.doubleClick()
                self.last_click = now
            return "Double Click", "System Command", (255, 255, 255), 0.0

        # 2. Advanced Scroll
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

        # 3. Precision Drag & Move
        if right_lm:
            if r_fist:
                if not self.is_dragging:
                    pyautogui.mouseDown()
                    self.is_dragging = True
                self.process_movement(right_lm)
                return "Moving Window", "Hold FIST to carry", Config.UI_WARN, 0.0
            else:
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                
                self.process_movement(right_lm)
                
                if now - self.last_click > Config.CLICK_COOLDOWN:
                    if r_pinch[1]:
                        pyautogui.click()
                        self.last_click = now
                        return "Select", "Index Pinch", Config.UI_ACCENT, 0.0
                    elif r_pinch[2]:
                        pyautogui.rightClick()
                        self.last_click = now
                        return "Options", "Middle Pinch", Config.UI_ACCENT, 0.0
                        
        return main_txt, sub_txt, col, prog
