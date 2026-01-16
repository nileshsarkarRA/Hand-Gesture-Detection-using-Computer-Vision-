import time
import math
import pyautogui
from .config import Config
from .filters import OneEuroFilter
from .geometry import GeometryEngine

class CursorController:
    """
    Core state machine that translates hand landmarks into OS-level mouse actions.
    Coordinates filtering, gesture recognition (clicks, scrolls), and movement.
    """
    def __init__(self):
        # Smoothing filters for stable X/Y movement.
        self.filter_x = OneEuroFilter(Config.FPS, Config.ONE_EURO_MIN_CUTOFF, Config.ONE_EURO_BETA)
        self.filter_y = OneEuroFilter(Config.FPS, Config.ONE_EURO_MIN_CUTOFF, Config.ONE_EURO_BETA)
        
        # State tracking for ongoing interactions.
        self.is_dragging = False
        self.is_scrolling = False
        self.pinch_start_time = 0
        self.pinch_active = False

    def process_hand(self, hand_landmarks):
        """
        Processing Pipeline:
        1. Extract relevant landmark points.
        2. Analyze finger proximity for pinch/click detection.
        3. Apply smoothing and precision damping (Sniper Mode).
        4. Recognize complex gestures (scrolling vs. moving).
        5. Trigger OS mouse events via PyAutoGUI.
        """
        
        # --- 1. Identify Key Landmarks ---
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        
        # Using the middle knuckle (node 9) for stabilized cursor movement.
        tracker_node = hand_landmarks.landmark[9] 
        
        # --- 2. Distance Analysis ---
        dist_pinch = GeometryEngine.get_distance(thumb, index)
        
        # --- 3. Coordinate Mapping & Smoothing ---
        target_x, target_y, roi_rect = GeometryEngine.map_coordinates(
            tracker_node.x, tracker_node.y, Config.TILT_ANGLE_DEG
        )
        
        # Precision Mode: Reduce responsiveness when fingers are near clicking range.
        current_beta = Config.ONE_EURO_BETA
        if dist_pinch < Config.PINCH_STABILIZE_DIST:
             factor = max(0.1, (dist_pinch / Config.PINCH_STABILIZE_DIST))
             current_beta = Config.ONE_EURO_BETA * factor * Config.STABILIZE_FACTOR

        smooth_x = self.filter_x.filter(target_x, custom_beta=current_beta)
        smooth_y = self.filter_y.filter(target_y, custom_beta=current_beta)

        # --- 4. Gesture Detection ---
        
        # A. Scroll Recognition
        # Criteria: Index and Middle fingers extended, Ring finger tucked.
        idx_up = index.y < hand_landmarks.landmark[6].y
        mid_up = middle.y < hand_landmarks.landmark[10].y
        ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
        
        if idx_up and mid_up and ring_down:
            return self._handle_scroll(hand_landmarks, smooth_x, smooth_y)
            
        self.is_scrolling = False
        
        # B. Click & Drag (Hysteresis-based Schmitt Trigger)
        # Prevents unintended "double-clicking" at the sensitivity threshold.
        if not self.pinch_active and dist_pinch < Config.PINCH_DOWN_THRESH:
            self.pinch_active = True
            self.pinch_start_time = time.time()
            
        elif self.pinch_active and dist_pinch > Config.PINCH_UP_THRESH:
            self.pinch_active = False
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
            else:
                # Fast tap results in a single click.
                if time.time() - self.pinch_start_time < Config.DRAG_ACTIVATION_TIME:
                    pyautogui.click()
        
        # Persistent hold results in a drag operation.
        if self.pinch_active:
            if not self.is_dragging and (time.time() - self.pinch_start_time > Config.DRAG_ACTIVATION_TIME):
                pyautogui.mouseDown()
                self.is_dragging = True
        
        # --- 5. Execution ---
        if not self.is_scrolling:
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
        return {
            "pos": (int(smooth_x), int(smooth_y)),
            "roi": roi_rect,
            "state": "DRAG" if self.is_dragging else ("PINCH" if self.pinch_active else "MOVE"),
            "dist": dist_pinch,
            "beta": current_beta 
        }

    def _handle_scroll(self, hand_landmarks, x, y):
        # Maps vertical hand movement to mouse wheel scrolling.
        if not self.is_scrolling:
            self.is_scrolling = True
            self.scroll_y_origin = hand_landmarks.landmark[9].y
            return {"pos": (x,y), "state": "SCROLL_READY", "roi": None}
            
        curr_y = hand_landmarks.landmark[9].y
        diff = self.scroll_y_origin - curr_y 
        
        if abs(diff) > Config.SCROLL_DEADZONE:
            # Apply non-linear scaling for natural scrolling speed.
            speed = int(math.copysign(pow(abs(diff) * Config.SCROLL_SENSITIVITY, 1.5), diff) * 10)
            pyautogui.scroll(speed)
            
        return {"pos": (x, y), "state": "SCROLLING", "roi": None, "dist": 0}
