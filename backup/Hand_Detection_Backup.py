import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import threading
import platform
from collections import deque

# ==============================================================================
#                           SYSTEM CONFIGURATION
# ==============================================================================
class Config:
    # Camera Settings
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    FPS = 80
    
    # Screen Settings
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    
    # Smoothing & Filtering (OneEuroFilter parameters)
    # Beta: Higher = less lag, more jitter. Lower = smoother, more lag.
    # MinCutoff: Low speed jitter filter.
    ONE_EURO_BETA = 2.0         # Increased base speed
    ONE_EURO_MIN_CUTOFF = 0.8 
    
    # --- NEW: ROTATION CORRECTION ---
    # Rotates the coordinate system to match natural hand tilt.
    # If your hand moves "up" but the cursor moves "up-right", adjust this.
    # Positive = Clockwise, Negative = Counter-Clockwise.
    TILT_ANGLE_DEG = -20  # Compensates for natural arm angle
    
    # Interaction Area (The "Virtual Trackpad")
    ROI_SCALE = 0.65  # Slightly tighter box for less arm movement
    
    # Hysteresis Thresholds (Schmitt Trigger for clicks)
    PINCH_DOWN_THRESH = 0.035  # Trigger click
    PINCH_UP_THRESH = 0.050    # Release click
    
    # --- NEW: PRECISION STABILIZATION ---
    # When fingers are this close, we dampen movement to help select small items.
    PINCH_STABILIZE_DIST = 0.08 
    STABILIZE_FACTOR = 0.2  # Multiplier for speed when stabilizing (0.2 = 5x slower)
    
    # Scroll Physics
    SCROLL_SENSITIVITY = 10
    SCROLL_DEADZONE = 0.02
    
    # Timings
    DOUBLE_CLICK_TIMEOUT = 0.25
    DRAG_ACTIVATION_TIME = 0.3

    # Visuals
    COLOR_POINTER = (0, 255, 120)  # Neon Green
    COLOR_CLICK = (0, 100, 255)    # Orange
    COLOR_SCROLL = (255, 0, 255)   # Magenta

    # PyAutoGUI Optimization
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False


# ==============================================================================
#                       1. LOW-LATENCY CAMERA THREAD
# ==============================================================================
class ThreadedCamera:
    """
    Decouples frame capture from processing logic. 
    Ensures the CV2 buffer never fills up, reducing latency to strictly 1 frame.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        # Hardware optimization for some backends
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.success, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            success, frame = self.cap.read()
            if success:
                # We only keep the latest frame
                with self.lock:
                    self.success = success
                    self.frame = frame
            else:
                self.stopped = True

    def read(self):
        with self.lock:
            return self.success, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.cap.release()


# ==============================================================================
#                       2. SIGNAL PROCESSING (SMOOTHING)
# ==============================================================================
class OneEuroFilter:
    """
    Standard Human-Computer Interaction filter. 
    Filters out high-frequency noise (jitter) while maintaining low latency 
    during high-speed movement.
    """
    def __init__(self, freq=60, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.last_time = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, custom_beta=None):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            self.x_prev = x
            self.dx_prev = 0.0
            return x
            
        dt = now - self.last_time
        if dt <= 0: return x  # Avoid division by zero
        
        # Estimate derivative (velocity)
        dx = (x - self.x_prev) / dt
        edx = self.alpha(self.dcutoff, dt) * dx + (1 - self.alpha(self.dcutoff, dt)) * self.dx_prev
        
        # Adjust cutoff frequency based on velocity
        # High velocity -> High cutoff (less lag)
        # Low velocity -> Low cutoff (less jitter)
        # Allow dynamic beta override for "Sniper Mode"
        beta = custom_beta if custom_beta is not None else self.beta
        
        cutoff = self.mincutoff + beta * abs(edx)
        alpha = self.alpha(cutoff, dt)
        
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = edx
        self.last_time = now
        return x_hat


# ==============================================================================
#                       3. GEOMETRY & MAPPING ENGINE
# ==============================================================================
class GeometryEngine:
    @staticmethod
    def get_distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    @staticmethod
    def rotate_point(x, y, degrees, cx=0.5, cy=0.5):
        """
        Rotates a point (x, y) around a center (cx, cy) by 'degrees'.
        Used to align the hand's natural movement axis with the screen.
        """
        rads = math.radians(degrees)
        cos_a = math.cos(rads)
        sin_a = math.sin(rads)
        
        # Translate to origin
        tx = x - cx
        ty = y - cy
        
        # Rotate
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        
        # Translate back
        return rx + cx, ry + cy

    @staticmethod
    def map_coordinates(norm_x, norm_y, tilt_angle=0):
        """
        Maps normalized coordinates (0-1) from the camera ROI to screen pixels.
        Includes rotation correction and aspect ratio correction.
        """
        # 0. Apply Rotation Correction (New Feature)
        rot_x, rot_y = GeometryEngine.rotate_point(norm_x, norm_y, tilt_angle)
        
        # 1. Define ROI (Region of Interest) centered in frame
        # We want an ROI that matches the screen aspect ratio (e.g., 16:9)
        screen_aspect = Config.SCREEN_WIDTH / Config.SCREEN_HEIGHT
        
        roi_w = Config.ROI_SCALE
        roi_h = roi_w * (Config.CAM_WIDTH / Config.CAM_HEIGHT) / screen_aspect
        
        # 2. Normalize input relative to ROI
        roi_x_start = (1 - roi_w) / 2
        roi_y_start = (1 - roi_h) / 2
        
        # Clamp and map
        safe_x = max(0, min(1, (rot_x - roi_x_start) / roi_w))
        safe_y = max(0, min(1, (rot_y - roi_y_start) / roi_h))
        
        # 3. Mouse Acceleration Curve (Non-linear mapping)
        # This makes small movements smaller (precision) and fast movements larger
        # Using a simple power curve: x^1.2
        # curve_x = pow(safe_x, 1.0) # Linear for now, let OneEuro handle smoothing
        # curve_y = pow(safe_y, 1.0)
        
        # 4. Scale to screen
        screen_x = safe_x * Config.SCREEN_WIDTH
        screen_y = safe_y * Config.SCREEN_HEIGHT
        
        return screen_x, screen_y, (roi_x_start, roi_y_start, roi_w, roi_h)


# ==============================================================================
#                       4. STATE MACHINE & LOGIC
# ==============================================================================
class CursorController:
    def __init__(self):
        # Smoothing filters for X and Y
        self.filter_x = OneEuroFilter(Config.FPS, Config.ONE_EURO_MIN_CUTOFF, Config.ONE_EURO_BETA)
        self.filter_y = OneEuroFilter(Config.FPS, Config.ONE_EURO_MIN_CUTOFF, Config.ONE_EURO_BETA)
        
        # State
        self.is_dragging = False
        self.is_scrolling = False
        self.pinch_start_time = 0
        self.last_click_time = 0
        
        # Hysteresis State
        self.pinch_active = False

    def process_hand(self, hand_landmarks):
        """
        Main logic pipeline:
        1. Extract Landmarks
        2. Detect Intent (Move, Scroll, Click)
        3. Apply Physics (Smoothing, Thresholds, Damping)
        4. Execute Action
        """
        # --- 1. Landmarks ---
        # Thumb(4), Index(8), Middle(12)
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        
        # "Knuckle" for stable tracking (Index PIP - 6)
        tracker_node = hand_landmarks.landmark[9] 
        
        # --- 2. Pinch Analysis (For Precision Mode) ---
        dist_pinch = GeometryEngine.get_distance(thumb, index)
        
        # --- 3. Geometry & Mapping ---
        raw_x, raw_y = tracker_node.x, tracker_node.y
        target_x, target_y, roi_rect = GeometryEngine.map_coordinates(
            raw_x, raw_y, Config.TILT_ANGLE_DEG
        )
        
        # --- 4. Precision Stabilization (Sniper Mode) ---
        # If the user is about to click (pinch distance is small), 
        # we reduce the Beta (responsiveness) to stabilize the cursor.
        
        current_beta = Config.ONE_EURO_BETA
        if dist_pinch < Config.PINCH_STABILIZE_DIST:
             # Scale beta down based on how close we are to clicking
             # Closer = slower/smoother
             factor = max(0.1, (dist_pinch / Config.PINCH_STABILIZE_DIST))
             current_beta = Config.ONE_EURO_BETA * factor * Config.STABILIZE_FACTOR

        # Apply smoothing with dynamic beta
        smooth_x = self.filter_x.filter(target_x, custom_beta=current_beta)
        smooth_y = self.filter_y.filter(target_y, custom_beta=current_beta)

        # --- 5. Gesture Detection ---
        
        # A. Scroll Gesture
        idx_up = index.y < hand_landmarks.landmark[6].y
        mid_up = middle.y < hand_landmarks.landmark[10].y
        ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
        
        # Scroll Mode: Index & Middle UP, Ring DOWN
        if idx_up and mid_up and ring_down:
            return self._handle_scroll(hand_landmarks, smooth_x, smooth_y)
            
        self.is_scrolling = False
        
        # B. Click / Drag Logic (Index + Thumb Pinch)
        
        # Schmitt Trigger (Hysteresis)
        if not self.pinch_active and dist_pinch < Config.PINCH_DOWN_THRESH:
            self.pinch_active = True
            self.pinch_start_time = time.time()
            # Don't click immediately; wait to see if it's a drag
            
        elif self.pinch_active and dist_pinch > Config.PINCH_UP_THRESH:
            self.pinch_active = False
            # Release Logic
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
            else:
                # It was a click (short duration)
                if time.time() - self.pinch_start_time < Config.DRAG_ACTIVATION_TIME:
                    pyautogui.click()
        
        # Drag Holding Logic
        if self.pinch_active:
            if not self.is_dragging and (time.time() - self.pinch_start_time > Config.DRAG_ACTIVATION_TIME):
                pyautogui.mouseDown()
                self.is_dragging = True
        
        # --- 6. Execution ---
        # Move cursor if not scrolling
        if not self.is_scrolling:
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
        return {
            "pos": (int(smooth_x), int(smooth_y)),
            "roi": roi_rect,
            "state": "DRAG" if self.is_dragging else ("PINCH" if self.pinch_active else "MOVE"),
            "dist": dist_pinch,
            "beta": current_beta # For debug
        }

    def _handle_scroll(self, hand_landmarks, x, y):
        
        if not self.is_scrolling:
            self.is_scrolling = True
            self.scroll_y_origin = hand_landmarks.landmark[9].y
            return {"pos": (x,y), "state": "SCROLL_READY", "roi": None}
            
        curr_y = hand_landmarks.landmark[9].y
        diff = self.scroll_y_origin - curr_y # Up is negative in MP, positive in scroll
        
        if abs(diff) > Config.SCROLL_DEADZONE:
            # Non-linear scrolling speed
            speed = int(math.copysign(pow(abs(diff) * Config.SCROLL_SENSITIVITY, 1.5), diff) * 10)
            pyautogui.scroll(speed)
            
        return {"pos": (x, y), "state": "SCROLLING", "roi": None, "dist": 0}


# ==============================================================================
#                       5. MAIN APPLICATION
# ==============================================================================
def main():
    # 1. Setup
    cam = ThreadedCamera().start()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1, # 1 is balanced, 0 is fast
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    controller = CursorController()
    
    print("[SYSTEM] Gesture Controller Active. Press 'Q' to Quit.")
    
    prev_time = 0
    
    while True:
        # 2. Capture
        success, frame = cam.read()
        if not success: continue
        
        # Mirror for intuition
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Process
        results = hands.process(rgb)
        
        status = {}
        
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            
            # Core Logic
            status = controller.process_hand(hand_lms)
            
            # --- Visual Debugging ---
            # Draw ROI Box
            if "roi" in status and status["roi"]:
                rx, ry, rw, rh = status["roi"]
                h, w, _ = frame.shape
                
                # Draw the tilted box? No, just draw the regular bounds for reference.
                # Since we rotate input, the visual box should ideally be rotated, 
                # but standard rect is fine for reference.
                cv2.rectangle(frame, 
                             (int(rx*w), int(ry*h)), 
                             (int((rx+rw)*w), int((ry+rh)*h)), 
                             (100, 100, 100), 2)
            
            # Draw Skeleton
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Visual Feedback for States
            h, w, _ = frame.shape
            cx, cy = int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h)
            
            color = Config.COLOR_POINTER
            if status.get("state") == "PINCH": color = (0, 255, 255)
            if status.get("state") == "DRAG": color = Config.COLOR_CLICK
            if status.get("state") == "SCROLLING": color = Config.COLOR_SCROLL
            
            cv2.circle(frame, (cx, cy), 10, color, -1)
            
            # Show "Sniper Mode" indicator
            if status.get("beta", 1.0) < Config.ONE_EURO_BETA:
                cv2.putText(frame, "PRECISION", (cx+20, cy+10), 
                           cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

        # 4. Performance Metrics
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        cv2.imshow("HyperGesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    