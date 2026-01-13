import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import threading
import platform
# from controller import Controller  # Logic now moved into internal Controller class



# ==============================================================================
#                               CONFIG & SETTINGS
# ==============================================================================
class Config:
    # Camera
    WIDTH = 640
    HEIGHT = 480
    FPS = 60
    
    # Sensitivity
    TRACKING_SCALE = 1.5    # Acceleration factor
    CLICK_COOLDOWN = 0.4
    SCROLL_SPEED = 40
    ZOOM_STEP = 0.04
    
    # Physics
    SMOOTHING = 0.02      # Lower = Smoother, Higher = Snappier
    
    # Gestures
    PINCH_THRESH = 0.06
    FIST_THRESH = 0.4
    
    # Calibration (THE FIX IS HERE)
    CALIB_TIME = 2.0
    CALIB_DIST = 50       # Max pixel movement allowed during calibration
    
    # UI
    UI_BG = (10, 10, 10)
    UI_ACCENT = (0, 255, 120)
    UI_WARN = (0, 100, 255)
    UI_INFO = (255, 132, 10)
    UI_TEXT = (220, 220, 220)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

# ==============================================================================
#                           1. THREADED CAMERA
# ==============================================================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.success, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            success, frame = self.cap.read()
            if success:
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
#                           2. ONE EURO FILTER
# ==============================================================================
class OneEuroFilter:
    def __init__(self, freq=60, mincutoff=1.0, beta=0.05, dcutoff=1.0):
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

    def filter(self, x):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dt = now - self.last_time
        if dt <= 0: return x
        
        dx = (x - self.x_prev) / dt
        alpha_d = self.alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha_x = self.alpha(cutoff, dt)
        x_hat = alpha_x * x + (1 - alpha_x) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# ==============================================================================
#                           3.5 CONTROLLER LOGIC (FIXED)
# ==============================================================================
class Controller:
    """Self-contained Controller for hand gesture handling."""
    l_hand = None
    r_hand = None
    r_filters = [OneEuroFilter(beta=Config.SMOOTHING) for _ in range(2)]
    prev_zoom_dist = None
    last_click_time = 0
    is_dragging = False
    
    # OS Support
    OS_CMD = 'command' if platform.system() == 'Darwin' else 'ctrl'
    
    # CLICK LOCK: Stabilizes cursor by freezing it when a pinch is detected
    click_lock_start = 0
    CLICK_LOCK_DURATION = 0.35  # Slightly longer for better stabilization
    last_pinch_state = False
    
    # Window Navigation
    last_swipe_time = 0
    SWIPE_COOLDOWN = 0.8

    @classmethod
    def set_hands(cls, left, right):
        cls.l_hand = left
        cls.r_hand = right

    @classmethod
    def update_right_hand_status(cls):
        pass # Placeholder for external tracking if needed

    @classmethod
    def update_left_hand_status(cls):
        pass # Placeholder for external tracking if needed

    @classmethod
    def get_dist(cls, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    @classmethod
    def cursor_moving(cls):
        if not cls.r_hand: return
        
        # Click-Lock: Don't move cursor if we just clicked (stabilize)
        if time.time() - cls.click_lock_start < cls.CLICK_LOCK_DURATION:
            return

        # Use Middle Finger Knuckle (9) for stable tracking
        node = cls.r_hand.landmark[9]
        scr_w, scr_h = pyautogui.size()
        
        # Scaling and padding
        margin = 0.15
        x = (node.x - margin) / (1 - 2*margin)
        y = (node.y - margin) / (1 - 2*margin)
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        # Acceleration
        x = (x - 0.5) * Config.TRACKING_SCALE + 0.5
        y = (y - 0.5) * Config.TRACKING_SCALE + 0.5
        x = max(0, min(1, x))
        y = max(0, min(1, y))

        fx = cls.r_filters[0].filter(x * scr_w)
        fy = cls.r_filters[1].filter(y * scr_h)
        pyautogui.moveTo(fx, fy, duration=0)

    @classmethod
    def detect_clicking(cls):
        if not cls.r_hand: return
        now = time.time()
        if now - cls.last_click_time < Config.CLICK_COOLDOWN: return

        thumb = cls.r_hand.landmark[4]
        index = cls.r_hand.landmark[8]
        middle = cls.r_hand.landmark[12]
        pinky = cls.r_hand.landmark[20]

        # Left Click (Index + Thumb pinch)
        if cls.get_dist(thumb, index) < Config.PINCH_THRESH:
            if not cls.last_pinch_state:
                pyautogui.click()
                cls.last_click_time = now
                cls.click_lock_start = now # Start Click-Lock
            cls.last_pinch_state = True
        # Right Click (Middle + Thumb pinch)
        elif cls.get_dist(thumb, middle) < Config.PINCH_THRESH:
            if not cls.last_pinch_state:
                pyautogui.rightClick()
                cls.last_click_time = now
                cls.click_lock_start = now # Start Click-Lock
            cls.last_pinch_state = True
        else:
            cls.last_pinch_state = False

        # Double Click Protection (Optional - user mentioned thumb displacement)
        # We handle this via the pinch logic above.
        fingers_up = 0
        for i in [8, 12, 16, 20]:
            if cls.r_hand.landmark[i].y < cls.r_hand.landmark[i-2].y:
                fingers_up += 1
        
        if fingers_up == 0 and not cls.is_dragging:
            pyautogui.mouseDown()
            cls.is_dragging = True
        elif fingers_up > 1 and cls.is_dragging:
            pyautogui.mouseUp()
            cls.is_dragging = False

    @classmethod
    def detect_zooming(cls):
        if not (cls.l_hand and cls.r_hand):
            cls.prev_zoom_dist = None
            return
        
        dist = cls.get_dist(cls.l_hand.landmark[9], cls.r_hand.landmark[9])
        if cls.prev_zoom_dist is None:
            cls.prev_zoom_dist = dist
            return

        delta = dist - cls.prev_zoom_dist
        if abs(delta) > Config.ZOOM_STEP:
            if delta > 0: pyautogui.hotkey(cls.OS_CMD, '=')
            else: pyautogui.hotkey(cls.OS_CMD, '-')
            cls.prev_zoom_dist = dist

    @classmethod
    def detect_scrolling(cls):
        if not (cls.l_hand and cls.r_hand): return
        # Left hand fist = Scroll Mode
        l_fingers_up = sum([1 for i in [8,12,16,20] if cls.l_hand.landmark[i].y < cls.l_hand.landmark[i-2].y])
        if l_fingers_up == 0:
            # Use right hand vertical movement to scroll
            node = cls.r_hand.landmark[9]
            if hasattr(cls, 'prev_sc_y') and cls.prev_sc_y is not None:
                dy = cls.prev_sc_y - node.y
                if abs(dy) > 0.01:
                    pyautogui.scroll(int(dy * 1000))
            cls.prev_sc_y = node.y
        else:
            cls.prev_sc_y = None

    @classmethod
    def detect_swipe_gesture(cls):
        """Window Navigation (App Switcher) using Left Hand swipe."""
        if not cls.l_hand: return
        now = time.time()
        if now - cls.last_swipe_time < cls.SWIPE_COOLDOWN: return

        # Use index finger tip velocity for swipe
        tip = cls.l_hand.landmark[8]
        if hasattr(cls, 'prev_l_tip'):
            dx = tip.x - cls.prev_l_tip.x
            # Fast horizontal move detected
            if abs(dx) > 0.08:
                if dx > 0: # Swipe Right
                    pyautogui.hotkey(cls.OS_CMD, 'tab')
                else: # Swipe Left
                    pyautogui.hotkey(cls.OS_CMD, 'shift', 'tab')
                cls.last_swipe_time = now
        cls.prev_l_tip = tip
class DynamicIsland:
    @staticmethod
    def render(img, main, sub, color, progress=0.0, fps=0, show_video=True):
        h, w = img.shape[:2]
        iw, ih = 420, 80
        ix = (w - iw) // 2
        iy = 20
        
        overlay = img.copy()
        cv2.rectangle(overlay, (ix+20, iy), (ix+iw-20, iy+ih), Config.UI_BG, -1)
        cv2.circle(overlay, (ix+20, iy+ih//2), ih//2, Config.UI_BG, -1)
        cv2.circle(overlay, (ix+iw-20, iy+ih//2), ih//2, Config.UI_BG, -1)
        
        alpha = 0.85 if show_video else 1.0
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        
        if progress > 0:
            pw = int((iw - 60) * progress)
            cv2.line(img, (ix+30, iy+ih-5), (ix+30+pw, iy+ih-5), color, 4)

        cv2.circle(img, (ix+40, iy+ih//2), 8, color, -1)
        cv2.putText(img, main, (ix+70, iy+35), Config.FONT, 0.7, Config.UI_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, sub, (ix+70, iy+65), Config.FONT, 0.5, (160,160,160), 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(fps)}", (ix+iw-40, iy+45), Config.FONT, 0.6, (80,80,80), 1)

# ==============================================================================
#                           4. GESTURE LOGIC
# ==============================================================================
class GestureEngine:
    def __init__(self):
        self.scr_w, self.scr_h = pyautogui.size()
        self.filter_x = OneEuroFilter(mincutoff=0.01, beta=Config.SMOOTHING)
        self.filter_y = OneEuroFilter(mincutoff=0.01, beta=Config.SMOOTHING)
        
        # State
        self.state = "CALIBRATION" 
        self.tutorial_step = 0
        
        # Hand Data
        self.r_fingers = [0]*5
        self.r_pinches = [0]*5
        self.r_fist = False
        # Left hand tracking
        self.l_fingers = [0]*5
        self.l_pinches = [0]*5
        self.l_fist = False
        
        # Logic vars
        self.last_click = 0
        self.is_dragging = False
        
        # Zoom vars
        self.prev_zoom_dist = None
        
        # Calibration vars
        self.calib_start = 0
        self.calib_ref = None
        # Scroll helper
        self.prev_scroll_y = None
        # Use controller's gesture handlers for navigation (prefer controller)
        self.use_controller = True

    def update_fingers(self, lm, label):
        """Analyzes hand landmarks."""
        if label == "Right":
            thumb = (lm[4].x, lm[4].y)
            
            # Thumb
            self.r_fingers[0] = 1 if lm[4].x < lm[3].x else 0
            
            # Fingers (Tip vs PIP)
            for i, tip_id in enumerate([8,12,16,20]):
                pip_id = tip_id - 2 
                self.r_fingers[i+1] = 1 if lm[tip_id].y < lm[pip_id].y else 0
                
            # Pinches
            for i, tip_id in enumerate([8,12,16,20]):
                tip = (lm[tip_id].x, lm[tip_id].y)
                dist = math.hypot(tip[0]-thumb[0], tip[1]-thumb[1])
                self.r_pinches[i+1] = 1 if dist < Config.PINCH_THRESH else 0
            
            # Fist Detection (Index, Middle, Ring, Pinky all down)
            fingers_down = sum([1 for f in self.r_fingers[1:] if f == 0]) 
            self.r_fist = (fingers_down >= 3)
        elif label == "Left":
            thumb = (lm[4].x, lm[4].y)

            # Thumb (left hand mirrored)
            self.l_fingers[0] = 1 if lm[4].x > lm[3].x else 0

            # Fingers (Tip vs PIP)
            for i, tip_id in enumerate([8,12,16,20]):
                pip_id = tip_id - 2 
                self.l_fingers[i+1] = 1 if lm[tip_id].y < lm[pip_id].y else 0

            # Pinches
            for i, tip_id in enumerate([8,12,16,20]):
                tip = (lm[tip_id].x, lm[tip_id].y)
                dist = math.hypot(tip[0]-thumb[0], tip[1]-thumb[1])
                self.l_pinches[i+1] = 1 if dist < Config.PINCH_THRESH else 0

            # Fist Detection (Index, Middle, Ring, Pinky all down)
            fingers_down = sum([1 for f in self.l_fingers[1:] if f == 0]) 
            self.l_fist = (fingers_down >= 3)

    def process_cursor(self, right_lm):
        """MOVES CURSOR USING KNUCKLE TRACKING (STABLE)."""
        tracker_node = right_lm[9]  # Middle Finger Knuckle

        # Map Coordinates with Scaling tuned for laptops/flat screens
        margin = 0.12
        raw_x = (tracker_node.x - margin) / (1 - 2 * margin)
        raw_y = (tracker_node.y - margin) / (1 - 2 * margin)

        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))

        # Apply acceleration center-scaling so small hand moves near center
        raw_x = (raw_x - 0.5) * Config.TRACKING_SCALE + 0.5
        raw_y = (raw_y - 0.5) * Config.TRACKING_SCALE + 0.5

        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))

        sx = self.filter_x.filter(raw_x * self.scr_w)
        sy = self.filter_y.filter(raw_y * self.scr_h)

        pyautogui.moveTo(sx, sy, duration=0)

    def process_zoom(self, r_lm, l_lm):
        """Two Handed Zoom."""
        dist = math.hypot(r_lm[9].x - l_lm[9].x, r_lm[9].y - l_lm[9].y)
        
        if self.prev_zoom_dist is None:
            self.prev_zoom_dist = dist
            return None
        
        delta = dist - self.prev_zoom_dist
        
        if abs(delta) > Config.ZOOM_STEP:
            if delta > 0:
                pyautogui.hotkey('ctrl', '=')
                self.prev_zoom_dist = dist
                return "ZOOM IN"
            else:
                pyautogui.hotkey('ctrl', '-')
                self.prev_zoom_dist = dist
                return "ZOOM OUT"
        return None

    def run_calibration(self, right_lm):
        if not right_lm:
            self.calib_start = 0
            return "Searching...", "Show Right Hand", Config.UI_WARN, 0.0
        
        # Use Knuckle for calibration too
        curr = (right_lm[9].x, right_lm[9].y)
        if self.calib_start == 0:
            self.calib_start = time.time()
            self.calib_ref = curr
            return "Calibrating...", "Hold Hand Steady", Config.UI_INFO, 0.1
            
        # FIX: Ensure Config.CALIB_DIST is defined
        dist = math.hypot(curr[0]-self.calib_ref[0], curr[1]-self.calib_ref[1]) * Config.WIDTH
        if dist > Config.CALIB_DIST:
            self.calib_start = time.time()
            self.calib_ref = curr
            return "Moving too much!", "Keep Hand Still", Config.UI_WARN, 0.0
            
        elapsed = time.time() - self.calib_start
        prog = min(elapsed / Config.CALIB_TIME, 1.0)
        
        if elapsed > Config.CALIB_TIME:
            self.state = "TUTORIAL"
            return "Success!", "Starting Tutorial...", Config.UI_ACCENT, 1.0
            
        return "Calibrating...", f"Hold Steady {(Config.CALIB_TIME-elapsed):.1f}s", Config.UI_INFO, prog

    def run_tutorial(self, right_lm, left_lm):
        # Extended multi-step tutorial for laptop gestures
        # Steps: 0 Move, 1 Pinch click, 2 Double-click both pinches, 3 Left-fist scroll, 4 Two-hand zoom, 5 Fist-drag
        if not hasattr(self, 'tutorial_step'):
            self.tutorial_step = 0

        # Step 0: Move cursor
        if self.tutorial_step == 0:
            if right_lm:
                self.process_cursor(right_lm)
                if not hasattr(self, 'tut_move_timer'): self.tut_move_timer = time.time()
                if time.time() - self.tut_move_timer > 2.0:
                    self.tutorial_step = 1
            else:
                self.tut_move_timer = time.time()
            return "Tutorial 1/6", "Move hand to control cursor", Config.UI_INFO, 1/6

        # Step 1: Pinch index to click
        if self.tutorial_step == 1:
            status = "Waiting"
            if right_lm and self.r_pinches[1]:
                status = "Pinch Detected"
                if not hasattr(self, 'tut_pinchtimer'): self.tut_pinchtimer = time.time()
                if time.time() - self.tut_pinchtimer > 0.5:
                    self.tutorial_step = 2
            else:
                self.tut_pinchtimer = time.time()
            return "Tutorial 2/6", f"Pinch Index (Right) to Click ({status})", Config.UI_WARN, 2/6

        # Step 2: Both index pinches -> double click
        if self.tutorial_step == 2:
            status = "Waiting"
            if right_lm and left_lm and self.r_pinches[1] and self.l_pinches[1]:
                status = "Both Pinch"
                if not hasattr(self, 'tut_bothpinch_timer'): self.tut_bothpinch_timer = time.time()
                if time.time() - self.tut_bothpinch_timer > 0.5:
                    self.tutorial_step = 3
            else:
                self.tut_bothpinch_timer = time.time()
            return "Tutorial 3/6", f"Pinch Both Indexes to Double-Click ({status})", Config.UI_ACCENT, 3/6

        # Step 3: Left-fist scroll mode (move right knuckle up/down)
        if self.tutorial_step == 3:
            status = "Waiting"
            if left_lm and self.l_fist and right_lm:
                status = "Scroll Mode"
                if not hasattr(self, 'tut_scroll_timer'): self.tut_scroll_timer = time.time()
                if time.time() - self.tut_scroll_timer > 1.0:
                    self.tutorial_step = 4
            else:
                self.tut_scroll_timer = time.time()
            return "Tutorial 4/6", f"Make Left Fist + Move Right Knuckle to Scroll ({status})", Config.UI_INFO, 4/6

        # Step 4: Two-hand zoom
        if self.tutorial_step == 4:
            status = "Waiting"
            if right_lm and left_lm:
                action = self.process_zoom(right_lm, left_lm)
                if action:
                    status = action
                    if not hasattr(self, 'tut_zoom_timer'): self.tut_zoom_timer = time.time()
                    if time.time() - self.tut_zoom_timer > 0.6:
                        self.tutorial_step = 5
                else:
                    self.tut_zoom_timer = time.time()
            return "Tutorial 5/6", f"Use Both Hands Knuckles to Zoom ({status})", Config.UI_INFO, 5/6

        # Step 5: Fist drag to move windows
        if self.tutorial_step == 5:
            status = "Waiting"
            if right_lm and self.r_fist:
                status = "Fist Detected"
                if not hasattr(self, 'tut_drag_timer'): self.tut_drag_timer = time.time()
                if time.time() - self.tut_drag_timer > 1.0:
                    self.state = "ACTIVE"
                    return "Tutorial Complete", "Entering Active Mode", Config.UI_ACCENT, 1.0
            else:
                self.tut_drag_timer = time.time()
            return "Tutorial 6/6", f"Make Fist (Right) and Move to Drag ({status})", Config.UI_WARN, 6/6

        return "Tutorial", "Follow Instructions", Config.UI_INFO, 0.0

    def run_active(self, right_lm, left_lm):
        now = time.time()
        main_txt = "Active"
        sub_txt = "Tracking Knuckle"
        col = Config.UI_ACCENT

        # If using controller module, skip internal pyautogui actions
        if getattr(self, 'use_controller', False):
            return "Active (Controller)", "Using Controller Gestures", Config.UI_ACCENT, 0.0
        
        # 0. ZOOM (Priority)
        if right_lm and left_lm:
            zoom_action = self.process_zoom(right_lm, left_lm)
            if zoom_action:
                return "ZOOMING", zoom_action, Config.UI_INFO, 0.0
        else:
            self.prev_zoom_dist = None

        # Double-click (both index pinched) - high priority to avoid conflicts
        if right_lm and left_lm and self.r_pinches[1] and self.l_pinches[1]:
            if now - self.last_click > Config.CLICK_COOLDOWN:
                pyautogui.doubleClick()
                self.last_click = now
                return "DOUBLE CLICK", "Both Index Pinch", Config.UI_ACCENT, 0.0

        # Scroll mode: left hand makes a fist -> use right knuckle vertical movement to scroll
        if left_lm and self.l_fist and right_lm:
            ry = right_lm[9].y
            if self.prev_scroll_y is None:
                self.prev_scroll_y = ry
            else:
                dy = self.prev_scroll_y - ry
                if abs(dy) > 0.005:
                    amount = int(dy * Config.SCROLL_SPEED * 10)
                    if amount != 0:
                        pyautogui.scroll(amount)
            self.prev_scroll_y = ry
            return "SCROLL MODE", "Left Fist + Move", Config.UI_INFO, 0.0
        else:
            self.prev_scroll_y = None

        # 1. RIGHT HAND
        if right_lm:
            # DRAG (Fist)
            if self.r_fist:
                if not self.is_dragging:
                    try:
                        pyautogui.mouseDown(button='left')
                    except TypeError:
                        pyautogui.mouseDown()
                    self.is_dragging = True
                # Move cursor while holding mouse down for reliable dragging
                self.process_cursor(right_lm)
                main_txt = "DRAGGING"
                sub_txt = "Fist Locked"
                col = Config.UI_WARN
            else:
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp(button='left')
                    except TypeError:
                        pyautogui.mouseUp()
                    self.is_dragging = False
                
                # Move Cursor
                self.process_cursor(right_lm)

                # CLICKS
                if now - self.last_click > Config.CLICK_COOLDOWN:
                    if self.r_pinches[1]:
                        pyautogui.click()
                        self.last_click = now
                        main_txt = "CLICK"
                        sub_txt = "Left Click"
                    elif self.r_pinches[2]:
                        pyautogui.rightClick()
                        self.last_click = now
                        main_txt = "R-CLICK"
                        sub_txt = "Right Click"
                    elif self.r_pinches[4]:
                         pyautogui.scroll(-Config.SCROLL_SPEED)
                         main_txt = "SCROLL"
                         sub_txt = "Down"

        return main_txt, sub_txt, col, 0.0

# ==============================================================================
#                               MAIN LOOP
# ==============================================================================
def main():
    # Support for external webcams: Try index 0 then 1
    cam_index = 0
    cap_test = cv2.VideoCapture(cam_index)
    if not cap_test.isOpened():
        cam_index = 1
    cap_test.release()

    hands = mp.solutions.hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    cam = ThreadedCamera(src=cam_index).start()
    engine = GestureEngine()
    
    show_video = False
    pTime = None
    
    print("Started. Video Hidden. Press 'V' to Toggle Camera.")

    while True:
        success, raw = cam.read()
        if not success: continue

        # GPU Flip
        try:
            u_frame = cv2.UMat(raw)
            u_frame = cv2.flip(u_frame, 1)
            disp = u_frame.get()
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        except:
            disp = cv2.flip(raw, 1)
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        # AI
        results = hands.process(rgb)
        
        right_lm = None
        left_lm = None
        
        if results.multi_hand_landmarks:
            # Keep both the Mediapipe landmark objects (for Controller) and lists (for GestureEngine)
            right_obj = None
            left_obj = None
            for lm, h in zip(results.multi_hand_landmarks, results.multi_handedness):
                lbl = h.classification[0].label
                engine.update_fingers(lm.landmark, lbl)
                if lbl == "Right":
                    right_lm = lm.landmark
                    right_obj = lm
                else:
                    left_lm = lm.landmark
                    left_obj = lm
        else:
            right_obj = None
            left_obj = None

        # Feed Controller with the Mediapipe hand objects
        Controller.set_hands(left_obj, right_obj)
        Controller.update_right_hand_status()
        Controller.update_left_hand_status()

        # STATE
        if engine.state == "CALIBRATION":
            t1, t2, col, prog = engine.run_calibration(right_lm)
        elif engine.state == "TUTORIAL":
            t1, t2, col, prog = engine.run_tutorial(right_lm, left_lm)
        elif engine.state == "ACTIVE":
            # If controller integration is enabled, use Controller handlers
            if getattr(engine, 'use_controller', False):
                # Controller reads Mediapipe objects via set_hands above
                Controller.cursor_moving()
                Controller.detect_clicking()
                Controller.detect_zooming()
                Controller.detect_scrolling()
                Controller.detect_swipe_gesture()
                t1, t2, col, prog = "Active (Controller)", "Controller Mode", Config.UI_ACCENT, 0.0
            else:
                t1, t2, col, prog = engine.run_active(right_lm, left_lm)

        # RENDER
        if not show_video:
            disp = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)
        
        cTime = time.time()
        if pTime is None:
            fps = 0.0
        else:
            dt = cTime - pTime
            fps = 1.0 / dt if dt > 0 else 0.0
        pTime = cTime
        
        DynamicIsland.render(disp, t1, t2, col, prog, fps, show_video)
        
        if show_video and results.multi_hand_landmarks:
             for lm in results.multi_hand_landmarks:
                 mp.solutions.drawing_utils.draw_landmarks(disp, lm, mp.solutions.hands.HAND_CONNECTIONS)
                 # Visualize the Tracker Point (Knuckle)
                 if right_lm:
                     cx, cy = int(right_lm[9].x * Config.WIDTH), int(right_lm[9].y * Config.HEIGHT)
                     cv2.circle(disp, (cx, cy), 8, Config.UI_ACCENT, -1)

        cv2.imshow("Gesture OS", disp)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('v'): show_video = not show_video

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


