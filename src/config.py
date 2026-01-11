import cv2
import pyautogui

class Config:
    """Central configuration for AuraHand system.
    
    All parameters are tuned for laptop/integrated webcam scenarios.
    Adjust these values based on your hardware and preferences.
    """
    
    # --- Camera Settings ---
    WIDTH = 640
    HEIGHT = 480
    FPS = 30  # Reduced from 60 for better stability on lower-end hardware
    
    # --- Tracking & Sensitivity ---
    TRACKING_SCALE = 1.6         # Cursor speed multiplier (>1.0 = faster)
    SMOOTHING_BETA = 0.05        # Filter responsiveness (lower = smoother)
    CLICK_COOLDOWN = 0.35        # Debounce time between clicks (seconds)
    PINCH_THRESH = 0.05          # Distance threshold for pinch detection (0.0-1.0)
    ZOOM_STEP = 0.03             # Minimum hand distance change to trigger zoom
    SCROLL_SPEED = 500            # Pixels per second when scrolling
    
    # --- Calibration ---
    CALIB_TIME = 2.0              # Duration to hold hand steady (seconds)
    CALIB_DRIFT_MAX = 0.05        # Max hand drift during calibration (normalized)
    
    # --- UI Theme (BGR format) ---
    UI_BG = (15, 15, 15)          # Dark background
    UI_ACCENT = (120, 255, 0)     # Bright green for active states
    UI_WARN = (0, 100, 255)       # Orange for warnings
    UI_INFO = (255, 132, 10)      # Cyan for info
    UI_TEXT = (240, 240, 240)     # Off-white text
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- Safety & Performance ---
    pyautogui.PAUSE = 0           # No delay between mouse commands
    pyautogui.FAILSAFE = True     # Emergency stop on corner
    
    # --- Hand Detection Thresholds ---
    MEDIAPIPE_CONFIDENCE = 0.75   # Detection confidence threshold
    MAX_HANDS = 2                 # Maximum hands to detect
    
    @classmethod
    def validate(cls):
        """Validate all configuration parameters."""
        assert 0 < cls.WIDTH <= 1920, "Width must be between 0 and 1920"
        assert 0 < cls.HEIGHT <= 1080, "Height must be between 0 and 1080"
        assert 0 < cls.FPS <= 120, "FPS must be between 0 and 120"
        assert 0.5 <= cls.TRACKING_SCALE <= 3.0, "Tracking scale should be 0.5-3.0"
        assert 0 <= cls.SMOOTHING_BETA <= 0.5, "Smoothing beta should be 0-0.5"
        assert 0 < cls.CLICK_COOLDOWN <= 1.0, "Click cooldown should be 0-1.0s"
        assert 0 < cls.PINCH_THRESH <= 0.2, "Pinch threshold should be 0-0.2"
        assert 0 < cls.CALIB_TIME <= 10, "Calib time should be 0-10s"
        assert 0 < cls.MEDIAPIPE_CONFIDENCE <= 1.0, "Confidence must be 0-1.0"
