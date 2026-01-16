import pyautogui

# ==============================================================================
#                           SYSTEM CONFIGURATION
# ==============================================================================
# This file centralizes all settings for the application.
# Adjust these values to tune the hand tracking and cursor behavior.
class Config:
    # --- Camera Settings ---
    # CAM_WIDTH/HEIGHT: Resolution of the camera capture. 
    # Lower = faster processing but less detail. Higher = better precision but slower.
    # Recommended: (1280, 720) for 720p if your hardware allows.
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    
    # FPS: Targeted frames per second. High FPS makes the cursor feel "liquid".
    # Tuning: If the preview lags, lower this to 30 or 60.
    FPS = 80
    
    # --- Screen Settings ---
    # Gets your monitor's resolution so we can map camera coordinates to pixels.
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    
    # --- Smoothing & Filtering (OneEuroFilter parameters) ---
    # Filters out hand jitter for smoother movement.
    
    # ONE_EURO_BETA: How much we prioritize speed (low lag).
    # Increase if the cursor feels sluggish/behind.
    # Decrease if the cursor is shaking too much during fast movement.
    ONE_EURO_BETA = 2.0
    
    # ONE_EURO_MIN_CUTOFF: Filters jitter when your hand is still.
    # Lower this to stop the cursor from "dancing" when you aren't moving.
    # Higher if the cursor feels unresponsive to tiny movements.
    ONE_EURO_MIN_CUTOFF = 0.8 
    
    # --- ROTATION CORRECTION ---
    # Re-aligns the coordinate system to match natural arm angle.
    # Adjust this until "Straight Up" in real life is "Straight Up" on screen.
    TILT_ANGLE_DEG = -20
    
    # --- Interaction Area (Virtual Trackpad) ---
    # ROI_SCALE: Only use the center 65% of the camera for tracking.
    # Prevents having to reach to the extreme edges of the camera view.
    ROI_SCALE = 0.65
    
    # --- Click Sensitivity (Hysteresis) ---
    # PINCH_DOWN_THRESH: Distance between thumb/index to trigger a click.
    # Decrease if it's clicking too easily by accident.
    PINCH_DOWN_THRESH = 0.035
    
    # PINCH_UP_THRESH: Distance to release the click.
    # Keep slightly higher than PINCH_DOWN_THRESH to avoid rapid clicking "chatter".
    PINCH_UP_THRESH = 0.050
    
    # --- PRECISION STABILIZATION (Sniper Mode) ---
    # Slows down the cursor when fingers are close to clicking for fine-tuned control.
    PINCH_STABILIZE_DIST = 0.08 
    STABILIZE_FACTOR = 0.2  # 0.2 = cursor moves 5x slower during precision mode.
    
    # --- Scroll Settings ---
    SCROLL_SENSITIVITY = 10
    SCROLL_DEADZONE = 0.02
    
    # --- Timings ---
    DRAG_ACTIVATION_TIME = 0.3

    # --- Visual Feedback (BGR Colors) ---
    COLOR_POINTER = (0, 255, 120)  # Neon Green
    COLOR_CLICK = (0, 100, 255)    # Orange
    COLOR_SCROLL = (255, 0, 255)   # Magenta

    # --- PyAutoGUI Optimization ---
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
