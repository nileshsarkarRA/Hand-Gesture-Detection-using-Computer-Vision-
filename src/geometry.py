import math
from .config import Config

class GeometryEngine:
    """
    Handles coordinate mapping, point rotation, and spatial calculations.
    Translates raw camera landmark data into screen-aware pixel coordinates.
    """
    @staticmethod
    def get_distance(p1, p2):
        # Calculates Euclidean distance between two 2D/3D landmarks.
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    @staticmethod
    def rotate_point(x, y, degrees, cx=0.5, cy=0.5):
        # Rotates a coordinate point (x, y) around a specific center point.
        # Used to compensate for webcam or arm tilt.
        rads = math.radians(degrees)
        cos_a = math.cos(rads)
        sin_a = math.sin(rads)
        
        # Translate point to origin.
        tx = x - cx
        ty = y - cy
        
        # Perform rotation using the standard 2D rotation matrix.
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        
        # Translate point back.
        return rx + cx, ry + cy

    @staticmethod
    def map_coordinates(norm_x, norm_y, tilt_angle=0):
        # Maps normalized [0.0, 1.0] coordinates to screen pixel dimensions.
        
        # 0. Apply rotation to align coordinates with the monitor.
        rot_x, rot_y = GeometryEngine.rotate_point(norm_x, norm_y, tilt_angle)
        
        # 1. Calculate the Region of Interest (ROI) dimensions.
        # Defines the "Trackpad" area within the camera's field of view.
        screen_aspect = Config.SCREEN_WIDTH / Config.SCREEN_HEIGHT
        roi_w = Config.ROI_SCALE
        roi_h = roi_w * (Config.CAM_WIDTH / Config.CAM_HEIGHT) / screen_aspect
        
        # 2. Determine the trackpad offset to keep it centered.
        roi_x_start = (1 - roi_w) / 2
        roi_y_start = (1 - roi_h) / 2
        
        # 3. Normalize coordinates within the ROI and constrain to bounds.
        safe_x = max(0, min(1, (rot_x - roi_x_start) / roi_w))
        safe_y = max(0, min(1, (rot_y - roi_y_start) / roi_h))
        
        # 4. Final conversion to absolute screen pixels.
        screen_x = safe_x * Config.SCREEN_WIDTH
        screen_y = safe_y * Config.SCREEN_HEIGHT
        
        return screen_x, screen_y, (roi_x_start, roi_y_start, roi_w, roi_h)
