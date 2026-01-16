import cv2
import threading
from .config import Config

class ThreadedCamera:
    """
    Handles camera capture in a separate thread to reduce latency.
    Ensures the application always has access to the most recent frame 
    without waiting for the camera's capture cycle.
    """
    def __init__(self, src=0):
        # src=0 is typically the default webcam.
        self.cap = cv2.VideoCapture(src)
        
        # Apply camera settings from the configuration module.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        # Limit buffer size to minimize visual delay.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Initial frame capture.
        self.success, self.frame = self.cap.read()
        self.stopped = False
        
        # Thread lock for safe concurrent access to the current frame.
        self.lock = threading.Lock()

    def start(self):
        # Starts the background thread for continuous frame updates.
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        # Continuously captures frames until the camera is stopped.
        while not self.stopped:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.success = success
                    self.frame = frame
            else:
                self.stopped = True

    def read(self):
        # Returns the latest successfully captured frame.
        with self.lock:
            # Return a copy to prevent the update thread from modifying 
            # the frame while it's being processed.
            return self.success, self.frame.copy() if self.frame is not None else None

    def release(self):
        # Safely stops the background thread and releases camera hardware.
        self.stopped = True
        self.cap.release()
