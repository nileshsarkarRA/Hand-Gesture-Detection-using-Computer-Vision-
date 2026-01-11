import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class ThreadedCamera:
    """Thread-safe camera handler with error recovery."""
    
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.success = False
        self.stopped = True
        self.lock = threading.Lock()
        self.thread = None
        
        # Initialize camera
        if not self._init_camera():
            logger.warning("Camera initialization failed, will retry on start")
    
    def _init_camera(self):
        """Initialize camera with proper error handling."""
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.src}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Read initial frame
            self.success, self.frame = self.cap.read()
            return self.success
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False

    def start(self):
        """Start the camera thread."""
        if self.cap is None or not self.cap.isOpened():
            if not self._init_camera():
                raise RuntimeError("Failed to initialize camera")
        
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera started: {self.width}x{self.height}@{self.fps}fps")
        return self

    def _update_loop(self):
        """Main camera update loop."""
        frame_count = 0
        while not self.stopped:
            try:
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self.success = True
                        self.frame = frame
                    frame_count += 1
                else:
                    logger.warning("Failed to read frame")
                    break
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                break

    def read(self):
        """Get current frame safely."""
        with self.lock:
            if self.frame is not None and self.success:
                return True, self.frame.copy()
            return False, None

    def is_running(self):
        """Check if camera is actively running."""
        return not self.stopped and self.cap is not None and self.cap.isOpened()

    def release(self):
        """Safely release camera resources."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logger.info("Camera released")
