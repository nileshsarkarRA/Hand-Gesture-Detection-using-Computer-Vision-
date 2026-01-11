import cv2
import numpy as np
import logging
from typing import Tuple
from src.config import Config

logger = logging.getLogger(__name__)

class HUD:
    """Heads-Up Display for real-time feedback."""
    
    @staticmethod
    def render_island(img: np.ndarray, main_text: str, sub_text: str, color: Tuple[int, int, int],
                     progress: float = 0.0, fps: float = 0, show_video: bool = True) -> None:
        """Render Dynamic Island HUD overlay.
        
        Args:
            img: Frame to render on
            main_text: Primary status text
            sub_text: Secondary status text
            color: Status indicator color (BGR)
            progress: Progress bar fill (0.0-1.0)
            fps: Current FPS for display
            show_video: Whether video feed is visible
        """
        try:
            if img is None or img.size == 0:
                return
            
            h, w = img.shape[:2]
            iw, ih = 400, 70
            ix = (w - iw) // 2
            iy = 15
            
            # Rounded Rectangle Background
            overlay = img.copy()
            cv2.rectangle(overlay, (ix+25, iy), (ix+iw-25, iy+ih), Config.UI_BG, -1)
            cv2.circle(overlay, (ix+25, iy+ih//2), ih//2, Config.UI_BG, -1)
            cv2.circle(overlay, (ix+iw-25, iy+ih//2), ih//2, Config.UI_BG, -1)
            
            alpha = 0.8 if show_video else 1.0
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
            
            # Progress Bar
            if progress > 0:
                bar_w = int((iw - 60) * min(progress, 1.0))
                cv2.line(img, (ix+30, iy+ih-2), (ix+30+bar_w, iy+ih-2), color, 3)

            # Status Dot
            cv2.circle(img, (ix+35, iy+ih//2), 6, color, -1)
            
            # Text rendering with bounds checking
            cv2.putText(img, main_text.upper()[:30], (ix+60, iy+30), Config.FONT, 0.6, Config.UI_TEXT, 2, cv2.LINE_AA)
            cv2.putText(img, sub_text[:40], (ix+60, iy+55), Config.FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            
            # FPS Counter
            cv2.putText(img, f"FPS: {int(fps)}", (ix+iw-80, iy+ih//2+5), Config.FONT, 0.4, (100, 100, 100), 1)
        except Exception as e:
            logger.error(f"HUD rendering error: {e}")

    @staticmethod
    def draw_pointer(img, pos, color):
        if pos:
            cv2.circle(img, pos, 10, color, 2)
            cv2.circle(img, pos, 4, color, -1)
