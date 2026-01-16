import cv2
import time
from src.config import Config
from src.camera import ThreadedCamera
from src.controller import CursorController
from src.detector import HandDetector

# ==============================================================================
#                       MAIN ORCHESTRATOR
# ==============================================================================
def main():
    # --- 1. Initialization ---
    # Start the multi-threaded camera module.
    cam = ThreadedCamera().start()
    
    # Initialize the hand tracking detector.
    detector = HandDetector(
        max_num_hands=1,
        model_complexity=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize the gesture-to-cursor controller.
    controller = CursorController()
    
    print("[SYSTEM] Gesture Controller Active. Press 'Q' in the preview window to Quit.")
    
    prev_time = 0
    
    # --- 2. Main Processing Loop ---
    while True:
        # A. Capture current frame from the camera thread.
        success, frame = cam.read()
        if not success: continue
        
        # B. Flip frame horizontally to provide a natural "mirror" feel.
        frame = cv2.flip(frame, 1)
        
        # C. Process frame to extract hand skeleton data.
        results = detector.find_hands(frame)
        
        status = {}
        
        # D. Execute gesture logic if a hand is detected in frame.
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            
            # Map hand movement to cursor actions.
            status = controller.process_hand(hand_lms)
            
            # --- E. Visual Feedback Overlay ---
            # Draw the interaction trackpad bounds (ROI).
            if "roi" in status and status["roi"]:
                rx, ry, rw, rh = status["roi"]
                h, w, _ = frame.shape
                cv2.rectangle(frame, 
                             (int(rx*w), int(ry*h)), 
                             (int((rx+rw)*w), int((ry+rh)*h)), 
                             (100, 100, 100), 2)
            
            # Render hand landmarks and skeletal connections.
            detector.draw_landmarks(frame, hand_lms)
            
            # Draw a status indicator circle at the index finger tip.
            h, w, _ = frame.shape
            cx, cy = int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h)
            
            color = Config.COLOR_POINTER
            state = status.get("state")
            if state == "PINCH": color = (0, 255, 255) # Feedback for active pinch.
            elif state == "DRAG": color = Config.COLOR_CLICK # Feedback for drag state.
            elif state == "SCROLLING": color = Config.COLOR_SCROLL # Feedback for scroll state.
            
            cv2.circle(frame, (cx, cy), 10, color, -1)
            
            # Display "PRECISION" label when movement damping (Sniper Mode) is active.
            if status.get("beta", 1.0) < Config.ONE_EURO_BETA:
                cv2.putText(frame, "PRECISION", (cx+20, cy+10), 
                           cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

        # --- F. Performance Analytics ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # --- G. UI Management ---
        cv2.imshow("HyperGesture", frame)
        
        # Graceful exit on 'q' key press.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources.
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    