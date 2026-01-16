import cv2
import mediapipe as mp

class HandDetector:
    """
    Interfaces with the MediaPipe framework to detect and track hand landmarks.
    Provides skeletal data for up to 21 key points on the hand.
    """
    def __init__(self, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Configure the MediaPipe Hands solution.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img):
        # MediaPipe requires RGB color format; convert from OpenCV's BGR.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Execute hand detection model.
        self.results = self.hands.process(img_rgb)
        
        # Return structured hand data including landmarks.
        return self.results

    def draw_landmarks(self, img, hand_lms):
        # Helper method to overlay skeletal connections on the video feed.
        self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
