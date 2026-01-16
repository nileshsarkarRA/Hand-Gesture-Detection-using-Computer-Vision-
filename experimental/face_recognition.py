import cv2
import face_recognition
import numpy as np
import threading
import pickle
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ==============================================================================
#                               SYSTEM CONFIG
# ==============================================================================
class Config:
    # Camera
    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    FPS = 30
    
    # Model Physics
    # The max Euclidean distance to consider a match.
    # Lower = Stricter (less false positives, more false negatives).
    # Higher = Looser. 0.6 is the industry standard heuristic for dlib.
    TOLERANCE = 0.55 
    
    # Performance
    # Running Neural Net inference on every frame is slow. 
    # We process every Nth frame (skip frames) to keep UI smooth.
    PROCESS_EVERY_N_FRAMES = 3 
    
    # Model Scale
    # 'hog' is faster/lighter (CPU). 'cnn' is more accurate but requires CUDA/GPU.
    MODEL_TYPE = "hog" 
    
    # Database
    DB_PATH = "face_db.pkl"
    KNOWN_FACES_DIR = "known_faces"  # Put images of people here (e.g., "nilesh.jpg")

    # Visuals
    COLOR_MATCH = (0, 255, 120)    # Neon Green
    COLOR_UNKNOWN = (0, 0, 255)    # Red
    FONT = cv2.FONT_HERSHEY_DUPLEX

# ==============================================================================
#                       1. LOW-LATENCY CAMERA (REUSED)
# ==============================================================================
class ThreadedCamera:
    """
    Standard threaded camera to prevent I/O blocking.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
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
#                       2. THE VECTOR DATABASE
# ==============================================================================
@dataclass
class Identity:
    name: str
    vector: np.ndarray  # The 128-dimensional embedding
    source_img: str

class FaceVectorDB:
    """
    Manages the 'Manifold'. Loads images, computes 128-d vectors, 
    and stores them in a serialized database (Pickle) for speed.
    """
    def __init__(self):
        self.identities: List[Identity] = []
        self._load_or_build_db()

    def _load_or_build_db(self):
        # 1. Check if DB exists
        if os.path.exists(Config.DB_PATH):
            print(f"[DB] Loading existing face database from {Config.DB_PATH}...")
            with open(Config.DB_PATH, 'rb') as f:
                self.identities = pickle.load(f)
            print(f"[DB] Loaded {len(self.identities)} identities.")
        else:
            print("[DB] No database found. Building from 'known_faces' directory...")
            self._build_from_directory()

    def _build_from_directory(self):
        # 2. Scan directory
        path = Path(Config.KNOWN_FACES_DIR)
        if not path.exists():
            path.mkdir()
            print(f"[WARNING] Created empty directory '{Config.KNOWN_FACES_DIR}'. Put images there!")
            return

        image_files = [f for f in path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        for img_path in image_files:
            print(f"[ENCODING] Processing {img_path.name}...")
            
            # Load and convert to RGB (dlib expects RGB)
            image = face_recognition.load_image_file(str(img_path))
            
            # Detect faces first
            face_locations = face_recognition.face_locations(image, model=Config.MODEL_TYPE)
            
            if not face_locations:
                print(f"  -> SKIPPING: No face detected in {img_path.name}")
                continue

            # Compute the 128-D embedding
            # We take the first face found [0]
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Name is filename without extension (e.g., "nilesh.jpg" -> "nilesh")
            name = img_path.stem.title()
            
            self.identities.append(Identity(name=name, vector=encoding, source_img=img_path.name))

        # 3. Save to disk
        with open(Config.DB_PATH, 'wb') as f:
            pickle.dump(self.identities, f)
        print(f"[DB] Database built and saved with {len(self.identities)} faces.")

    def get_vectors(self) -> List[np.ndarray]:
        return [id.vector for id in self.identities]

    def get_names(self) -> List[str]:
        return [id.name for id in self.identities]

# ==============================================================================
#                       3. THE RECOGNITION ENGINE (MATH)
# ==============================================================================
class RecognitionEngine:
    """
    Handles the Linear Algebra of comparing the live face vector 
    against the database vectors.
    """
    def __init__(self, db: FaceVectorDB):
        self.db = db
        self.known_vectors = self.db.get_vectors()
        self.known_names = self.db.get_names()

    def recognize(self, unknown_encoding: np.ndarray) -> str:
        """
        Calculates L2 Euclidean distance between the unknown vector (u)
        and all known vectors (v_i).
        
        d(u, v) = || u - v ||2 = sqrt( sum( (u_i - v_i)^2 ) )
        """
        if not self.known_vectors:
            return "Unknown"

        # 1. Vectorized Distance Calculation (Numpy is optimized for this)
        # axis=1 means we compute norm across the 128 dimensions for each known face
        distances = np.linalg.norm(self.known_vectors - unknown_encoding, axis=1)

        # 2. Find the nearest neighbor (ArgMin)
        min_dist_index = np.argmin(distances)
        min_dist = distances[min_dist_index]

        # 3. Apply Threshold (The 'Hypersphere' radius)
        # If the distance is too large, the point is outside the known cluster.
        if min_dist <= Config.TOLERANCE:
            return self.known_names[min_dist_index] # Match found
        else:
            return "Unknown"

# ==============================================================================
#                       4. MAIN APPLICATION LOOP
# ==============================================================================
def main():
    # Setup
    if not os.path.exists(Config.KNOWN_FACES_DIR):
        os.makedirs(Config.KNOWN_FACES_DIR)
        print(f"PLEASE ADD IMAGES TO THE '{Config.KNOWN_FACES_DIR}' FOLDER AND RESTART.")
        
    db = FaceVectorDB()
    engine = RecognitionEngine(db)
    cam = ThreadedCamera().start()

    print("[SYSTEM] Face Recognition Active. Press 'Q' to Quit.")

    frame_count = 0
    
    # State variables for frame skipping
    face_locations = []
    face_names = []

    while True:
        success, frame = cam.read()
        if not success: continue

        # Optimization: Resize frame for faster processing (1/4th size)
        # Detection is expensive; doing it on small images is standard practice.
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # --- HEAVY LIFTING (Every N frames) ---
        if frame_count % Config.PROCESS_EVERY_N_FRAMES == 0:
            
            # 1. Detect Faces (Geometry)
            face_locations = face_recognition.face_locations(rgb_small_frame, model=Config.MODEL_TYPE)
            
            # 2. Encode Faces (128-D Projection)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for encoding in face_encodings:
                # 3. Match Vectors (Algebra)
                name = engine.recognize(encoding)
                face_names.append(name)

        frame_count += 1

        # --- VISUALIZATION ---
        # Loop through results and draw boxes
        # Note: We must scale coordinates back up (x4) because we resized earlier
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Color logic
            color = Config.COLOR_MATCH if name != "Unknown" else Config.COLOR_UNKNOWN

            # Draw Box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw Label Background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw Text
            cv2.putText(frame, name, (left + 6, bottom - 6), Config.FONT, 0.8, (255, 255, 255), 1)

        # Display
        cv2.imshow('Vector Face ID', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()