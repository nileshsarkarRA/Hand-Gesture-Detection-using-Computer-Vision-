import urllib.request
import os

model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
save_path = "assets/hand_landmarker.task"

if not os.path.exists("assets"):
    os.makedirs("assets")

if not os.path.exists(save_path):
    print(f"Downloading Hand Landmarker model to {save_path}...")
    urllib.request.urlretrieve(model_url, save_path)
    print("Download complete.")
else:
    print("Model already exists.")
