# 🖐️ HyperGesture: AI-Powered Hand Gesture Controller

[![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/AI-MediaPipe-brightgreen.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-orange.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HyperGesture** is a high-performance, low-latency computer vision application that transforms your hand into a secondary input device. Using advanced AI hand tracking and signal processing, it allows you to control your cursor, click, drag, and scroll with fluid, natural gestures through any standard webcam.

---

## 🚀 Key Features

- **💎 Liquid Management**: Implements the **One Euro Filter** to virtually eliminate hand jitter while maintaining sub-millisecond responsiveness.
- **🎯 Sniper Mode (Precision)**: Dynamically scales sensitivity based on finger proximity. The closer you get to a click, the more stable the cursor becomes for pixel-perfect accuracy.
- **🔄 Tilt Correction**: Compensates for the natural angle of the human arm (approx. -20°) so that "straight up" in real life translates to "straight up" on your screen.
- **⚡ Background Threading**: Uses a dedicated capture thread to decouple camera hardware latency from gesture logic, ensuring a smooth 60+ FPS experience.
- **🖱️ Full Mouse Emulation**: Supports moving, clicking, holding (dragging), and vertical scrolling without touching physical hardware.

---

## 🖐️ Gesture Guide

| Gesture | Action | Visual Cue | Description |
| :--- | :--- | :--- | :--- |
| **Middle Knuckle** | **Cursor Move** | Green Circle | Move your hand; the cursor follows your middle knuckle (Node 9) for maximum stability. |
| **Index Pinch** | **Left Click** | Yellow Circle | Tap your thumb and index tips together quickly. |
| **Pinch & Hold** | **Drag & Drop** | Orange Circle | Hold the pinch for >0.3s to "grab" an object. Release to drop. |
| **Two Fingers Up** | **Scroll Mode** | Magenta Circle | Raise index and middle fingers. Move hand up/down to scroll vertically. |
| **Slow Pinch** | **Precision** | "PRECISION" | Close the gap slowly to dampen movement for fine-tuned selection. |

---

## 🛠️ Technology Stack

- **Python 3.8 - 3.10** (Optimization for AI libraries)
- **MediaPipe**: Google's lightning-fast hand landmark detection.
- **OpenCV**: Real-time image processing and UI feedback.
- **PyAutoGUI**: Cross-platform OS-level mouse control.
- **One Euro Filter**: State-of-the-art signal smoothing for real-time human-computer interaction.

---

## 🔧 Installation & Setup

### 1. Prerequisites
- A functional webcam.
- **Python 3.9+** (Recommended).

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/HyperGesture.git
cd HyperGesture

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

---

## ⚙️ Customization (The Settings Hub)

All performance tuning is centralized in [`src/config.py`](src/config.py). You can adjust:

- **Sensitivity**: Modify `ONE_EURO_BETA` for a faster or smoother cursor.
- **Interaction Area**: Change `ROI_SCALE` to shrink or expand the "Trackpad" area on your camera.
- **Click Thresholds**: Tune `PINCH_DOWN_THRESH` if you have larger/smaller hands.
- **Monitor Mapping**: The system automatically detects resolution, but you can override `SCREEN_WIDTH/HEIGHT` for multi-monitor setups.

---

## 📂 Project Structure

```text
├── main.py              # Application entry point & orchestration
├── src/
│   ├── camera.py        # Multi-threaded camera capture
│   ├── config.py        # Centralized settings (Sensitivity, ROI, Tilt)
│   ├── controller.py    # Gesture state machine & mouse execution
│   ├── detector.py      # MediaPipe Hand tracking wrapper
│   ├── filters.py       # One Euro Filter implementation
│   ├── geometry.py      # Coordinate mapping & math engine
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🧠 How it Works

1. **Capture**: Frames are pulled in a high-speed background thread using OpenCV.
2. **AI Inference**: MediaPipe identifies 21 landmarks on your hand in 3D space.
3. **Geometry Mapping**: The coordinates are rotated (to fix arm tilt) and mapped from the camera's FOV to your screen's pixel resolution using a Region of Interest (ROI) logic.
4. **Filtering**: The raw coordinates pass through a One Euro Filter, which handles the trade-off between jitter and lag dynamically.
5. **State machine**: The `CursorController` monitors timings and distances to distinguish between a tap (click) and a hold (drag).

---

## ⚠️ Troubleshooting

- **Cursor is inverted?** If your camera isn't mirrored by default, check the `cv2.flip(frame, 1)` line in [main.py](main.py).
- **Hard to reach corners?** Decrease `ROI_SCALE` in [src/config.py](src/config.py). This makes the "virtual trackpad" smaller so you move your hand less.
- **Jittery movement?** Ensure you are in a well-lit environment. Low light adds noise to the AI detection.

---

## 🗺️ Roadmap
- [ ] Multi-hand support for complex gestures (zoom, rotate).
- [ ] System volume and brightness control.
- [ ] Integration of a settings GUI (PySide6).
- [ ] Customizable gesture mappings for specific apps (Gaming, Design).

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
**Nilesh Sarkar**  
*Passionate about Computer Vision and Intuitive UX.*

---
⭐ **Star this project if you found it useful!**
