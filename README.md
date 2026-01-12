<div align="center">

# 🌊 GestureFlow

### *Natural Hand Gestures, Effortless Control*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6.svg)](https://www.microsoft.com/windows)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform your webcam into a powerful gesture interface. Control your Windows computer with natural hand movements using cutting-edge computer vision.

**Optimized for Windows • Best Implementation with Existing Tech • Ready for Production**

[Features](#-features) •
[Installation](#-quick-start) •
[Gestures](#-gesture-library) •
[Usage](#-usage) •
[Future Work](#-future-work)

</div>

---

> [!NOTE]
> **Platform Focus:** This implementation is optimized and tested for **Windows 10/11**. While the code is cross-platform compatible, Windows is the primary supported platform for this version.

> [!IMPORTANT]
> **Future Development:** A new repository will be created for a **custom ML/CNN/DL model** implementation optimized for Raspberry Pi 5 and other platforms. This current codebase represents the **best implementation using existing MediaPipe technology**.

---

## 🌟 Features

GestureFlow brings **spatial computing** to your Windows PC:

- 🚀 **Zero-Latency Response** — Multi-threaded architecture ensures real-time tracking at maximum camera FPS
- 🎯 **Optimized Detection** — Relaxed thresholds for reliable gesture recognition (tested on Windows)
- 🧘 **Natural Ergonomics** — Comfortable hand positions, no awkward poses required
- 🎨 **Dynamic Island HUD** — Beautiful, minimal overlay inspired by modern UI design
- ⚡ **GPU Accelerated** — Leverages MediaPipe's hardware acceleration on Windows
- 🔒 **Privacy First** — All processing happens locally on your device
- ⏱️ **Fast Calibration** — Smart 2.5-second calibration with lighting quality checks
- 👆 **Responsive Clicks** — 80ms hold time for natural interaction

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (Python 3.10+ also supported)
- **Webcam** (built-in laptop camera or external USB camera)
- **Operating System:** macOS, Windows 10/11, or Linux (Ubuntu 20.04+)

### Installation

#### Option 1: Using pyenv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/nileshsarkarRA/Hand-Gesture-Detection-using-Computer-Vision-.git
cd Hand-Gesture-Detection-using-Computer-Vision-

# 2. Set Python version (using pyenv)
pyenv local 3.11.14

# 3. Create virtual environment (optional but recommended)
python -m venv venv

# 4. Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download the MediaPipe hand tracking model
python download_model.py
```

#### Option 2: Using System Python

```bash
# 1. Clone the repository
git clone https://github.com/nileshsarkarRA/Hand-Gesture-Detection-using-Computer-Vision-.git
cd Hand-Gesture-Detection-using-Computer-Vision-

# 2. Verify Python version (must be 3.10 or higher)
python --version

# 3. Create virtual environment
python -m venv venv

# 4. Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download the MediaPipe hand tracking model
python download_model.py
```

### Run GestureFlow

```bash
# Make sure virtual environment is activated
python src/main.py
```

**Keyboard Controls:**
- Press **`V`** to toggle video feed visualization
- Press **`Q`** to quit the application
- Move mouse to screen corner for emergency stop

---

## 🎮 Gesture Library

Master these intuitive gestures to control your computer:

| 🎯 Action | ✋ Gesture | 📝 Description |
|:----------|:----------|:---------------|
| **Move Cursor** | Right hand movement | Smooth cursor tracking via knuckle detection |
| **Left Click** | ✨ **Left hand:** Index + Thumb pinch | Hold for 0.15s - Standard click action |
| **Right Click** | ✨ **Right hand:** Index + Thumb pinch | Hold for 0.15s - Context menu activation |
| **Drag & Drop** | Right hand: Closed fist → Move → Open | Lock cursor, move, then release |
| **Double Click** | Both hands: Index + Thumb pinch | Quick double-tap action |
| **Scroll** | Left fist + Right hand vertical | Natural scrolling motion |

> **Pro Tip:** Clicks require holding the pinch gesture for 0.15 seconds to prevent accidental clicks from finger movement. Watch the progress indicator!

---

## 📖 Usage

### Basic Workflow

1. **Launch** the application: `python src/main.py`
2. **Position** your hand(s) in front of the webcam (about 30-60cm away)
3. **Perform** gestures from the library above
4. **Toggle** video feed with `V` to see hand tracking visualization
5. **Exit** anytime with `Q` key

### Display Modes

- **HUD Mode** (default): Clean grid interface with gesture status overlay
- **Video Mode** (`V` key): Live camera feed with hand landmark skeleton overlay

### Configuration

Customize settings in [`src/config.py`](src/config.py):

```python
# Camera settings
WIDTH = 640           # Camera resolution width
HEIGHT = 480          # Camera resolution height
FPS = 30             # Target frames per second

# Tracking parameters
MAX_HANDS = 2        # Maximum number of hands to track
MEDIAPIPE_CONFIDENCE = 0.7  # Detection confidence threshold

# Gesture sensitivity
PINCH_THRESHOLD = 0.05      # Pinch detection sensitivity
SCROLL_SENSITIVITY = 1.5    # Scroll speed multiplier
```

---

## 🏗️ Architecture

### Project Structure

```
GestureFlow/
├── src/
│   ├── main.py          # Application entry point
│   ├── camera.py        # Threaded camera capture
│   ├── engine.py        # Gesture recognition logic
│   ├── filters.py       # OneEuro smoothing filter
│   ├── ui.py            # HUD rendering system
│   └── config.py        # Configuration parameters
├── assets/
│   └── hand_landmarker.task  # MediaPipe model file
├── download_model.py    # Model download script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Technology Stack

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Hand Tracking** | MediaPipe Tasks API | 21-point hand landmark detection |
| **Computer Vision** | OpenCV | Image processing & display |
| **OS Control** | PyAutoGUI + pynput | Cross-platform mouse & keyboard automation |
| **Smoothing** | OneEuro Filter | Jitter reduction for cursor stability |
| **UI Framework** | NumPy + OpenCV | Lightweight HUD rendering |

### Performance Features

- ⚡ **Threaded Camera** — Non-blocking frame capture for consistent FPS
- 🔄 **Filter Pipeline** — OneEuro predictive smoothing eliminates cursor shake
- 🎨 **Efficient Rendering** — Lightweight HUD overlay with minimal overhead
- 🧠 **Smart Detection** — Adaptive confidence thresholds for stability

---

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Working

**Problem:** "Failed to initialize camera" or black screen

**Solutions:**
- **macOS:** Grant camera permissions in System Preferences → Security & Privacy → Camera
- **Windows:** Check Camera privacy settings in Settings → Privacy → Camera
- **Linux:** Ensure your user is in the `video` group: `sudo usermod -a -G video $USER`
- Try a different camera index in `config.py` (change camera index from 0 to 1)

#### Model File Missing

**Problem:** "Model file not found at assets/hand_landmarker.task"

**Solution:**
```bash
python download_model.py
```

If download fails, manually download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place in `assets/` folder.

#### PyAutoGUI Issues on Linux

**Problem:** PyAutoGUI not working on Wayland

**Solution:**
- Switch to X11 session, or
- Install additional dependencies:
  ```bash
  sudo apt-get install python3-tk python3-dev
  ```

#### Low FPS / Performance Issues

**Solutions:**
- Lower camera resolution in `config.py` (e.g., 320x240)
- Reduce `MAX_HANDS` to 1 if only using one hand
- Close other applications using the webcam
- Update graphics drivers for GPU acceleration

#### Permission Errors (macOS)

**Problem:** "Operation not permitted" when controlling mouse

**Solution:**
- Go to System Preferences → Security & Privacy → Privacy → Accessibility
- Add Terminal (or your IDE) to the list of allowed apps

### Platform-Specific Notes

#### macOS
- Requires camera and accessibility permissions
- Works on macOS 10.14 (Mojave) and later
- Apple Silicon (M1/M2) fully supported

#### Windows
- Tested on Windows 10/11
- May require Visual C++ Redistributables
- Windows Defender might flag on first run (safe to allow)

#### Linux
- Tested on Ubuntu 20.04, 22.04, Fedora 36+
- X11 recommended over Wayland for best compatibility
- Requires `python3-dev` and `libopencv-dev` on some distributions

---

## 🤝 Contributing

> [!NOTE]
> **Project Status:** This repository contains the **production-ready MediaPipe implementation** of GestureFlow, optimized for Windows 10/11. This version is feature-complete and stable.

### Current Repository
This codebase represents the best implementation using existing MediaPipe technology. Bug fixes and minor improvements are welcome:

1. Fork the repository
2. Create a bugfix branch (`git checkout -b fix/issue-description`)
3. Commit your changes (`git commit -m 'Fix: description'`)
4. Push to the branch (`git push origin fix/issue-description`)
5. Open a Pull Request

### Future Development
For **new features, custom gestures, and ML model improvements**, please wait for the upcoming **GestureFlow-ML** repository, which will feature:
- Custom CNN/DL model architecture
- Raspberry Pi 5 optimization
- Advanced gesture customization
- Community-driven model training

Stay tuned for the GestureFlow-ML repository announcement!

---

## 📋 System Requirements

### Minimum
- **CPU:** Dual-core 2.0 GHz
- **RAM:** 4 GB
- **Camera:** 720p @ 30fps
- **Python:** 3.10+

### Recommended
- **CPU:** Quad-core 2.5 GHz or better
- **RAM:** 8 GB
- **Camera:** 1080p @ 60fps
- **GPU:** Integrated or discrete GPU for MediaPipe acceleration
- **Python:** 3.11+

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MediaPipe** team for the incredible hand tracking model
- **Apple Vision Pro** for spatial computing inspiration
- **Open Source Community** for continuous support and feedback

---


---

## 🔮 Future Work

### Custom ML/CNN/DL Model Repository (Coming Soon)

This current implementation represents the **best possible solution using existing MediaPipe technology**. It's optimized, tested, and production-ready for Windows.

**Next Phase:** A new repository will be created featuring:

- 🤖 **Custom Deep Learning Model** - Lightweight CNN/DL architecture  
- 🎯 **Higher Accuracy** - Fine-tuned on diverse hand data  
- 🥧 **Raspberry Pi 5 Optimized** - ARM64 NEON optimizations, TFLite INT8 quantization  
- ⚡ **Sub-10ms Inference** - Faster than MediaPipe on edge devices  
- 📊 **Custom Training Pipeline** - Data collection, augmentation, and training scripts  
- 🔧 **Gesture Customization** - Easy to add/modify gestures  

Stay tuned for: **GestureFlow-ML** (separate repository)

---

<div align="center">

**Built with ❤️ for seamless human-computer interaction**

*Current Version: MediaPipe Implementation (Production Ready)*

[⬆ Back to Top](#-gestureflow)

</div>

[⬆ Back to Top](#-gestureflow)

</div>