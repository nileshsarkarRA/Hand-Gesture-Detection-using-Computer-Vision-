# System Architecture & Component Breakdown

This document provides a detailed technical overview of the components used in the HyperGesture system and the reasoning behind their implementation.

## 1. Core Components

### `src/config.py`
**What**: A centralized configuration class.
**Why**: Centralizing parameters like `CAM_WIDTH`, `FPS`, and `ONE_EURO_BETA` allows for rapid tuning and experimentation without modifying the core logic. This is crucial for adapting the system to different hardware (cameras) and user environments.

### `src/camera.py` (`ThreadedCamera`)
**What**: A class that runs the OpenCV video capture in a dedicated background thread.
**Why**: Standard synchronous capture (`cap.read()`) can block the main loop, leading to frame buffer buildup and significant latency. By decoupling capture from processing, we ensure the system always operates on the *latest* possible frame, keeping latency at a minimum.

### `src/filters.py` (`OneEuroFilter`)
**What**: An implementation of the 1€ Filter, a first-order low-pass filter with an adaptive cutoff frequency.
**Why**: Human hand movement is naturally shaky. A simple static filter causes lag during fast movements. The One Euro Filter solves this by:
- Using a low cutoff for slow movements (eliminating jitter).
- Using a high cutoff for fast movements (eliminating lag).
This provides a "mechanical" feel to the cursor movement.

### `src/geometry.py` (`GeometryEngine`)
**What**: Handles coordinate mapping, point rotation, and distance calculations.
**Why**: 
- **Mapping**: Hand landmarks are in 0-1 normalized space. We map this to screen pixels while accounting for a "Virtual Trackpad" (ROI) so the user doesn't have to move their arm across the entire camera view.
- **Rotation Correction**: Most users hold their hand at a slight tilt. `rotate_point` allows us to align the "natural" vertical axis of the hand with the screen's vertical axis, making interactions more ergonomic.

### `src/controller.py` (`CursorController`)
**What**: The central logic machine that interprets gestures into system actions.
**Why**:
- **State Management**: Tracks if the user is currently clicking, dragging, or scrolling.
- **Hysteresis**: Uses "Schmitt Trigger" logic (different thresholds for activation/deactivation) to prevent accidental rapid clicks when the user's fingers are hovering near the trigger point.
- **Sniper Mode**: Dynamically reduces the filter responsiveness when fingers are close, allowing for pixel-perfect precision.

---

## 2. Technical Stack Reasoning

| Library | Purpose | Why? |
| :--- | :--- | :--- |
| **MediaPipe** | Hand Tracking | Provides ultra-fast, CPU-optimized hand landmark detection (21 points) in real-time. |
| **OpenCV** | Image Processing | The industry standard for capturing video and rendering visual feedback overlays. |
| **PyAutoGUI** | System Control | Cross-platform library for programmatically controlling the mouse and keyboard. |
| **Threading** | Performance | Essential for maintaining high FPS and low latency in computer vision tasks. |

## 3. Gesture Mapping Logic

- **Move**: Tracked via the Index PIP joint (knuckle) for maximum stability.
- **Click/Drag**: Measured by the Euclidean distance between Thumb and Index tips.
- **Scroll**: Triggered by a specific finger configuration (Index/Middle UP, Ring DOWN) to avoid confusion with normal movement.
