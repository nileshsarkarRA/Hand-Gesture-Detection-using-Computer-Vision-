# Hand Gesture Detection using Computer Vision

This project enables computer interaction through hand gestures using Mediapipe, OpenCV, and PyAutoGUI. It allows for mouse movement, clicking, dragging, and scrolling, all performed with simple hand gestures captured via webcam.

## Features
- **Smooth Cursor Movement**: Uses the One Euro Filter to reduce jitter while maintaining low latency.
- **Precision Mode (Sniper Mode)**: Automatically dampens movement when fingers are close together, allowing for high-accuracy selection of small buttons/icons.
- **Rotation Correction**: Compensates for natural arm tilt (e.g., -20 degrees) to ensure vertical hand movement maps to vertical screen movement.
- **Threaded Camera**: Prevents visual lag by decoupling camera capture from logic processing.

## Gesture Guide
| Gesture | Action | Description |
| :--- | :--- | :--- |
| **Move Hand** | Cursor Movement | The cursor follows your middle knuckle (`Landmark 9`). |
| **Pinch (Quick)** | Left Click | Pinch thumb and index tips together and release quickly. |
| **Pinch (Hold)** | Drag & Drop | Pinch and hold for > 0.3s to start dragging; release to drop. |
| **Two Fingers Up** | Scroll Mode | Raise index and middle fingers (keep ring down). Move whole hand up/down to scroll. |
| **Pinch Distance** | Precision | The closer your pinch, the slower/more precise the cursor becomes. |

## Project Structure
- `main.py`: Clean entry point coordinating all modules.
- `src/`: Modular source code.
  - `config.py`: **The Settings Hub**. Change sensitivity, thresholds, and tilt here.
  - `camera.py`: High-speed background camera handler.
  - `filters.py`: The "One Euro" jitter-removal math.
  - `geometry.py`: Coordinate mapping and "Trackpad" ROI logic.
  - `detector.py`: MediaPipe AI implementation.
  - `controller.py`: The logic engine (decides what a gesture means).

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```bash
python main.py
```
- **Quit**: Press 'Q' while the preview window is focused.
- **Configuration**: Edit `src/config.py` to tune the sensitivity to your liking.

## Troubleshooting
- **Cursor moving wrong way?**: Go to `src/config.py` and adjust `TILT_ANGLE_DEG`.
- **Double clicking randomly?**: Increase `PINCH_DOWN_THRESH` in `src/config.py`.
- **Can't reach screen edges?**: Decrease `ROI_SCALE` in `src/config.py`.
- **Visual lag?**: Ensure you are running in a well-lit room. MediaPipe requires clear finger visibility.

## Author
**Nilesh Sarkar**

## License
MIT License
