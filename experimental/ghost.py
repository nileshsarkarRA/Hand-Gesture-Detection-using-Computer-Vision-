import tkinter as tk
from tkinter import font
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyttsx3
import threading
import time
import random
import sys
import os
import datetime

# --- SYSTEM CONFIGURATION ---
EXIT_PASSWORD = "fuck you"
VIDEO_FILENAME = "evidence_of_fear.avi"
SNAPSHOT_DIR = "CAPTURED_SCREAMS"

class ChaosSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("SYSTEM_HALTED")
        
        # 0. FILE SYSTEM INIT
        if not os.path.exists(SNAPSHOT_DIR):
            os.makedirs(SNAPSHOT_DIR)

        # 1. INTERFACE LOCKDOWN
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.configure(background="black", cursor="none")
        self.root.bind("<Key>", self.handle_input)
        
        # 2. UI LAYOUT
        self.main_frame = tk.Frame(root, bg="black")
        self.main_frame.pack(fill="both", expand=True)
        
        # Video Feed (The Mirror)
        self.video_label = tk.Label(self.main_frame, bg="black", borderwidth=0)
        self.video_label.pack(expand=True)

        # Heads Up Display (The Terminal)
        self.hud = tk.Label(self.main_frame, text="INITIALIZING...", bg="black", fg="#00FF00",
                            font=("Courier New", 20, "bold"))
        self.hud.place(relx=0.5, rely=0.9, anchor="center")
        
        # 3. CORE PROCESSING UNITS
        self.cap = cv2.VideoCapture(0)
        
        # Video Recorder Setup
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recorder = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 20.0, (width, height))

        # Face Tracking
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Audio Unit
        self.audio_lock = threading.Lock()
        
        # 4. STATE MACHINE
        self.running = True
        self.input_buffer = ""
        self.chaos_level = 0.0      # 0.0 to 1.0
        self.mode = "OBSERVE"       # OBSERVE, GLITCH, JOKER, DEMON, STROBE
        self.strobe_state = False   # For flashing logic
        
        # 5. EXECUTE
        self.root.after(100, self.boot_sequence)

    def audio_dispatch(self, text, speed=150, pitch="NORMAL"):
        """Non-blocking, polymorphic audio engine."""
        def _speak():
            with self.audio_lock:
                engine = pyttsx3.init()
                if pitch == "DEMON":
                    engine.setProperty('rate', 50)
                    engine.setProperty('volume', 1.0)
                elif pitch == "MANIC":
                    engine.setProperty('rate', 350)
                else:
                    engine.setProperty('rate', speed)
                engine.say(text)
                engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()

    def boot_sequence(self):
        """Phase 1: Immersion."""
        threading.Thread(target=self.ai_director, daemon=True).start()
        self.root.after(0, self.render_pipeline)

    def ai_director(self):
        """The Brain: Probabilistic State Machine."""
        time.sleep(3)
        self.audio_dispatch("I am inside.", speed=100)
        
        phrases = ["Look behind you.", "Melting.", "Do not scream.", "I own this hardware.", "Smile."]
        
        while self.running:
            # Increase Entropy
            self.chaos_level = min(self.chaos_level + 0.02, 1.0)
            
            # Roll the dice based on Chaos Level
            dice = random.random()
            
            # EVENT: EXTREME STROBE (High Priority)
            if dice < (0.05 + self.chaos_level * 0.1):
                self.mode = "STROBE"
                self.audio_dispatch("WAKE UP", pitch="MANIC")
                time.sleep(2) # Duration of strobe
                self.mode = "OBSERVE"
            
            # EVENT: DEMON MODE
            elif dice < (0.15 + self.chaos_level * 0.2):
                self.mode = "DEMON"
                self.audio_dispatch("Do you see the blood?", pitch="DEMON")
                self.hud.config(text="CRITICAL_FAILURE: BLOOD_DETECTED", fg="red")
                time.sleep(4)
                self.mode = "OBSERVE"

            # EVENT: JOKER/GLITCH
            elif dice < 0.4:
                self.mode = random.choice(["JOKER", "GLITCH"])
                self.audio_dispatch(random.choice(phrases), speed=130)
                time.sleep(3)
                self.mode = "OBSERVE"

            # IDLE STATE
            else:
                self.hud.config(text="SYSTEM: MONITORING BIOMETRICS...", fg="#00FF00")
                pass

            time.sleep(random.randint(2, 5))

    def process_frame_fx(self, frame):
        """The Visual Processing Unit (VPU)."""
        rows, cols, _ = frame.shape
        
        # 1. STROBE FX (Extreme Flashing)
        if self.mode == "STROBE":
            self.strobe_state = not self.strobe_state
            if self.strobe_state:
                # Fill Red
                frame[:] = (0, 0, 255)
                cv2.putText(frame, "RUN", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
            else:
                # Invert
                frame = cv2.bitwise_not(frame)
            return frame # Return early, no faces needed

        # 2. MELT FX (Sine Wave Displacement)
        if self.mode == "DEMON" or self.mode == "GLITCH":
            for i in range(rows):
                offset = int(25 * np.sin(i / 20.0 + time.time() * 15))
                frame[i] = np.roll(frame[i], offset, axis=0)

        # 3. COLOR FX
        if self.mode == "DEMON":
            frame[:, :, 0] = 0 # No Blue
            frame[:, :, 1] = 0 # No Green
            frame[:, :, 2] = cv2.add(frame[:, :, 2], 60) # Boost Red

        if self.mode == "GLITCH":
            # RGB Split
            frame[:, :, 2] = np.roll(frame[:, :, 2], 15, axis=1)

        # 4. FACE OVERLAYS
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            if self.mode == "JOKER":
                # Smile
                cv2.ellipse(frame, (x+w//2, y+int(h*0.7)), (w//2, h//5), 0, 0, 180, (0,0,255), 10)
                # Eyes
                cv2.circle(frame, (x+int(w*0.3), y+int(h*0.4)), 15, (0,0,0), -1)
                cv2.circle(frame, (x+int(w*0.7), y+int(h*0.4)), 15, (0,0,0), -1)
            elif self.mode == "OBSERVE":
                # Sci-Fi Box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "SUBJECT_LOCKED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def render_pipeline(self):
        """The 30FPS Loop."""
        if not self.running: return

        ret, frame = self.cap.read()
        if ret:
            # 1. Pre-process (Mirror)
            frame = cv2.flip(frame, 1)

            # 2. Apply FX
            processed_frame = self.process_frame_fx(frame)

            # 3. Record Evidence (Saves the GLITCHED version)
            self.recorder.write(processed_frame)

            # 4. Snapshot Logic (During Strobe)
            if self.mode == "STROBE" and random.random() > 0.7:
                ts = datetime.datetime.now().strftime("%H%M%S_%f")
                cv2.imwrite(f"{SNAPSHOT_DIR}/scare_{ts}.jpg", processed_frame)

            # 5. Render to GUI
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(30, self.render_pipeline)

    def handle_input(self, event):
        """Password Gatekeeper."""
        if event.char and event.char.isprintable():
            self.input_buffer += event.char
            self.input_buffer = self.input_buffer[-20:]

        if EXIT_PASSWORD in self.input_buffer.lower():
            self.terminate()
        elif event.keysym == "Escape":
            self.audio_dispatch("No escape.", pitch="DEMON")
            self.hud.config(text="ERROR: KEYBOARD_INTERRUPT_FAILED", fg="red")

    def terminate(self):
        self.running = False
        self.hud.config(text="SYSTEM RESTORED", fg="white")
        self.audio_dispatch("Goodbye.", speed=100)
        self.root.update()
        time.sleep(2)
        
        self.cap.release()
        self.recorder.release()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    # Fake Loader
    if os.name == 'nt': os.system('cls')
    else: os.system('clear')
    print("\033[91m[KERNEL] INJECTING PAYLOAD...", end="\r")
    time.sleep(2)
    print("\n[KERNEL] ROOT ACCESS GRANTED.\033[0m")
    time.sleep(1)

    root = tk.Tk()
    app = ChaosSystem(root)
    root.mainloop()