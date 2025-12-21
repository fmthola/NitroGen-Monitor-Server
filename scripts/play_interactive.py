"""
Interactive NitroGen Play Script - Human Override + DAgger Training

This script lets you correct the AI in real-time:
- AI plays normally
- When you touch your controller, YOUR input overrides the AI
- Corrections are recorded for future fine-tuning (DAgger)

CONTROLS:
  Your physical controller overrides AI when:
  - Joystick moved beyond deadzone (default 0.15)
  - Any button pressed

  Keyboard hotkeys:
  - ALT+F1 = Force AI mode (ignore your controller)
  - ALT+F2 = Force HUMAN mode (AI paused)
  - ALT+F3 = AUTO mode (default - AI with override)
  - ALT+F9 = Toggle joystick sensitivity (50% / 100%)
  - ALT+F10 = Quit

USAGE:
  python scripts/play_interactive.py --process "YourGame.exe"

OUTPUT:
  Corrections saved to: ./corrections/<timestamp>/
  Use these for DAgger fine-tuning!
"""
import os
import sys
import time
import json
import argparse
import pickle
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from PIL import Image
import dxcam
import vgamepad as vg
import zmq
import pygame
import pygetwindow as gw
import psutil

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Note: Install 'keyboard' for hotkeys: pip install keyboard")

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

from nitrogen.inference_client import ModelClient
from nitrogen.shared import BUTTON_ACTION_TOKENS

# =============================================================================
# CONFIGURATION
# =============================================================================
JOYSTICK_DEADZONE = 0.15  # Human input detected if joystick > this
BUTTON_PRESS_THRES = 0.5  # AI button threshold
CORRECTION_SAVE_INTERVAL = 1  # Save every N corrections (1 = save all)

# Control modes
MODE_AUTO = "auto"      # AI plays, human can override
MODE_AI = "ai"          # AI only, ignore human input
MODE_HUMAN = "human"    # Human only, AI paused

# =============================================================================
# GLOBAL STATE
# =============================================================================
current_mode = MODE_AUTO
mode_lock = threading.Lock()
joystick_scale = 1.0
correction_count = 0
session_corrections = []

# Overlay
overlay_window = None
overlay_label = None

# =============================================================================
# PHYSICAL CONTROLLER (pygame)
# =============================================================================
class PhysicalController:
    """Reads input from physical Xbox controller using pygame."""

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No physical controller detected!\n"
                "Connect an Xbox controller to provide corrections."
            )

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.name = self.joystick.get_name()
        print(f"Physical controller: {self.name}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")

    def get_state(self):
        """Get current controller state."""
        pygame.event.pump()

        # Joysticks (normalized to [-1, 1])
        j_left = [
            self.joystick.get_axis(0),  # Left X
            self.joystick.get_axis(1),  # Left Y
        ]
        j_right = [
            self.joystick.get_axis(2) if self.joystick.get_numaxes() > 2 else 0,
            self.joystick.get_axis(3) if self.joystick.get_numaxes() > 3 else 0,
        ]

        # Triggers (convert from [-1,1] to [0,1])
        left_trigger = (self.joystick.get_axis(4) + 1) / 2 if self.joystick.get_numaxes() > 4 else 0
        right_trigger = (self.joystick.get_axis(5) + 1) / 2 if self.joystick.get_numaxes() > 5 else 0

        # Buttons
        buttons = {}
        button_map = [
            'south', 'east', 'west', 'north',
            'left_shoulder', 'right_shoulder',
            'back', 'start', 'left_thumb', 'right_thumb', 'guide'
        ]
        for i, name in enumerate(button_map):
            if i < self.joystick.get_numbuttons():
                buttons[name] = self.joystick.get_button(i)
            else:
                buttons[name] = 0

        # D-pad
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)
            buttons['dpad_left'] = 1 if hat[0] < 0 else 0
            buttons['dpad_right'] = 1 if hat[0] > 0 else 0
            buttons['dpad_down'] = 1 if hat[1] < 0 else 0
            buttons['dpad_up'] = 1 if hat[1] > 0 else 0
        else:
            buttons['dpad_left'] = buttons['dpad_right'] = 0
            buttons['dpad_down'] = buttons['dpad_up'] = 0

        # Add triggers as buttons
        buttons['left_trigger'] = 1 if left_trigger > 0.5 else 0
        buttons['right_trigger'] = 1 if right_trigger > 0.5 else 0

        return {
            'j_left': j_left,
            'j_right': j_right,
            'left_trigger_analog': left_trigger,
            'right_trigger_analog': right_trigger,
            'buttons': buttons,
        }

    def is_active(self, state, deadzone=JOYSTICK_DEADZONE):
        """Check if human is providing input."""
        # Check joysticks
        if abs(state['j_left'][0]) > deadzone or abs(state['j_left'][1]) > deadzone:
            return True
        if abs(state['j_right'][0]) > deadzone or abs(state['j_right'][1]) > deadzone:
            return True

        # Check triggers
        if state['left_trigger_analog'] > 0.1 or state['right_trigger_analog'] > 0.1:
            return True

        # Check buttons (except guide)
        for name, pressed in state['buttons'].items():
            if name != 'guide' and pressed:
                return True

        return False


# =============================================================================
# OVERLAY WINDOW
# =============================================================================
def create_overlay():
    """Create overlay showing current mode."""
    global overlay_window, overlay_label
    if not TKINTER_AVAILABLE:
        return

    try:
        overlay_window = tk.Tk()
        overlay_window.title("NitroGen Interactive")
        overlay_window.attributes("-topmost", True)
        overlay_window.attributes("-alpha", 0.9)
        overlay_window.overrideredirect(True)
        overlay_window.geometry("280x60+10+10")
        overlay_window.configure(bg="black")

        overlay_label = tk.Label(
            overlay_window,
            text="MODE: AUTO | AI",
            font=("Consolas", 12, "bold"),
            fg="lime",
            bg="black"
        )
        overlay_label.pack(expand=True, fill="both")
        overlay_window.update()
    except Exception as e:
        print(f"Overlay disabled: {e}")
        overlay_window = None


def update_overlay(controlling="AI", corrections=0):
    """Update overlay with current state."""
    global overlay_window, overlay_label
    if not overlay_window or not overlay_label:
        return

    try:
        mode_colors = {
            MODE_AUTO: ("gray", "AUTO"),
            MODE_AI: ("cyan", "AI ONLY"),
            MODE_HUMAN: ("yellow", "HUMAN"),
        }

        color, mode_text = mode_colors.get(current_mode, ("white", "???"))

        # Controlling indicator
        if controlling == "HUMAN":
            ctrl_color = "yellow"
        else:
            ctrl_color = "lime"

        scale_text = "" if joystick_scale == 1.0 else " [50%]"

        # Build display text
        text = f"{mode_text} | {controlling}{scale_text}\nCorrections: {corrections}"

        overlay_label.config(text=text, fg=ctrl_color)
        overlay_window.update()
    except:
        pass


def destroy_overlay():
    global overlay_window
    if overlay_window:
        try:
            overlay_window.destroy()
        except:
            pass


# =============================================================================
# MODE SWITCHING
# =============================================================================
def set_mode(mode):
    global current_mode
    with mode_lock:
        current_mode = mode
        print(f">>> MODE: {mode.upper()}")


def toggle_joystick_scale():
    global joystick_scale
    joystick_scale = 0.5 if joystick_scale == 1.0 else 1.0
    print(f">>> JOYSTICK SCALE: {int(joystick_scale * 100)}%")


def setup_hotkeys():
    if not KEYBOARD_AVAILABLE:
        return

    keyboard.add_hotkey("alt+f1", lambda: set_mode(MODE_AI))
    keyboard.add_hotkey("alt+f2", lambda: set_mode(MODE_HUMAN))
    keyboard.add_hotkey("alt+f3", lambda: set_mode(MODE_AUTO))
    keyboard.add_hotkey("alt+f9", toggle_joystick_scale)

    print("Hotkeys:")
    print("  ALT+F1 = AI only (ignore controller)")
    print("  ALT+F2 = HUMAN only (pause AI)")
    print("  ALT+F3 = AUTO (AI + override)")
    print("  ALT+F9 = Toggle joystick sensitivity")


# =============================================================================
# CORRECTION RECORDING
# =============================================================================
class CorrectionRecorder:
    """Records human corrections for DAgger training."""

    def __init__(self, output_dir: str, game_name: str):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"{game_name}_{self.session_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        self.corrections = []
        self.count = 0

        print(f"Corrections will be saved to: {self.output_dir}")

    def record(self, frame: Image.Image, human_action: dict, ai_action: dict):
        """Record a correction."""
        self.count += 1

        # Save frame
        frame_path = self.frames_dir / f"{self.count:06d}.png"
        frame.save(frame_path)

        # Record action data
        correction = {
            "frame_idx": self.count,
            "frame_path": str(frame_path),
            "timestamp": time.time(),
            "human": {
                "j_left": human_action["j_left"],
                "j_right": human_action["j_right"],
                "buttons": human_action["buttons"],
            },
            "ai": {
                "j_left": ai_action["j_left"].tolist() if hasattr(ai_action["j_left"], "tolist") else ai_action["j_left"],
                "j_right": ai_action["j_right"].tolist() if hasattr(ai_action["j_right"], "tolist") else ai_action["j_right"],
                "buttons": ai_action["buttons"].tolist() if hasattr(ai_action["buttons"], "tolist") else ai_action["buttons"],
            }
        }
        self.corrections.append(correction)

        # Periodic save
        if self.count % 100 == 0:
            self.save()

        return self.count

    def save(self):
        """Save corrections to disk."""
        if not self.corrections:
            return

        # Save as JSON
        json_path = self.output_dir / "corrections.json"
        with open(json_path, "w") as f:
            json.dump(self.corrections, f, indent=2)

        # Also save as parquet for training
        try:
            import polars as pl

            # Flatten for parquet
            rows = []
            for c in self.corrections:
                row = {
                    "frame_idx": c["frame_idx"],
                    "frame_path": c["frame_path"],
                    "timestamp": c["timestamp"],
                    "j_left": c["human"]["j_left"],
                    "j_right": c["human"]["j_right"],
                }
                # Add button columns
                for btn_name, btn_val in c["human"]["buttons"].items():
                    row[btn_name] = btn_val
                rows.append(row)

            df = pl.DataFrame(rows)
            df.write_parquet(self.output_dir / "corrections.parquet")
        except ImportError:
            pass  # polars not available

        print(f"Saved {len(self.corrections)} corrections to {self.output_dir}")

    def close(self):
        self.save()

        # Save metadata
        metadata = {
            "session_id": self.session_id,
            "total_corrections": self.count,
            "output_dir": str(self.output_dir),
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# GAME WINDOW DETECTION
# =============================================================================
def find_game_window(process_name):
    """Find game window by process name."""
    game_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
            game_pid = proc.info['pid']
            break

    if not game_pid:
        raise Exception(f"Process not found: {process_name}")

    print(f"Found process: {process_name} (PID: {game_pid})")

    windows = gw.getAllWindows()
    game_window = None

    for window in windows:
        game_base = process_name.lower().replace('.exe', '')
        if game_base in window.title.lower():
            game_window = window
            print(f"Found window: '{window.title}'")
            break

    if not game_window:
        for window in windows:
            if window.visible and window.width > 640 and window.height > 480:
                if any(x in window.title.lower() for x in ['cyberpunk', 'game', process_name.lower().replace('.exe', '')]):
                    game_window = window
                    break

    if not game_window:
        raise Exception(f"No window found for {process_name}")

    left, top = max(0, game_window.left), max(0, game_window.top)
    width, height = game_window.width, game_window.height

    print(f"Window region: ({left}, {top}) - {width}x{height}")
    return (left, top, left + width, top + height)


# =============================================================================
# VIRTUAL CONTROLLER OUTPUT
# =============================================================================
BUTTON_MAP = {
    "SOUTH": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "EAST": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "WEST": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "NORTH": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "LEFT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    "RIGHT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
    "LEFT_THUMB": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
    "RIGHT_THUMB": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
    "BACK": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
    "START": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
    "GUIDE": vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,
    "DPAD_UP": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
    "DPAD_DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
    "DPAD_LEFT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
    "DPAD_RIGHT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
}

# Reverse mapping for human input
BUTTON_NAME_MAP = {
    "south": "SOUTH",
    "east": "EAST",
    "west": "WEST",
    "north": "NORTH",
    "left_shoulder": "LEFT_SHOULDER",
    "right_shoulder": "RIGHT_SHOULDER",
    "left_thumb": "LEFT_THUMB",
    "right_thumb": "RIGHT_THUMB",
    "back": "BACK",
    "start": "START",
    "guide": "GUIDE",
    "dpad_up": "DPAD_UP",
    "dpad_down": "DPAD_DOWN",
    "dpad_left": "DPAD_LEFT",
    "dpad_right": "DPAD_RIGHT",
}


def apply_human_action(gamepad, human_state):
    """Apply human controller input to virtual gamepad."""
    gamepad.reset()

    scale = joystick_scale

    # Joysticks
    lx, ly = human_state['j_left']
    rx, ry = human_state['j_right']

    gamepad.left_joystick(
        x_value=int(lx * 32767 * scale),
        y_value=int(-ly * 32767 * scale)
    )
    gamepad.right_joystick(
        x_value=int(rx * 32767 * scale),
        y_value=int(-ry * 32767 * scale)
    )

    # Triggers
    gamepad.left_trigger(value=int(human_state['left_trigger_analog'] * 255))
    gamepad.right_trigger(value=int(human_state['right_trigger_analog'] * 255))

    # Buttons
    for btn_lower, btn_upper in BUTTON_NAME_MAP.items():
        if human_state['buttons'].get(btn_lower, 0):
            if btn_upper in BUTTON_MAP:
                gamepad.press_button(button=BUTTON_MAP[btn_upper])

    gamepad.update()


def apply_ai_action(gamepad, j_left, j_right, buttons, token_set):
    """Apply AI prediction to virtual gamepad."""
    gamepad.reset()

    scale = joystick_scale

    # Joysticks
    gamepad.left_joystick(
        x_value=int(j_left[0] * 32767 * scale),
        y_value=int(-j_left[1] * 32767 * scale)
    )
    gamepad.right_joystick(
        x_value=int(j_right[0] * 32767 * scale),
        y_value=int(-j_right[1] * 32767 * scale)
    )

    # Triggers
    lt_idx = token_set.index("LEFT_TRIGGER") if "LEFT_TRIGGER" in token_set else -1
    rt_idx = token_set.index("RIGHT_TRIGGER") if "RIGHT_TRIGGER" in token_set else -1

    if lt_idx >= 0:
        gamepad.left_trigger(value=int(buttons[lt_idx] * 255))
    if rt_idx >= 0:
        gamepad.right_trigger(value=int(buttons[rt_idx] * 255))

    # Buttons
    for i, (name, value) in enumerate(zip(token_set, buttons)):
        if name in ["LEFT_TRIGGER", "RIGHT_TRIGGER"]:
            continue
        if value > BUTTON_PRESS_THRES and name in BUTTON_MAP:
            gamepad.press_button(button=BUTTON_MAP[name])

    gamepad.update()


def preprocess_frame(frame):
    """Resize frame to 256x256 for model input."""
    if frame is None:
        return None
    resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(resized)


# =============================================================================
# MAIN
# =============================================================================
def main():
    global correction_count

    parser = argparse.ArgumentParser(description="Interactive NitroGen with Human Override")
    parser.add_argument("--process", type=str, required=True, help="Game process name")
    parser.add_argument("--port", type=int, default=5555, help="Model server port")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS")
    parser.add_argument("--corrections-dir", type=str, default="./corrections",
                        help="Directory to save corrections")
    parser.add_argument("--deadzone", type=float, default=0.15,
                        help="Joystick deadzone for human detection")
    args = parser.parse_args()

    # Initialize overlay
    create_overlay()
    setup_hotkeys()

    # Initialize physical controller
    print("\n" + "="*60)
    print("Initializing physical controller...")
    try:
        physical_controller = PhysicalController()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nConnect an Xbox controller to enable corrections.")
        print("Running in AI-only mode...")
        physical_controller = None

    # Initialize correction recorder
    game_name = args.process.replace('.exe', '').replace('.', '_')
    recorder = CorrectionRecorder(args.corrections_dir, game_name)

    # Connect to model server
    print("\nConnecting to model server...")
    policy = ModelClient(port=args.port)
    policy.reset()
    policy_info = policy.info()
    print(f"Connected! Model: {policy_info['ckpt_path']}")

    # Initialize virtual gamepad
    print("\nInitializing virtual Xbox controller...")
    gamepad = vg.VX360Gamepad()
    gamepad.update()
    print("Virtual controller ready!")

    # Find game window
    print(f"\nLooking for game window: {args.process}")
    game_region = find_game_window(args.process)

    # Initialize screen capture
    print("\nInitializing screen capture...")
    camera = dxcam.create(output_color="RGB", region=game_region)
    camera.start(target_fps=60, video_mode=True)
    print("Screen capture ready!")

    TOKEN_SET = BUTTON_ACTION_TOKENS

    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print("="*60)
    print("AI will play. Touch your controller to override!")
    print("Corrections are recorded for fine-tuning.")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("GO!\n")

    frame_time = 1.0 / args.fps
    step_count = 0
    last_frame = None
    controlling = "AI"

    try:
        while True:
            loop_start = time.perf_counter()

            # Capture frame
            frame = camera.get_latest_frame()
            if frame is None:
                frame = last_frame
            else:
                last_frame = frame

            if frame is None:
                print("Waiting for frame...")
                time.sleep(0.1)
                continue

            # Preprocess for model
            obs = preprocess_frame(frame)

            # Get AI prediction
            pred = policy.predict(obs)
            j_left_seq, j_right_seq, buttons_seq = pred["j_left"], pred["j_right"], pred["buttons"]

            ai_action = {
                "j_left": j_left_seq[0] if len(j_left_seq) > 0 else [0, 0],
                "j_right": j_right_seq[0] if len(j_right_seq) > 0 else [0, 0],
                "buttons": buttons_seq[0] if len(buttons_seq) > 0 else [0] * len(TOKEN_SET),
            }

            # Get human input
            human_active = False
            human_state = None

            if physical_controller:
                human_state = physical_controller.get_state()
                human_active = physical_controller.is_active(human_state, args.deadzone)

            # Decide what to do based on mode
            with mode_lock:
                mode = current_mode

            if mode == MODE_AI:
                # AI only, ignore human
                controlling = "AI"
                apply_ai_action(gamepad, ai_action["j_left"], ai_action["j_right"],
                               ai_action["buttons"], TOKEN_SET)

            elif mode == MODE_HUMAN:
                # Human only
                controlling = "HUMAN"
                if human_state:
                    apply_human_action(gamepad, human_state)

                    # Record as correction (human is always "correcting" in this mode)
                    human_action = {
                        "j_left": human_state["j_left"],
                        "j_right": human_state["j_right"],
                        "buttons": human_state["buttons"],
                    }
                    correction_count = recorder.record(obs, human_action, ai_action)

            elif mode == MODE_AUTO:
                # AI with human override
                if human_active and human_state:
                    controlling = "HUMAN"
                    apply_human_action(gamepad, human_state)

                    # Record the correction!
                    human_action = {
                        "j_left": human_state["j_left"],
                        "j_right": human_state["j_right"],
                        "buttons": human_state["buttons"],
                    }
                    correction_count = recorder.record(obs, human_action, ai_action)
                else:
                    controlling = "AI"
                    apply_ai_action(gamepad, ai_action["j_left"], ai_action["j_right"],
                                   ai_action["buttons"], TOKEN_SET)

            step_count += 1

            # Update overlay
            if step_count % 5 == 0:
                update_overlay(controlling, correction_count)

            # Print status
            if step_count % 30 == 0:
                elapsed = time.perf_counter() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                print(f"Step {step_count:5d} | {controlling:5s} | FPS: {fps:.1f} | Corrections: {correction_count}")

            # Maintain target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        # Cleanup
        gamepad.reset()
        gamepad.update()
        camera.stop()
        recorder.close()
        destroy_overlay()
        pygame.quit()

        print(f"\n{'='*60}")
        print(f"Session complete!")
        print(f"Total steps: {step_count}")
        print(f"Total corrections: {correction_count}")
        print(f"Corrections saved to: {recorder.output_dir}")
        print("="*60)


if __name__ == "__main__":
    main()
