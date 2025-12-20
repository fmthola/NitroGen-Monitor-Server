"""
Simplified NitroGen play script - No DLL injection
Works with games that don't support speedhack (like Cyberpunk 2077)
"""
import os
import sys
import time
import json
import argparse
import pickle
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import dxcam
import vgamepad as vg
import zmq
import pygetwindow as gw
import psutil

from nitrogen.inference_client import ModelClient
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO


def find_game_window(process_name):
    """Find the game window by process name and return its bounding box"""
    # Find the process
    game_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
            game_pid = proc.info['pid']
            break

    if not game_pid:
        raise Exception(f"Process not found: {process_name}")

    print(f"Found process: {process_name} (PID: {game_pid})")

    # Find the window - try multiple methods
    windows = gw.getAllWindows()
    game_window = None

    for window in windows:
        # Check if window title contains game name (without .exe)
        game_base = process_name.lower().replace('.exe', '')
        if game_base in window.title.lower() or 'cyberpunk' in window.title.lower():
            game_window = window
            print(f"Found window: '{window.title}'")
            break

    if not game_window:
        # Fallback: just find any visible window from the list
        for window in windows:
            if window.visible and window.width > 640 and window.height > 480:
                if 'cyberpunk' in window.title.lower():
                    game_window = window
                    break

    if not game_window:
        raise Exception(f"No window found for {process_name}")

    # Get window region (left, top, width, height) for dxcam
    left, top = game_window.left, game_window.top
    width, height = game_window.width, game_window.height

    # Clamp to screen bounds
    left = max(0, left)
    top = max(0, top)

    print(f"Window region: ({left}, {top}) - {width}x{height}")
    return (left, top, left + width, top + height)

parser = argparse.ArgumentParser(description="Simple NitroGen Player (No DLL)")
parser.add_argument("--process", type=str, required=True, help="Game process name")
parser.add_argument("--port", type=int, default=5555, help="Model server port")
parser.add_argument("--monitor-port", type=int, default=5556, help="Monitor publish port")
parser.add_argument("--fps", type=int, default=15, help="Target FPS for AI decisions")
args = parser.parse_args()

# Setup ZeroMQ publisher for monitoring
print("Setting up monitor publisher on port", args.monitor_port)
zmq_context = zmq.Context()
monitor_socket = zmq_context.socket(zmq.PUB)
monitor_socket.bind(f"tcp://*:{args.monitor_port}")

# Connect to model server
print("Connecting to model server...")
policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
print(f"Connected! Model: {policy_info['ckpt_path']}")

# Initialize virtual gamepad
print("Initializing virtual Xbox controller...")
gamepad = vg.VX360Gamepad()
gamepad.update()
print("Virtual controller ready!")

# Find game window and get capture region
print(f"Looking for game window: {args.process}")
game_region = find_game_window(args.process)
print(f"Capture region: {game_region}")

# Initialize screen capture with game window region
print("Initializing screen capture...")
camera = dxcam.create(output_color="RGB", region=game_region)
camera.start(target_fps=60, video_mode=True)
print("Screen capture ready - capturing game window only!")

# Button mappings for vgamepad
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

TOKEN_SET = BUTTON_ACTION_TOKENS
BUTTON_PRESS_THRES = 0.5

def preprocess_frame(frame):
    """Resize frame to 256x256 for model input - OPTIMIZED"""
    if frame is None:
        return None
    # Use INTER_NEAREST for speed (was INTER_AREA)
    resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(resized)

def apply_action(j_left, j_right, buttons):
    """Apply predicted action to virtual gamepad - OPTIMIZED"""
    # Reset all buttons
    gamepad.reset()

    # Set joysticks (values are in [-1, 1], scale to [-32768, 32767])
    gamepad.left_joystick(x_value=int(j_left[0] * 32767), y_value=int(-j_left[1] * 32767))
    gamepad.right_joystick(x_value=int(j_right[0] * 32767), y_value=int(-j_right[1] * 32767))

    # Set buttons - unrolled for speed
    # Triggers (indices based on TOKEN_SET order)
    lt_idx = TOKEN_SET.index("LEFT_TRIGGER") if "LEFT_TRIGGER" in TOKEN_SET else -1
    rt_idx = TOKEN_SET.index("RIGHT_TRIGGER") if "RIGHT_TRIGGER" in TOKEN_SET else -1
    if lt_idx >= 0:
        gamepad.left_trigger(value=int(buttons[lt_idx] * 255))
    if rt_idx >= 0:
        gamepad.right_trigger(value=int(buttons[rt_idx] * 255))

    # Regular buttons
    for i, (name, value) in enumerate(zip(TOKEN_SET, buttons)):
        if name in ["START", "BACK", "GUIDE", "LEFT_TRIGGER", "RIGHT_TRIGGER"]:
            continue
        if value > BUTTON_PRESS_THRES and name in BUTTON_MAP:
            gamepad.press_button(button=BUTTON_MAP[name])

    gamepad.update()

print(f"\n{'='*60}")
print(f"Starting AI control for: {args.process}")
print(f"Target FPS: {args.fps}")
print(f"Press Ctrl+C to stop")
print(f"{'='*60}\n")

# Countdown
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)
print("GO!\n")

frame_time = 1.0 / args.fps
step_count = 0
last_frame = None

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

        # Get prediction from model
        inference_start = time.perf_counter()
        pred = policy.predict(obs)
        inference_time = time.perf_counter() - inference_start
        j_left_seq, j_right_seq, buttons_seq = pred["j_left"], pred["j_right"], pred["buttons"]

        # Calculate timing now (after inference)
        elapsed = time.perf_counter() - loop_start
        actual_fps = 1.0 / elapsed if elapsed > 0 else 0

        # Calculate pressed buttons for logging
        pressed_buttons = []
        if len(j_left_seq) > 0:
            pressed_buttons = [TOKEN_SET[i] for i, b in enumerate(buttons_seq[0]) if b > BUTTON_PRESS_THRES]

        # Apply first action from sequence
        if len(j_left_seq) > 0:
            apply_action(j_left_seq[0], j_right_seq[0], buttons_seq[0])

            # Publish to monitor - only every 5th frame to reduce overhead
            if step_count % 5 == 0:
                lx, ly = j_left_seq[0]
                rx, ry = j_right_seq[0]

                # Send frame message
                frame_msg = {
                    'type': 'frame',
                    'image': np.array(obs),  # 256x256 RGB
                    'latency_ms': elapsed * 1000
                }
                try:
                    monitor_socket.send(pickle.dumps(frame_msg), zmq.NOBLOCK)
                except:
                    pass

                # Send action message
                action_msg = {
                    'type': 'action',
                    'action': {
                        'left_x': float(lx),
                        'left_y': float(ly),
                        'right_x': float(rx),
                        'right_y': float(ry),
                        'buttons': [float(b) for b in buttons_seq[0]]
                    },
                    'step': step_count
                }
                try:
                    monitor_socket.send(pickle.dumps(action_msg), zmq.NOBLOCK)
                except:
                    pass

        step_count += 1

        # Print status every 10 steps with flush
        if step_count % 10 == 0:
            lx, ly = j_left_seq[0] if len(j_left_seq) > 0 else (0, 0)
            rx, ry = j_right_seq[0] if len(j_right_seq) > 0 else (0, 0)
            btns = pressed_buttons if 'pressed_buttons' in dir() else []
            print(f"Step {step_count:5d} | FPS: {actual_fps:5.1f} | Inf: {inference_time*1000:.0f}ms | L:({lx:+.2f},{ly:+.2f}) R:({rx:+.2f},{ry:+.2f}) | Btns: {btns}", flush=True)

        # Maintain target FPS
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    # Reset gamepad
    gamepad.reset()
    gamepad.update()
    camera.stop()
    monitor_socket.close()
    zmq_context.term()
    print(f"Completed {step_count} steps")
    print("Done!")
