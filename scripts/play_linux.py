"""
Linux NitroGen play script (Bazzite / Proton) — no DLL injection.

This is the Linux counterpart of play_simple.py. It keeps the same control loop,
gameplay modes, and monitor protocol, but swaps the three Windows-only backends
for Linux-native ones:

  dxcam (DirectX capture) -> nitrogen.linux_capture  (mss, X11/XWayland)
  vgamepad (ViGEmBus)     -> nitrogen.linux_gamepad  (uinput virtual Xbox 360)
  pygetwindow            -> nitrogen.linux_window   (xdotool / python-Xlib)

Typical Bazzite run (game on the Arc A770, inference on the RTX 3070):

  # Terminal 1 (distrobox, CUDA): model server
  python scripts/serve.py models/ng.pt --port 5555 --timesteps 2

  # Terminal 2: monitor (optional)
  python scripts/monitor.py --port 5556

  # Terminal 3: the agent (run the game in a desktop/borderless window first)
  python scripts/play_linux.py --window-name Cyberpunk --fps 15
  # ...or capture an explicit region instead of locating the window:
  python scripts/play_linux.py --region 0,0,1920,1080 --fps 15

LIVE HOTKEYS (need an X server + `pynput`):
  ALT+F1=DRIVING ALT+F2=COMBAT ALT+F3=EXPLORE ALT+F4=NONE
  ALT+F5=Toggle SOUTH(A) ALT+F6=Toggle NORTH(Y)
  ALT+F7=Toggle RIGHT_TRIGGER ALT+F8=Toggle LEFT_TRIGGER
  ALT+F9=Toggle joystick sensitivity (50% / 100%)
"""
import os
import time
import pickle
import argparse
import threading

import cv2
import numpy as np
from PIL import Image

import nitrogen.linux_capture as dxcam
import nitrogen.linux_gamepad as vg
from nitrogen.linux_window import find_game_window, region_from_str
import zmq

try:
    from pynput import keyboard as _pk
    HOTKEYS_AVAILABLE = True
except Exception:
    HOTKEYS_AVAILABLE = False

from nitrogen.inference_client import ModelClient
from nitrogen.shared import BUTTON_ACTION_TOKENS

# =============================================================================
# GAMEPLAY MODES - Block certain buttons to constrain AI behavior
# =============================================================================
# Every digital button on the pad. The "driving" mode blocks all of them, so
# only the two sticks and the two triggers reach the game. That keeps the agent
# from opening menus, swapping weapons, or leaving the car while it drives.
ALL_BUTTONS = [
    "SOUTH", "EAST", "WEST", "NORTH",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_THUMB", "RIGHT_THUMB",
    "BACK", "START", "GUIDE",
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
]
GAMEPLAY_MODES = {
    "none": [],
    "driving": ALL_BUTTONS,                 # sticks + triggers only
    "combat": ["BACK", "START"],
    "explore": ["RIGHT_TRIGGER", "LEFT_TRIGGER", "RIGHT_SHOULDER"],
}

current_mode = "none"
blocked_buttons = set()
mode_lock = threading.Lock()
joystick_scale = 1.0


def toggle_joystick_scale():
    global joystick_scale
    joystick_scale = 0.5 if joystick_scale == 1.0 else 1.0
    print(f">>> JOYSTICK SCALE: {int(joystick_scale * 100)}%")


def set_mode(mode_name):
    global current_mode, blocked_buttons
    with mode_lock:
        if mode_name in GAMEPLAY_MODES:
            current_mode = mode_name
            blocked_buttons = set(GAMEPLAY_MODES[mode_name])
            print(f">>> MODE: {mode_name.upper()} | Blocked: {list(blocked_buttons) or 'none'}")


def toggle_button(btn_name):
    global blocked_buttons
    with mode_lock:
        if btn_name in blocked_buttons:
            blocked_buttons.remove(btn_name)
            print(f">>> UNBLOCKED: {btn_name}")
        else:
            blocked_buttons.add(btn_name)
            print(f">>> BLOCKED: {btn_name}")


def setup_hotkeys():
    if not HOTKEYS_AVAILABLE:
        print("Note: install 'pynput' for live hotkeys (pip install pynput)")
        return
    hk = _pk.GlobalHotKeys({
        "<alt>+<f1>": lambda: set_mode("driving"),
        "<alt>+<f2>": lambda: set_mode("combat"),
        "<alt>+<f3>": lambda: set_mode("explore"),
        "<alt>+<f4>": lambda: set_mode("none"),
        "<alt>+<f5>": lambda: toggle_button("SOUTH"),
        "<alt>+<f6>": lambda: toggle_button("NORTH"),
        "<alt>+<f7>": lambda: toggle_button("RIGHT_TRIGGER"),
        "<alt>+<f8>": lambda: toggle_button("LEFT_TRIGGER"),
        "<alt>+<f9>": lambda: toggle_joystick_scale(),
    })
    hk.daemon = True
    hk.start()
    print("Hotkeys: ALT+F1=Driving F2=Combat F3=Explore F4=None | F5-F8=Toggle | F9=Sensitivity")


parser = argparse.ArgumentParser(description="Linux NitroGen Player (Proton/Bazzite)")
parser.add_argument("--process", type=str, default=None, help="Game process name (informational)")
parser.add_argument("--window-name", type=str, default="Cyberpunk", help="Window name to locate")
parser.add_argument("--region", type=str, default=None, help="Capture region 'left,top,width,height'")
parser.add_argument("--port", type=int, default=5555, help="Model server port")
parser.add_argument("--monitor-port", type=int, default=5556, help="Monitor publish port")
parser.add_argument("--fps", type=int, default=15, help="Target control rate (actions applied per second)")
parser.add_argument("--actions-per-step", type=int, default=8,
                    help="How many actions from each predicted chunk to play before re-inferring "
                         "(1 = upstream behavior; the model predicts a 16-action chunk, so 8 halves "
                         "inference load and smooths control). Clamped to the chunk length.")
parser.add_argument("--mode", type=str, default="none", choices=["none", "driving", "combat", "explore"])
parser.add_argument("--max-throttle", type=float, default=1.0,
                    help="Scale the accelerator (right trigger) from 0 to 1. Lower values drive "
                         "slower, which gives the reactive model more time to steer and does less "
                         "damage on contact. Try 0.35-0.5 for careful driving.")
parser.add_argument("--capture", choices=["auto", "pipewire", "mss"], default="auto",
                    help="Screen capture backend. 'auto' uses pipewire on Wayland (required: "
                         "X11/mss capture returns black for XWayland games) and mss on X11. "
                         "pipewire pops a one-time KDE picker; choose the game window.")
args = parser.parse_args()

use_pipewire = args.capture == "pipewire" or (
    args.capture == "auto" and os.environ.get("XDG_SESSION_TYPE") == "wayland")

set_mode(args.mode)
setup_hotkeys()

print("Setting up monitor publisher on port", args.monitor_port)
zmq_context = zmq.Context()
monitor_socket = zmq_context.socket(zmq.PUB)
monitor_socket.bind(f"tcp://*:{args.monitor_port}")

print("Connecting to model server...")
policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
print(f"Connected! Model: {policy_info['ckpt_path']}")

print("Initializing virtual Xbox 360 controller (uinput)...")
gamepad = vg.VX360Gamepad()
gamepad.update()
print("Virtual controller ready!")

# Set up screen capture. On Wayland the X11/mss path returns black for XWayland
# games, so use the PipeWire/portal backend there.
if use_pipewire:
    import nitrogen.linux_capture_pipewire as pwcap
    print("Initializing screen capture (pipewire/portal)...")
    print(">>> A KDE share picker will appear the first time. Pick the Cyberpunk window.")
    camera = pwcap.create(output_color="RGB")
    camera.start()
    print("Screen capture ready - capturing the shared source via PipeWire!")
else:
    if args.region:
        game_region = region_from_str(args.region)
        print(f"Using explicit capture region: {game_region}")
    else:
        print(f"Looking for game window matching: {args.window_name}")
        game_region = find_game_window(process_name=args.process, window_name=args.window_name)
        print(f"Found window region: {game_region}")
    print("Initializing screen capture (mss)...")
    camera = dxcam.create(output_color="RGB", region=game_region)
    camera.start(target_fps=60, video_mode=True)
    print("Screen capture ready - capturing game window region!")

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
    if frame is None:
        return None
    resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(resized)


def apply_action(j_left, j_right, buttons):
    gamepad.reset()
    scale = joystick_scale
    gamepad.left_joystick(x_value=int(j_left[0] * 32767 * scale), y_value=int(-j_left[1] * 32767 * scale))
    gamepad.right_joystick(x_value=int(j_right[0] * 32767 * scale), y_value=int(-j_right[1] * 32767 * scale))

    with mode_lock:
        current_blocked = blocked_buttons.copy()
    lt_idx = TOKEN_SET.index("LEFT_TRIGGER") if "LEFT_TRIGGER" in TOKEN_SET else -1
    rt_idx = TOKEN_SET.index("RIGHT_TRIGGER") if "RIGHT_TRIGGER" in TOKEN_SET else -1
    if lt_idx >= 0 and "LEFT_TRIGGER" not in current_blocked:
        gamepad.left_trigger(value=int(buttons[lt_idx] * 255))
    if rt_idx >= 0 and "RIGHT_TRIGGER" not in current_blocked:
        gamepad.right_trigger(value=int(buttons[rt_idx] * 255 * args.max_throttle))

    for name, value in zip(TOKEN_SET, buttons):
        if name in ["START", "BACK", "GUIDE", "LEFT_TRIGGER", "RIGHT_TRIGGER"]:
            continue
        if name in current_blocked:
            continue
        if value > BUTTON_PRESS_THRES and name in BUTTON_MAP:
            gamepad.press_button(button=BUTTON_MAP[name])

    gamepad.update()


print(f"\n{'='*60}")
_src = "pipewire portal" if use_pipewire else f"region {region_from_str(args.region) if args.region else args.window_name}"
print(f"Starting AI control (capture: {_src})")
print(f"Target FPS: {args.fps} | Press Ctrl+C to stop")
print(f"{'='*60}\n")

for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)
print("GO!\n")

frame_time = 1.0 / args.fps
step_count = 0
last_frame = None


def publish(obs, lx, ly, rx, ry, buttons, latency_ms, step):
    """Send the current frame + action to the monitor (best-effort, non-blocking)."""
    try:
        monitor_socket.send(pickle.dumps({
            'type': 'frame', 'image': np.array(obs), 'latency_ms': latency_ms,
        }), zmq.NOBLOCK)
        monitor_socket.send(pickle.dumps({
            'type': 'action',
            'action': {
                'left_x': float(lx), 'left_y': float(ly),
                'right_x': float(rx), 'right_y': float(ry),
                'buttons': [float(b) for b in buttons],
            },
            'step': step,
        }), zmq.NOBLOCK)
    except Exception:
        pass


try:
    while True:
        # --- inference: predict one chunk of actions from the latest frame ----
        frame = camera.get_latest_frame()
        if frame is None:
            frame = last_frame
        else:
            last_frame = frame
        if frame is None:
            print("Waiting for frame...")
            time.sleep(0.1)
            continue

        obs = preprocess_frame(frame)
        obs_brightness = float(np.asarray(obs).mean())  # >0 means real frames (not black)
        inference_start = time.perf_counter()
        pred = policy.predict(obs)
        inference_time = time.perf_counter() - inference_start
        j_left_seq, j_right_seq, buttons_seq = pred["j_left"], pred["j_right"], pred["buttons"]
        if len(j_left_seq) == 0:
            continue

        # --- play the predicted chunk at the target control rate -------------
        # The model emits a multi-action chunk; executing several actions before
        # re-inferring decouples control smoothness from inference latency and
        # cuts inference load by `actions-per-step`x (the key real-time win).
        n_play = max(1, min(args.actions_per_step, len(j_left_seq)))
        for k in range(n_play):
            act_start = time.perf_counter()
            apply_action(j_left_seq[k], j_right_seq[k], buttons_seq[k])

            if step_count % 5 == 0:
                lx, ly = j_left_seq[k]
                rx, ry = j_right_seq[k]
                publish(obs, lx, ly, rx, ry, buttons_seq[k],
                        inference_time * 1000, step_count)

            if step_count % 10 == 0:
                lx, ly = j_left_seq[k]
                rx, ry = j_right_seq[k]
                pressed = [TOKEN_SET[i] for i, b in enumerate(buttons_seq[k]) if b > BUTTON_PRESS_THRES]
                print(f"Step {step_count:5d} | target {args.fps}fps | Inf: {inference_time*1000:.0f}ms "
                      f"| vis {obs_brightness:5.1f} (>0=seeing) | L:({lx:+.2f},{ly:+.2f}) R:({rx:+.2f},{ry:+.2f}) | Btns: {pressed}",
                      flush=True)

            step_count += 1
            sleep_time = frame_time - (time.perf_counter() - act_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    gamepad.close()
    camera.stop()
    monitor_socket.close()
    zmq_context.term()
    print(f"Completed {step_count} steps")
    print("Done!")
