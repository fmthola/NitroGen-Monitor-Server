"""
Linux interactive DAgger recorder (Bazzite/Proton) — collect human corrections.

The Linux counterpart of play_interactive.py. The AI plays; when you touch your
physical controller it takes over and the (frame, AI action, your action) triple
is recorded. Those corrections train the model with scripts/train_dagger.py.

Backends swapped for Linux (Windows logic unchanged in the original):
  dxcam      -> nitrogen.linux_capture_pipewire  (Wayland-native capture)
  vgamepad   -> nitrogen.linux_gamepad           (uinput virtual Xbox 360 pad)
  pygame stays (reads your physical controller on Linux too)

Setup:
  - Plug in a physical Xbox/compatible controller (this is how you demonstrate).
  - Start the model server, then run this. Pick the Cyberpunk window in the
    share picker the first time.

  python scripts/play_interactive_linux.py --port 5555 --out ./corrections

Modes (ALT+F1/F2/F3 if pynput is available):
  AUTO  (default) AI drives; touching the controller overrides and records.
  AI    AI only, your input ignored.
  HUMAN you drive, AI paused (still records your input as demonstrations).

Corrections are written to ./corrections/<timestamp>/ as frames/*.png plus
corrections.json, the format scripts/train_dagger.py expects.
"""
import os
import sys
import json
import time
import argparse
import threading

import numpy as np
import cv2
from PIL import Image

import nitrogen.linux_capture_pipewire as pwcap
import nitrogen.linux_gamepad as vg
from nitrogen.inference_client import ModelClient
from nitrogen.shared import BUTTON_ACTION_TOKENS

import pygame

try:
    from pynput import keyboard as _pk
    HOTKEYS = True
except Exception:
    HOTKEYS = False

DEADZONE = 0.15            # human is "active" when a stick passes this
TOKENS = BUTTON_ACTION_TOKENS
THRES = 0.5

# Output button names train_dagger.py / CorrectionDataset expect.
BUTTON_ORDER = [
    "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "east", "left_shoulder", "left_thumb", "left_trigger",
    "north", "right_shoulder", "right_thumb", "right_trigger",
    "south", "west",
]

mode = "auto"
mode_lock = threading.Lock()


class PhysicalController:
    """Reads a physical Xbox/compatible pad via pygame (works on Linux)."""

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No physical controller detected. Plug in an Xbox/compatible pad "
                "to demonstrate corrections.")
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"Physical controller: {self.js.get_name()} "
              f"({self.js.get_numaxes()} axes, {self.js.get_numbuttons()} buttons)")

    def get_state(self):
        pygame.event.pump()
        n_ax = self.js.get_numaxes()
        ax = lambda i: self.js.get_axis(i) if n_ax > i else 0.0
        j_left = [ax(0), ax(1)]
        j_right = [ax(2), ax(3)]
        lt = (ax(4) + 1) / 2 if n_ax > 4 else 0.0
        rt = (ax(5) + 1) / 2 if n_ax > 5 else 0.0

        face = ["south", "east", "west", "north", "left_shoulder",
                "right_shoulder", "back", "start", "left_thumb", "right_thumb", "guide"]
        buttons = {}
        for i, name in enumerate(face):
            buttons[name] = self.js.get_button(i) if i < self.js.get_numbuttons() else 0
        if self.js.get_numhats() > 0:
            hx, hy = self.js.get_hat(0)
        else:
            hx = hy = 0
        buttons["dpad_left"] = 1 if hx < 0 else 0
        buttons["dpad_right"] = 1 if hx > 0 else 0
        buttons["dpad_down"] = 1 if hy < 0 else 0
        buttons["dpad_up"] = 1 if hy > 0 else 0
        buttons["left_trigger"] = 1 if lt > 0.5 else 0
        buttons["right_trigger"] = 1 if rt > 0.5 else 0
        return {"j_left": j_left, "j_right": j_right,
                "lt": lt, "rt": rt, "buttons": buttons}

    def is_active(self, s):
        if abs(s["j_left"][0]) > DEADZONE or abs(s["j_left"][1]) > DEADZONE:
            return True
        if abs(s["j_right"][0]) > DEADZONE or abs(s["j_right"][1]) > DEADZONE:
            return True
        if s["lt"] > 0.3 or s["rt"] > 0.3:
            return True
        return any(s["buttons"].values())


class CorrectionRecorder:
    def __init__(self, out_dir, timestamp):
        self.dir = os.path.join(out_dir, timestamp)
        self.frames = os.path.join(self.dir, "frames")
        os.makedirs(self.frames, exist_ok=True)
        self.records = []
        self.count = 0

    def record(self, frame_img, human, ai):
        path = os.path.join(self.frames, f"{self.count:06d}.png")
        frame_img.save(path)
        to_list = lambda v: v.tolist() if hasattr(v, "tolist") else list(v)
        self.records.append({
            "frame_path": path,
            "human": {"j_left": human["j_left"], "j_right": human["j_right"],
                      "buttons": human["buttons"]},
            "ai": {"j_left": to_list(ai["j_left"]), "j_right": to_list(ai["j_right"]),
                   "buttons": to_list(ai["buttons"])},
        })
        self.count += 1
        if self.count % 20 == 0:
            self.save()

    def save(self):
        with open(os.path.join(self.dir, "corrections.json"), "w") as f:
            json.dump(self.records, f, indent=2)


def set_mode(m):
    global mode
    with mode_lock:
        mode = m
    print(f">>> MODE: {m.upper()}")


def setup_hotkeys():
    if not HOTKEYS:
        print("Note: install pynput for ALT+F1/F2/F3 mode switching")
        return
    hk = _pk.GlobalHotKeys({
        "<alt>+<f1>": lambda: set_mode("ai"),
        "<alt>+<f2>": lambda: set_mode("human"),
        "<alt>+<f3>": lambda: set_mode("auto"),
    })
    hk.daemon = True
    hk.start()


def apply_to_pad(pad, j_left, j_right, buttons_named):
    """Drive the virtual pad from a named-button action (the human passthrough)."""
    pad.reset()
    pad.left_joystick(x_value=int(j_left[0] * 32767), y_value=int(-j_left[1] * 32767))
    pad.right_joystick(x_value=int(j_right[0] * 32767), y_value=int(-j_right[1] * 32767))
    pad.left_trigger(value=int(buttons_named.get("left_trigger", 0) * 255))
    pad.right_trigger(value=int(buttons_named.get("right_trigger", 0) * 255))
    name_to_btn = {
        "south": vg.XUSB_BUTTON.XUSB_GAMEPAD_A, "east": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "west": vg.XUSB_BUTTON.XUSB_GAMEPAD_X, "north": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "left_shoulder": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        "right_shoulder": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        "left_thumb": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        "right_thumb": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
        "dpad_up": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        "dpad_down": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        "dpad_left": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        "dpad_right": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    }
    for name, code in name_to_btn.items():
        if buttons_named.get(name, 0):
            pad.press_button(button=code)
    pad.update()


def apply_ai(pad, j_left, j_right, buttons_seq):
    """Drive the virtual pad from the model's token-ordered action."""
    pad.reset()
    pad.left_joystick(x_value=int(j_left[0] * 32767), y_value=int(-j_left[1] * 32767))
    pad.right_joystick(x_value=int(j_right[0] * 32767), y_value=int(-j_right[1] * 32767))
    name_to_btn = {
        "SOUTH": vg.XUSB_BUTTON.XUSB_GAMEPAD_A, "EAST": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "WEST": vg.XUSB_BUTTON.XUSB_GAMEPAD_X, "NORTH": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "LEFT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        "RIGHT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
    }
    for i, name in enumerate(TOKENS):
        if name == "LEFT_TRIGGER":
            pad.left_trigger(value=int(buttons_seq[i] * 255))
        elif name == "RIGHT_TRIGGER":
            pad.right_trigger(value=int(buttons_seq[i] * 255))
        elif name in name_to_btn and buttons_seq[i] > THRES:
            pad.press_button(button=name_to_btn[name])
    pad.update()


def _decide_control(m, human_active, pad, obs, hs, ai_action, jl, jr, btn, rec):
    """Apply the action for the current mode and return the control label."""
    if m == "human" or (m == "auto" and human_active):
        # Human in control: neutralize the virtual pad so the physical
        # controller drives the game, and record the correction.
        pad.reset(); pad.update()
        rec.record(obs, hs, ai_action)
        return "HUMAN"
    elif m == "ai" or m == "auto":
        apply_ai(pad, jl[0], jr[0], btn[0])
        return "AI"
    return "?"


def main():
    ap = argparse.ArgumentParser(description="Linux interactive DAgger recorder")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--out", type=str, default="./corrections")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    timestamp = str(int(time.time()))
    controller = PhysicalController()
    setup_hotkeys()

    print("Connecting to model server...")
    policy = ModelClient(port=args.port)
    policy.reset()
    print(f"Connected: {policy.info()['ckpt_path']}")

    pad = vg.VX360Gamepad()
    pad.update()

    print("Starting capture (pick the Cyberpunk window in the picker)...")
    cam = pwcap.create()
    cam.start()

    rec = CorrectionRecorder(args.out, timestamp)
    print(f"\nRecording corrections to {rec.dir}")
    print("AUTO mode: AI drives; touch the controller to correct. Ctrl+C to stop.\n")
    for i in (3, 2, 1):
        print(f"{i}...")
        time.sleep(1)

    frame_time = 1.0 / args.fps
    last_frame = None
    steps = 0
    try:
        while True:
            t0 = time.perf_counter()
            raw = cam.get_latest_frame()
            if raw is None:
                raw = last_frame
            else:
                last_frame = raw
            if raw is None:
                time.sleep(0.05)
                continue
            obs = Image.fromarray(cv2.resize(raw, (256, 256), interpolation=cv2.INTER_NEAREST))

            pred = policy.predict(obs)
            jl, jr, btn = pred["j_left"], pred["j_right"], pred["buttons"]
            if len(jl) == 0:
                continue
            ai_action = {"j_left": jl[0], "j_right": jr[0], "buttons": btn[0]}

            with mode_lock:
                m = mode
            hs = controller.get_state()
            human_active = controller.is_active(hs)

            ctrl = _decide_control(m, human_active, pad, obs, hs, ai_action, jl, jr, btn, rec)

            steps += 1
            if steps % 10 == 0:
                print(f"step {steps:5d} | {ctrl:5s} | corrections {rec.count:4d} "
                      f"| vis {float(np.asarray(obs).mean()):4.0f}", flush=True)

            dt = frame_time - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        rec.save()
        pad.close()
        cam.stop()
        print(f"Saved {rec.count} corrections to {rec.dir}")
        print(f"Train with: python scripts/train_dagger.py --corrections {os.path.dirname(rec.dir)} --epochs 5")


if __name__ == "__main__":
    main()
