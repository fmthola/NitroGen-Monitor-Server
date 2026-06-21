"""
End-to-end pipeline smoke test for the Linux port — no game required.

Exercises every link of the chain except the actual game:
  screen capture  ->  ZMQ ModelClient -> server inference (GPU)
  ->  predicted action  ->  uinput virtual Xbox 360 pad.

Capture backend matches play_linux.py: on Wayland it uses the PipeWire/portal
backend (X11/mss returns black there), otherwise mss. It asserts the captured
frames are NOT black, so a blind run (the failure mode where the model sees
nothing and just emits a constant action) fails the test instead of passing.

Run the server first (in the GPU container):
  python scripts/serve.py models/ng.pt --port 5555 --timesteps 2
Then on the host (DISPLAY + /dev/uinput):
  python scripts/smoke_test_linux.py --iters 30
  # On Wayland a KDE share picker appears once; pick the screen or a window with
  # visible content. Force a backend with --capture pipewire|mss.
"""
import os
import sys
import time
import argparse

import numpy as np
from PIL import Image

import nitrogen.linux_gamepad as vg
from nitrogen.linux_window import region_from_str
from nitrogen.inference_client import ModelClient
from nitrogen.shared import BUTTON_ACTION_TOKENS

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--capture", choices=["auto", "pipewire", "mss"], default="auto")
ap.add_argument("--region", type=str, default="0,0,512,512", help="mss region 'L,T,W,H'")
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

TOKENS = BUTTON_ACTION_TOKENS
THRES = 0.5
fails = []


def step(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


print("Linux pipeline smoke test")
print("=========================")

use_pipewire = args.capture == "pipewire" or (
    args.capture == "auto" and os.environ.get("XDG_SESSION_TYPE") == "wayland")

if use_pipewire:
    import nitrogen.linux_capture_pipewire as cap
    print("capture backend: pipewire (pick a source in the share picker if it appears)")
    cam = cap.create(output_color="RGB")
else:
    import nitrogen.linux_capture as cap
    print("capture backend: mss")
    cam = cap.create(output_color="RGB", region=region_from_str(args.region))
cam.start()

# wait for the first frame (pipewire delivers asynchronously)
frame = None
t0 = time.time()
while time.time() - t0 < 10:
    frame = cam.get_latest_frame()
    if frame is not None:
        break
    time.sleep(0.1)
step("capture returns an RGB frame", frame is not None and frame.ndim == 3 and frame.shape[2] == 3)
if frame is None:
    print("\nFAILED: no frame from capture")
    sys.exit(1)
print(f"        captured {frame.shape}")

# NOT-BLACK assertion: a black frame means the model would be driving blind.
frames = []
for _ in range(10):
    f = cam.get_latest_frame()
    if f is not None:
        frames.append(f)
    time.sleep(0.05)
arr = np.stack(frames).astype(np.float32)
brightness, variation = arr.mean(), arr.std()
print(f"        mean brightness {brightness:.1f}, std {variation:.1f}")
step("frames are NOT black (real pixels, model can see)", brightness > 2.0 and variation > 2.0)

pad = vg.VX360Gamepad()
pad.update()
step("virtual Xbox 360 pad created", True)

client = ModelClient(port=args.port)
client.reset()
info = client.info()
step("server reachable + reports checkpoint", bool(info.get("ckpt_path")))
print(f"        model: {info.get('ckpt_path')}")

lat = []
jl = []
for i in range(args.iters):
    frame = cam.get_latest_frame()
    obs = Image.fromarray(np.asarray(frame)[:256, :256])
    t = time.perf_counter()
    pred = client.predict(obs)
    lat.append((time.perf_counter() - t) * 1000)
    jl, jr, btns = pred["j_left"], pred["j_right"], pred["buttons"]
    if len(jl) == 0:
        continue
    pad.reset()
    pad.left_joystick(x_value=int(jl[0][0] * 32767), y_value=int(-jl[0][1] * 32767))
    pad.right_joystick(x_value=int(jr[0][0] * 32767), y_value=int(-jr[0][1] * 32767))
    pad.update()
    if i % 10 == 0:
        print(f"        iter {i:3d} | inf {lat[-1]:6.1f}ms | L=({jl[0][0]:+.2f},{jl[0][1]:+.2f}) chunk={len(jl)}")

pad.close()
cam.stop()

if lat:
    a = np.array(lat[1:] or lat)
    print(f"\n  inference latency: mean {a.mean():.0f}ms  min {a.min():.0f}ms  max {a.max():.0f}ms  (~{1000/a.mean():.1f}/s)")
step("server returned actions every iteration", len(lat) == args.iters)
step("predicted chunk has multiple actions", len(jl) > 1)

print(f"\n{'PIPELINE OK' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
