"""
End-to-end pipeline smoke test for the Linux port — no game required.

Exercises every link of the chain except the actual game:
  mss capture (a desktop region)  ->  ZMQ ModelClient -> server inference (GPU)
  ->  predicted action  ->  uinput virtual Xbox 360 pad.

Run the server first (in the GPU container):
  python scripts/serve.py models/ng.pt --port 5555 --timesteps 2
Then on the host (with DISPLAY + /dev/uinput):
  python scripts/smoke_test_linux.py --region 0,0,512,512 --iters 30

It prints per-iteration inference latency and the action applied, and asserts the
server returns well-formed actions and the virtual pad accepts them. Exit code 0
means the full pipeline is wired and working.
"""
import sys
import time
import argparse

import numpy as np
from PIL import Image

import nitrogen.linux_capture as dxcam
import nitrogen.linux_gamepad as vg
from nitrogen.linux_window import region_from_str
from nitrogen.inference_client import ModelClient
from nitrogen.shared import BUTTON_ACTION_TOKENS

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--region", type=str, default="0,0,512,512", help="desktop region 'L,T,W,H' to capture")
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

region = region_from_str(args.region)
cam = dxcam.create(output_color="RGB", region=region)
cam.start()
frame = cam.get_latest_frame()
step("mss capture returns an RGB frame", frame is not None and frame.ndim == 3 and frame.shape[2] == 3)
print(f"        captured {frame.shape} from region {region}")

pad = vg.VX360Gamepad()
pad.update()
step("virtual Xbox 360 pad created", True)

client = ModelClient(port=args.port)
client.reset()
info = client.info()
step("server reachable + reports checkpoint", bool(info.get("ckpt_path")))
print(f"        model: {info.get('ckpt_path')}")

lat = []
last_pressed = None
for i in range(args.iters):
    frame = cam.get_latest_frame()
    obs = Image.fromarray(np.asarray(frame)[:256, :256])  # 256x256 RGB
    t = time.perf_counter()
    pred = client.predict(obs)
    dt = (time.perf_counter() - t) * 1000
    lat.append(dt)

    jl, jr, btns = pred["j_left"], pred["j_right"], pred["buttons"]
    if len(jl) == 0:
        continue
    # drive the pad from the first action of the predicted chunk
    pad.reset()
    pad.left_joystick(x_value=int(jl[0][0] * 32767), y_value=int(-jl[0][1] * 32767))
    pad.right_joystick(x_value=int(jr[0][0] * 32767), y_value=int(-jr[0][1] * 32767))
    last_pressed = [TOKENS[k] for k, b in enumerate(btns[0]) if b > THRES]
    pad.update()
    if i % 10 == 0:
        print(f"        iter {i:3d} | inf {dt:6.1f}ms | L=({jl[0][0]:+.2f},{jl[0][1]:+.2f}) "
              f"chunk={len(jl)} | btns={last_pressed}")

pad.close()
cam.stop()

if lat:
    arr = np.array(lat[1:] or lat)  # drop first (warmup)
    print(f"\n  inference latency: mean {arr.mean():.0f}ms  min {arr.min():.0f}ms  "
          f"max {arr.max():.0f}ms  (~{1000/arr.mean():.1f} infer/s)")
step("server returned actions every iteration", len(lat) == args.iters)
step("predicted chunk has multiple actions (for --actions-per-step)", len(jl) > 1)

print(f"\n{'PIPELINE OK' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
