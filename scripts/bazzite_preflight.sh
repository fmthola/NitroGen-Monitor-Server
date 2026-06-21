#!/usr/bin/env bash
# Bazzite preflight for the NitroGen Linux play stack.
#
# Checks the host has what the Linux capture + virtual-gamepad path needs, and
# prints the exact commands to run the dual-GPU setup (game on the Arc A770,
# inference on the RTX 3070). It only inspects the system; it changes nothing.
set -u

ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; }

echo "NitroGen Bazzite preflight"
echo "=========================="

echo "GPUs:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L | sed 's/^/  /'
  ok "NVIDIA driver present (inference GPU)"
else
  bad "nvidia-smi not found — inference needs the NVIDIA GPU + CUDA"
fi
if lspci 2>/dev/null | grep -qiE 'Intel.*(Arc|DG2)'; then
  ok "Intel Arc present (can render the game, freeing the RTX for inference)"
else
  warn "No Intel Arc detected — the game and inference will share a GPU"
fi

echo "Virtual gamepad (uinput):"
if [ -w /dev/uinput ]; then
  ok "/dev/uinput is writable — virtual Xbox 360 pad will work without root"
else
  bad "/dev/uinput not writable — add a udev rule or the 'input' group"
fi

echo "Screen capture / window lookup:"
if [ -n "${WAYLAND_DISPLAY:-}" ]; then
  warn "Wayland session — run the game in a desktop/borderless window (XWayland) so mss can grab it; gamescope/Game-Mode output is not captured by region"
fi
[ -n "${DISPLAY:-}" ] && ok "DISPLAY=$DISPLAY (X11/XWayland reachable)" || bad "no DISPLAY — capture/window lookup need an X server"
command -v xdotool >/dev/null 2>&1 && ok "xdotool present (window lookup)" || warn "xdotool missing — pass --region instead of --window-name"

echo "Python deps (run inside your distrobox/venv):"
python3 - <<'PY' 2>/dev/null || true
mods = ["torch", "mss", "evdev", "Xlib", "cv2", "zmq", "numpy"]
for m in mods:
    try:
        __import__(m); print(f"  [ ok ] import {m}")
    except Exception as ex:
        print(f"  [warn] import {m} failed: {ex.__class__.__name__}")
PY

cat <<'NEXT'

Next steps (3 terminals)
------------------------
0) Launch Cyberpunk 2077 via Steam in a DESKTOP window (not Game Mode), ideally
   on the Arc A770:  Steam launch options ->  DRI_PRIME=1 %command%
   and set a borderless window so the whole viewport is on screen.

1) Model server (distrobox with CUDA + the RTX 3070):
     pip install -e ".[serve]"
     huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models
     CUDA_VISIBLE_DEVICES=0 python scripts/serve.py models/ng.pt --port 5555 --timesteps 2

2) Monitor (optional):
     python scripts/monitor.py --port 5556

3) Agent (host or a box with /dev/uinput + DISPLAY):
     pip install -e ".[play-linux]"
     python scripts/play_linux.py --window-name Cyberpunk --fps 15 --actions-per-step 8
   or capture an explicit region:
     python scripts/play_linux.py --region 0,0,1920,1080 --fps 15
NEXT
