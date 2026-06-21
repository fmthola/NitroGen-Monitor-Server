#!/usr/bin/env bash
# Run the NitroGen model server on the RTX 3070 on Bazzite, via plain podman.
#
# Why not distrobox? On this Bazzite build, podman 5.8 cannot `statfs` bind
# sources on the composefs-backed /usr, so `distrobox create` (which bind-mounts
# /usr/bin/distrobox-init) and the usual `--nvidia` driver-lib mounts both fail
# with "statfs ...: no such file or directory". The workaround here:
#   * run plain podman from a python:3.12-slim image (Python+pip already inside),
#   * pass the GPU through as DEVICE NODES (not /usr mounts),
#   * stage the NVIDIA userspace driver libs onto /var (where mounts DO work) and
#     point LD_LIBRARY_PATH at them,
#   * mount the project from /var/home (also works).
#
# Usage:  scripts/bazzite_server.sh {setup|start|stop|logs|shell|status}
set -euo pipefail

NAME=nitrogen-srv
IMAGE=docker.io/library/python:3.12-slim
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/var/home/$USER/nitrogen-nvidia
PORT="${PORT:-5555}"
TIMESTEPS="${TIMESTEPS:-2}"

stage_libs() {
  mkdir -p "$STAGE"
  cp -af /usr/lib64/libcuda.so* /usr/lib64/libnvidia-ml.so* "$STAGE"/ 2>/dev/null || true
  for p in /usr/lib64/libnvidia-ptxjitcompiler.so* /usr/lib64/libnvidia-nvvm.so* \
           /usr/lib64/libnvidia-allocator.so*; do
    [ -e "$p" ] && cp -af "$p" "$STAGE"/ 2>/dev/null || true
  done
  echo "Staged NVIDIA libs -> $STAGE"
}

setup() {
  command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver"; exit 1; }
  stage_libs
  local devs=""
  for d in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    [ -e "$d" ] && devs="$devs --device $d"
  done
  podman rm -f "$NAME" 2>/dev/null || true
  podman pull -q "$IMAGE"
  # shellcheck disable=SC2086
  podman run -d --name "$NAME" --network host $devs \
    -v "$REPO":/workspace:rw \
    -v "$STAGE":/nvidia-libs:ro \
    -e LD_LIBRARY_PATH=/nvidia-libs \
    -w /workspace "$IMAGE" sleep infinity
  echo "Installing serve deps + CUDA torch (one-time, multi-GB)..."
  podman exec "$NAME" bash -lc 'pip install -q --no-input -e ".[serve]"'
  podman exec "$NAME" python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA not visible"; print("CUDA OK:", torch.cuda.get_device_name(0))'
  echo "Setup complete. Put the model at models/ng.pt, then: $0 start"
}

start() {
  [ -f "$REPO/models/ng.pt" ] || { echo "missing models/ng.pt (huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models)"; exit 1; }
  podman exec -d "$NAME" bash -lc \
    "cd /workspace && python3 scripts/serve.py models/ng.pt --port $PORT --timesteps $TIMESTEPS > /workspace/.server.log 2>&1"
  echo "Server starting on port $PORT (timesteps=$TIMESTEPS). Tail: $0 logs"
  echo "NOTE: the RTX needs ~2GB free VRAM — move the desktop/game to the Arc A770 and"
  echo "      free other GPU users (e.g. 'ollama stop <model>') if you hit CUDA OOM."
}

case "${1:-}" in
  setup)  setup ;;
  start)  start ;;
  stop)   podman exec "$NAME" pkill -f serve.py 2>/dev/null || true; echo "server stopped" ;;
  logs)   podman exec "$NAME" tail -f /workspace/.server.log ;;
  shell)  podman exec -it "$NAME" bash ;;
  status) podman ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'; nvidia-smi --query-gpu=memory.used,memory.free --format=csv ;;
  *) echo "usage: $0 {setup|start|stop|logs|shell|status}"; exit 1 ;;
esac
