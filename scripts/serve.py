"""
NitroGen Model Inference Server
================================

This server runs NVIDIA's NitroGen 500M parameter gaming AI model and serves
predictions via ZeroMQ. It receives game screenshots and returns controller actions.

WHY THIS PROJECT EXISTS:
------------------------
This is a feasibility study to determine if consumer GPUs (like RTX 3070 with 8GB VRAM)
can run the NitroGen gaming AI in real-time. The goal is to explore dual-GPU setups where:
  - GPU 1: Runs the game
  - GPU 2: Runs AI inference (this server)

INSTALLATION:
-------------
1. Install ViGEmBus driver: https://github.com/nefarius/ViGEmBus/releases
2. Clone the repository and create virtual environment:

   git clone https://github.com/fmthola/NitroGen-Monitor-Server.git
   cd NitroGen-Monitor-Server
   python -m venv venv
   .\venv\Scripts\activate
   pip install -e ".[serve,play]"

3. Download the model (~2GB):

   huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models

USAGE:
------
Start the server:
    python scripts/serve.py models/ng.pt --port 5555 --timesteps 2

Then in another terminal, start the game agent:
    python scripts/play_simple.py --process "YourGame.exe" --monitor-port 5556

And optionally the monitor:
    python scripts/monitor.py --port 5556

PERFORMANCE (RTX 3070):
-----------------------
  --timesteps 16 (default): ~450ms/frame, ~2 FPS  (too slow)
  --timesteps 8:            ~265ms/frame, ~4 FPS  (sluggish)
  --timesteps 4:            ~160ms/frame, ~6 FPS  (slow games)
  --timesteps 2:            ~85ms/frame,  ~12 FPS (RECOMMENDED)
  --timesteps 1:            ~70ms/frame,  ~14 FPS (outputs too weak)

CREDITS:
--------
Original NitroGen by NVIDIA: https://github.com/MineDojo/NitroGen
Paper: https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf
Model: https://huggingface.co/nvidia/NitroGen
"""

import zmq
import argparse
import pickle
import torch

from nitrogen.inference_session import InferenceSession

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference server")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--ctx", type=int, default=1, help="Context length")
    parser.add_argument("--timesteps", type=int, default=16, help="Inference timesteps (16=default, 8=faster, 4=fastest)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster inference")
    args = parser.parse_args()

    # Enable TF32 matmuls on Ampere+ (e.g. RTX 3070). Free latency win for the
    # DiT/attention layers with no measurable quality cost; upstream leaves it off.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    session = InferenceSession.from_ckpt(args.ckpt, old_layout=args.old_layout, cfg_scale=args.cfg, context_length=args.ctx)

    # Optimize: reduce inference timesteps for faster speed
    if args.timesteps != 16:
        print(f"Setting inference timesteps: {args.timesteps} (default: 16)")
        session.model.num_inference_timesteps = args.timesteps

    # Optimize: compile model for faster inference
    if args.compile:
        print("Compiling model with torch.compile (first inference will be slow)...")
        session.model = torch.compile(session.model, mode="reduce-overhead")

    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Create poller
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"\n{'='*60}")
    print(f"Server running on port {args.port}")
    print("Waiting for requests...")
    print(f"{'='*60}\n")

    try:
        while True:
            # Poll with 100ms timeout to allow interrupt handling
            events = dict(poller.poll(timeout=100))
            if socket in events and events[socket] == zmq.POLLIN:
                # Receive request only when data is available
                request = socket.recv()
                request = pickle.loads(request)
                if request["type"] == "reset":
                    session.reset()
                    response = {"status": "ok"}
                    print("Session reset")
                elif request["type"] == "info":
                    info = session.info()
                    response = {"status": "ok", "info": info}
                    print("Sent session info")
                elif request["type"] == "predict":
                    raw_image = request["image"]
                    result = session.predict(raw_image)
                    response = {
                        "status": "ok",
                        "pred": result
                    }
                else:
                    response = {"status": "error", "message": f"Unknown request type: {request['type']}"}
                # Send response
                socket.send(pickle.dumps(response))
    except KeyboardInterrupt:
        print("\nShutting down server...")
        exit(0)
    finally:
        socket.close()
        context.term()
