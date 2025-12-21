# NitroGen Monitor & Server

A performance-optimized implementation of NVIDIA NitroGen gaming AI agent with real-time monitoring dashboard.

> **EXPERIMENTAL PROJECT** - This is an experimental exploration of running gaming AI on consumer hardware. Expect limitations and ongoing development.

![NitroGen Monitor in Action](screenshot.png)

---

## The Vision: Dual-GPU Gaming AI

This project explores the **feasibility of running gaming AI on consumer hardware** using a dual-GPU configuration:

- **GPU 1 (Intel ARC)**: Dedicated to running the game at full settings, preserving all VRAM for textures and effects
- **GPU 2 (NVIDIA RTX)**: Dedicated to AI inference, running the NitroGen model without impacting game performance

**Why Dual-GPU?**
- Keep your gaming GPU full VRAM available for the game
- Dedicated AI processing without resource contention
- Future-proof setup as AI models grow larger
- Cleaner separation of concerns

### Future Goals

This project is actively evolving:
- **Game-Specific Training**: Fine-tune the model for specific games
- **Task Guidance**: Guide the AI to perform specific in-game tasks
- **Multi-Game Profiles**: Train and switch between different game configurations
- **Improved Performance**: Optimize inference for faster response times

---

## Current Findings (RTX 3070)

| Configuration | Inference Time | AI FPS | Verdict |
|--------------|----------------|--------|---------|
| 16 timesteps (default) | ~450ms | ~2 FPS | Too slow |
| 8 timesteps | ~265ms | ~4 FPS | Sluggish |
| 4 timesteps | ~160ms | ~6 FPS | Playable for slow games |
| **2 timesteps** | 62-85ms | **~12-15 FPS** | **Best balance** |

**Observed Performance** (Cyberpunk 2077 testing):
- **Peak**: 15.4 FPS (62ms inference)
- **Average**: 11-14 FPS (70-90ms inference)
- **Minimum**: ~8 FPS under heavy GPU load

**Conclusion**: An RTX 3070 can run NitroGen at 12-15 FPS with 2 timesteps. Further optimization may be possible. Sufficient for slower-paced and turn-based games.

---

## What is NitroGen?

**NitroGen** is NVIDIA open-source foundation model for generalist gaming agents:
- **500M parameter** vision-action transformer
- Takes **256x256 game screenshots** as input
- Outputs **gamepad controller actions** (joysticks + 17 buttons)
- Trained on **40,000 hours of gameplay** across **1,000+ games**

---

## This Repository Provides

| Component | Description |
|-----------|-------------|
| scripts/serve.py | Optimized inference server with configurable timesteps |
| scripts/play_simple.py | Game controller without DLL injection (anti-cheat safe) |
| scripts/play_interactive.py | Human-in-the-loop mode with correction recording |
| scripts/train_dagger.py | DAgger training on human corrections |
| scripts/monitor.py | Real-time dashboard showing AI vision and decisions |
| scripts/record_training_data.py | Record gameplay for training |
| setup.ps1 | Automated setup script with environment validation |
| run.ps1 | Easy-to-use launcher script |

---

## Requirements

- **Windows 10/11** (required for DirectX capture)
- **NVIDIA GPU** with 4GB+ VRAM
- **Python 3.10+**
- **CUDA 11.8+** with PyTorch
- **ViGEmBus Driver** - [Download here](https://github.com/nefarius/ViGEmBus/releases)

---

## Quick Start

### Option 1: Automated Setup (Recommended)

    # Clone the repository
    git clone https://github.com/fmthola/NitroGen-Monitor-Server.git
    cd NitroGen-Monitor-Server

    # Run setup (validates environment, installs dependencies)
    .\setup.ps1

    # Download the model (~2GB)
    huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models

    # Start your game first, then run:
    .\run.ps1 -GameProcess "YourGame.exe"

### Option 2: Manual Setup

#### Step 1: Install ViGEmBus Driver
Download and install from: https://github.com/nefarius/ViGEmBus/releases

#### Step 2: Clone and Setup Environment

    git clone https://github.com/fmthola/NitroGen-Monitor-Server.git
    cd NitroGen-Monitor-Server
    python -m venv venv
    .\venv\Scripts\activate
    pip install --upgrade pip
    pip install -e ".[serve,play]"

#### Step 3: Download Model (~2GB)

    pip install huggingface_hub
    huggingface-cli download nvidia/NitroGen ng.pt --local-dir ./models

---

## Running the AI (Manual Method)

**Important: Follow this order!**

### Step 0: Start Your Game
Launch your game and get into gameplay (not menus).

### Step 1: Start Model Server (Terminal 1)

    .\venv\Scripts\activate
    python scripts/serve.py models/ng.pt --port 5555 --timesteps 2

Wait for "Server running on port 5555" (~30-60 seconds).

### Step 2: Start Monitor (Terminal 2)

    .\venv\Scripts\activate
    python scripts/monitor.py --port 5556

### Step 3: Start Game Agent (Terminal 3)

    .\venv\Scripts\activate
    python scripts/play_simple.py --process "YourGame.exe" --fps 60 --monitor-port 5556

After a 3-second countdown, the AI takes control!

---

## Server Configuration

    python scripts/serve.py <checkpoint> [options]

    Options:
      --port PORT         Server port (default: 5555)
      --timesteps N       Diffusion steps (default: 16, recommend: 2)
      --cfg SCALE         CFG scale (default: 1.0)
      --ctx LENGTH        Context length (default: 1)

---

## Monitor Dashboard

The monitor shows real-time AI state:
- **AI Vision**: 256x256 processed frame (what the model sees)
- **Controller State**: Joystick positions and button presses
- **Action Log**: Timestamped decisions
- **Performance**: Latency and FPS metrics

---

## Troubleshooting

### No CUDA GPUs available

    nvidia-smi  # Should show your GPU
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

### Process not found
- Ensure game is running before starting play_simple.py
- Check exact process name in Task Manager

### Low FPS
- Use --timesteps 2 (fastest stable setting)
- Lower game resolution
- Close other GPU applications

---

## Credits and Resources

### NVIDIA NitroGen (Original Project)

This project is built upon NVIDIA NitroGen:

- **GitHub**: https://github.com/MineDojo/NitroGen
- **Model**: https://huggingface.co/nvidia/NitroGen
- **Dataset**: https://huggingface.co/datasets/nvidia/NitroGen
- **Paper**: https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf
- **Website**: https://nitrogen.minedojo.org/

### Dependencies

- [ViGEmBus](https://github.com/nefarius/ViGEmBus) - Virtual gamepad driver
- [vgamepad](https://github.com/yannbouteiller/vgamepad) - Python gamepad library
- [dxcam](https://github.com/ra1nty/DXcam) - DirectX screen capture
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## License

This project builds upon NVIDIA NitroGen. See the [original repository](https://github.com/MineDojo/NitroGen) for license details.

---

## Interactive Mode & DAgger Training

This repository includes a **human-in-the-loop** training system. Correct the AI in real-time, and those corrections fine-tune the model.

### How It Works



### Step 1: Play with AI + Provide Corrections

    python scripts/play_interactive.py --process "YourGame.exe"

- AI plays normally
- Touch your controller to override
- Corrections saved automatically

**Hotkeys:** ALT+F1=AI only, ALT+F2=Human only, ALT+F3=Auto mode

### Step 2: Train on Corrections

    python scripts/train_dagger.py --corrections ./corrections --epochs 5

Outputs detailed analysis:
- What actions AI got wrong most often
- Which buttons needed most correction
- Joystick direction errors
- Training loss curves

### Step 3: Test Improved Model

    python scripts/serve.py models/ng_dagger.pt --timesteps 2
    python scripts/play_simple.py --process "YourGame.exe"

---

## Training Documentation

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for comprehensive fine-tuning guide.

---

## Contributing

This is an experimental project. Contributions welcome in these areas:
- Performance optimizations
- Multi-GPU support improvements
- Game-specific fine-tuning guides
- Task guidance system development
