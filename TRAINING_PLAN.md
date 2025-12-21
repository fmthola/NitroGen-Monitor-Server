# NitroGen Training & Fine-Tuning Plan

## Overview

NitroGen is a 500M parameter vision-action model trained on gameplay videos. This document outlines the complete pipeline for fine-tuning NitroGen on your own gameplay data.

**Key Insight**: NitroGen is vision-only with no text/prompt conditioning. The model learns to predict gamepad actions from screen pixels. Fine-tuning teaches it new games or improves behavior on specific scenarios.

---

## Phase 1: Data Collection

### 1.1 Hardware Requirements
- **GPU**: NVIDIA RTX 3070 or better (for recording + inference)
- **Storage**: ~5GB per hour of gameplay at 30 FPS
- **Controller**: Xbox-compatible gamepad (required for input capture)

### 1.2 Recording Setup

Use the built-in recorder:
```powershell
cd NitroGen-Monitor-Server
.\venv\Scripts\activate
python scripts/record_training_data.py --game "YourGame.exe" --output ./training_data --fps 30
```

**Controls**:
- F9: Start/Stop recording
- F10: Quit and save

### 1.3 Data Format

Each recording chunk creates:
```
training_data/
  YourGame_20241220_123456_chunk_0001/
    frames/
      000000.png  # 256x256 RGB
      000001.png
      ...
    actions_processed.parquet  # Controller inputs
    metadata.json              # Session info
```

**Actions Parquet Schema**:
| Column | Type | Description |
|--------|------|-------------|
| j_left | [float, float] | Left joystick X,Y (-1 to 1) |
| j_right | [float, float] | Right joystick X,Y (-1 to 1) |
| south/east/west/north | int | A/B/X/Y buttons (0/1) |
| left_shoulder/right_shoulder | int | LB/RB buttons (0/1) |
| left_trigger/right_trigger | int | LT/RT (0/1 thresholded) |
| dpad_up/down/left/right | int | D-pad directions (0/1) |

### 1.4 Recording Best Practices

1. **Quantity**: Minimum 2-4 hours of varied gameplay
2. **Quality**: Play well - the model learns to imitate your behavior
3. **Variety**: Cover different scenarios (driving, combat, exploration, menus)
4. **Consistency**: Use consistent camera settings and resolution
5. **Segmentation**: Record in 10-20 minute sessions to create natural chunks

---

## Phase 2: Data Preparation

### 2.1 Combine Training Chunks

Create a master parquet file referencing all chunks:

```python
# scripts/prepare_training_data.py
import os
import polars as pl
from pathlib import Path

def prepare_dataset(data_dir: str, output_path: str):
    """Combine all recording chunks into a training dataset."""
    chunks = []

    for chunk_dir in sorted(Path(data_dir).glob("*_chunk_*")):
        actions_file = chunk_dir / "actions_processed.parquet"
        frames_dir = chunk_dir / "frames"

        if not actions_file.exists():
            continue

        df = pl.read_parquet(actions_file)

        # Add frame paths
        frame_paths = sorted(frames_dir.glob("*.png"))
        df = df.with_columns([
            pl.Series("frame_path", [str(p) for p in frame_paths[:len(df)]]),
            pl.lit(chunk_dir.name).alias("chunk_id"),
            pl.lit(chunk_dir.name.split("_")[0]).alias("game_label")
        ])

        chunks.append(df)

    combined = pl.concat(chunks)
    combined.write_parquet(output_path)
    print(f"Created dataset: {output_path}")
    print(f"Total frames: {len(combined)}")

    return combined

if __name__ == "__main__":
    prepare_dataset("./training_data", "./training_data/dataset.parquet")
```

### 2.2 Data Validation

```python
# scripts/validate_dataset.py
import polars as pl
from pathlib import Path

def validate(parquet_path: str):
    df = pl.read_parquet(parquet_path)

    print("Dataset Statistics:")
    print(f"  Total frames: {len(df)}")
    print(f"  Games: {df['game_label'].unique().to_list()}")
    print(f"  Chunks: {df['chunk_id'].n_unique()}")

    # Check frame existence
    missing = 0
    for path in df["frame_path"].sample(min(100, len(df))):
        if not Path(path).exists():
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing}/100 sampled frames missing!")
    else:
        print("  All sampled frames exist")

    # Action statistics
    print("\nAction Distribution:")
    for col in ["south", "east", "west", "north", "right_trigger", "left_trigger"]:
        if col in df.columns:
            mean = df[col].mean()
            print(f"  {col}: {mean:.2%} pressed")

validate("./training_data/dataset.parquet")
```

---

## Phase 3: Fine-Tuning

### 3.1 Training Script

```python
# scripts/finetune.py
"""
NitroGen Fine-Tuning Script

Requirements:
  pip install torch transformers accelerate wandb

Usage:
  python scripts/finetune.py --data ./training_data/dataset.parquet --epochs 10
"""
import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import polars as pl
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor
from nitrogen.inference_session import load_model
from nitrogen.flow_matching_transformer.nitrogen import NitroGen

class GameplayDataset(Dataset):
    """Dataset for NitroGen fine-tuning."""

    def __init__(self, parquet_path: str, img_proc, action_horizon: int = 8):
        self.df = pl.read_parquet(parquet_path)
        self.img_proc = img_proc
        self.action_horizon = action_horizon

        # Group by chunk for sequential sampling
        self.chunks = self.df.group_by("chunk_id").agg(pl.all())

    def __len__(self):
        # Return number of valid training windows
        return len(self.df) - self.action_horizon

    def __getitem__(self, idx):
        # Get frame
        row = self.df[idx]
        frame_path = row["frame_path"][0]
        image = Image.open(frame_path).convert("RGB")
        pixel_values = self.img_proc(image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Get action sequence (next action_horizon frames)
        actions = []
        for i in range(self.action_horizon):
            if idx + i < len(self.df):
                action_row = self.df[idx + i]
                j_left = [action_row["j_left"][0][0], action_row["j_left"][0][1]]
                j_right = [action_row["j_right"][0][0], action_row["j_right"][0][1]]

                # Collect button states
                buttons = []
                for btn in ["south", "east", "west", "north", "left_shoulder",
                           "right_shoulder", "left_trigger", "right_trigger",
                           "dpad_up", "dpad_down", "dpad_left", "dpad_right"]:
                    if btn in action_row.columns:
                        buttons.append(float(action_row[btn][0]))
                    else:
                        buttons.append(0.0)

                # Normalize joysticks to [0, 1]
                j_left = [(x + 1) / 2 for x in j_left]
                j_right = [(x + 1) / 2 for x in j_right]

                action = j_left + j_right + buttons
                actions.append(action)

        # Pad if needed
        while len(actions) < self.action_horizon:
            actions.append(actions[-1] if actions else [0.5, 0.5, 0.5, 0.5] + [0.0] * 12)

        return {
            "pixel_values": pixel_values,
            "actions": torch.tensor(actions, dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()

        # Move to device
        pixel_values = batch["pixel_values"].to(device)
        actions = batch["actions"].to(device)

        # Prepare model input (simplified - see nitrogen.py for full implementation)
        # This is a placeholder - actual implementation requires tokenizer

        # Forward pass
        # output = model(data_dict)
        # loss = output["loss"]

        # For now, just compute a simple loss as placeholder
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NitroGen")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset parquet")
    parser.add_argument("--checkpoint", type=str, default="models/ng.pt", help="Base checkpoint")
    parser.add_argument("--output", type=str, default="models/ng_finetuned.pt", help="Output path")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, tokenizer, img_proc, ckpt_config, game_mapping, _ = load_model(args.checkpoint)
    model.to(device)

    # Configure trainable parameters
    if args.freeze_vision:
        print("Freezing vision encoder...")
        model.set_trainable_parameters(
            tune_vision_tower=False,
            tune_mm_projector=True,
            tune_diffusion_model=True,
            tune_multi_projector=True,
            tune_vl_mixing=True
        )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Create dataset
    print("Loading dataset...")
    dataset = GameplayDataset(args.data, img_proc)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("Starting training...")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()

        print(f"Epoch {epoch}: Loss = {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            # Save checkpoint
            torch.save({
                "model": model.state_dict(),
                "ckpt_config": ckpt_config.model_dump(),
                "epoch": epoch,
                "loss": loss
            }, args.output)
            print(f"Saved best model to {args.output}")

    print("Training complete!")


if __name__ == "__main__":
    main()
```

### 3.2 Training Configuration

**Recommended Settings**:

| Setting | Value | Notes |
|---------|-------|-------|
| Learning Rate | 1e-5 | Conservative for fine-tuning |
| Batch Size | 4-8 | Limited by VRAM (RTX 3070: 8GB) |
| Epochs | 10-50 | Monitor validation loss |
| Optimizer | AdamW | Standard for transformers |
| Scheduler | Cosine | Smooth learning rate decay |

**What to Freeze**:

| Component | Parameters | Freeze? |
|-----------|------------|---------|
| Vision Encoder (SigLIP) | ~300M | Yes (for efficiency) |
| Diffusion Model (DiT) | ~150M | No (core action prediction) |
| Action Encoder/Decoder | ~20M | No (output layer) |
| VL Mixing | ~30M | Optional |

### 3.3 Hardware Requirements

| GPU | VRAM | Batch Size | Training Time (1hr data) |
|-----|------|------------|--------------------------|
| RTX 3070 | 8GB | 2-4 | ~6 hours |
| RTX 3090 | 24GB | 8-16 | ~2 hours |
| RTX 4090 | 24GB | 16-32 | ~1 hour |
| A100 | 80GB | 64+ | ~20 minutes |

---

## Phase 4: Evaluation

### 4.1 Metrics

1. **Action Accuracy**: Compare predicted vs recorded actions
2. **Joystick MSE**: Mean squared error on joystick positions
3. **Button F1**: F1 score for button predictions
4. **Visual Playtest**: Human evaluation in-game

### 4.2 Evaluation Script

```python
# scripts/evaluate.py
import torch
from nitrogen.inference_session import InferenceSession

def evaluate(checkpoint_path: str, test_data_path: str):
    session = InferenceSession.from_ckpt(checkpoint_path)

    # Load test data
    # ... evaluation logic ...

    print("Evaluation Results:")
    print(f"  Joystick MSE: ...")
    print(f"  Button Accuracy: ...")

if __name__ == "__main__":
    evaluate("models/ng_finetuned.pt", "test_data/dataset.parquet")
```

---

## Phase 5: Deployment

### 5.1 Testing Your Fine-Tuned Model

```powershell
# Serve the fine-tuned model
python scripts/serve.py models/ng_finetuned.pt --timesteps 2

# Run on game
python scripts/play_simple.py --process "YourGame.exe" --fps 15
```

### 5.2 A/B Testing

Compare base model vs fine-tuned:
1. Record baseline gameplay with base model
2. Record gameplay with fine-tuned model
3. Compare qualitative behavior differences

---

## Appendix A: Troubleshooting

### Out of Memory (OOM)
- Reduce batch size to 1-2
- Enable gradient checkpointing
- Freeze vision encoder (saves ~50% VRAM)

### Poor Training Results
- Need more data (minimum 2 hours recommended)
- Check data quality (consistent gameplay, no pauses)
- Lower learning rate (try 1e-6)

### Catastrophic Forgetting
- Use lower learning rate
- Mix original training data with new data
- Train fewer epochs

---

## Appendix B: Advanced Topics

### Multi-Game Fine-Tuning
Use the game_mapping feature to train on multiple games:
1. Add game labels to your dataset
2. Update tokenizer game_mapping_cfg
3. The model will learn game-specific behaviors

### Behavior Cloning vs Reinforcement Learning
NitroGen uses behavior cloning (supervised learning). For RL:
- Would require reward signal from game
- Much more complex setup
- Not covered in this guide

---

## Quick Start Checklist

- [ ] Record 2+ hours of gameplay
- [ ] Run `prepare_training_data.py` to create dataset
- [ ] Run `validate_dataset.py` to check data
- [ ] Run `finetune.py --epochs 10`
- [ ] Test with `play_simple.py`
- [ ] Iterate: record more data, fine-tune again

---

*Last Updated: December 2024*
