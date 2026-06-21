"""
DAgger Training Script for NitroGen with Detailed Analytics

Uses corrections collected with play_interactive.py to fine-tune the model.
Outputs comprehensive analysis of what the AI learned.

USAGE:
  python scripts/train_dagger.py --corrections ./corrections --epochs 5

OUTPUT:
  - Training loss curves
  - Correction analysis (what AI got wrong)
  - Button/joystick error breakdown
  - Before/after comparison
  - Saved report in corrections directory
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


# =============================================================================
# CORRECTION ANALYSIS
# =============================================================================
class CorrectionAnalyzer:
    """Analyzes human corrections to understand AI mistakes."""

    def __init__(self):
        self.corrections = []
        self.button_errors = defaultdict(int)  # button -> count of times human pressed when AI didn't
        self.button_overcalls = defaultdict(int)  # button -> count of times AI pressed when human didn't
        self.joystick_errors = {
            'left_x': [], 'left_y': [],
            'right_x': [], 'right_y': []
        }
        self.total_corrections = 0

    def _record_joystick_diff(self, human_action: dict, ai_action: dict):
        """Accumulate the per-axis human-vs-AI joystick differences."""
        h_jl = human_action.get('j_left', [0, 0])
        h_jr = human_action.get('j_right', [0, 0])

        a_jl = ai_action.get('j_left', [0, 0])
        a_jr = ai_action.get('j_right', [0, 0])

        # Handle numpy arrays
        if hasattr(a_jl, 'tolist'):
            a_jl = a_jl.tolist()
        if hasattr(a_jr, 'tolist'):
            a_jr = a_jr.tolist()

        self.joystick_errors['left_x'].append(h_jl[0] - a_jl[0])
        self.joystick_errors['left_y'].append(h_jl[1] - a_jl[1])
        self.joystick_errors['right_x'].append(h_jr[0] - a_jr[0])
        self.joystick_errors['right_y'].append(h_jr[1] - a_jr[1])

    @staticmethod
    def _ai_buttons_to_dict(a_buttons):
        """Convert a list/array of AI button values into a name->0/1 dict."""
        if isinstance(a_buttons, (list, np.ndarray)):
            button_names = [
                "dpad_down", "dpad_left", "dpad_right", "dpad_up",
                "east", "left_shoulder", "left_thumb", "left_trigger",
                "north", "right_shoulder", "right_thumb", "right_trigger",
                "south", "west"
            ]
            a_buttons_dict = {}
            for i, name in enumerate(button_names):
                if i < len(a_buttons):
                    a_buttons_dict[name] = 1 if a_buttons[i] > 0.5 else 0
            return a_buttons_dict
        return a_buttons

    def _record_button_diff(self, human_action: dict, ai_action: dict):
        """Accumulate missed presses and overcalls for each button."""
        h_buttons = human_action.get('buttons', {})
        a_buttons = self._ai_buttons_to_dict(ai_action.get('buttons', {}))

        for btn_name in set(list(h_buttons.keys()) + list(a_buttons.keys())):
            h_pressed = h_buttons.get(btn_name, 0)
            a_pressed = a_buttons.get(btn_name, 0)

            if h_pressed and not a_pressed:
                self.button_errors[btn_name] += 1  # Human needed this, AI didn't press
            elif a_pressed and not h_pressed:
                self.button_overcalls[btn_name] += 1  # AI pressed, human didn't want

    def add_correction(self, human_action: dict, ai_action: dict):
        """Record a single correction for analysis."""
        self.total_corrections += 1
        self.corrections.append({
            'human': human_action,
            'ai': ai_action
        })

        self._record_joystick_diff(human_action, ai_action)
        self._record_button_diff(human_action, ai_action)

    def generate_report(self) -> dict:
        """Generate comprehensive analysis report."""
        if self.total_corrections == 0:
            return {"error": "No corrections to analyze"}

        report = {
            "summary": {
                "total_corrections": self.total_corrections,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "joystick_analysis": {},
            "button_analysis": {
                "missed_presses": {},
                "unwanted_presses": {},
            },
            "insights": [],
            "recommendations": [],
        }

        # Joystick analysis
        for axis, errors in self.joystick_errors.items():
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                max_error = np.max(np.abs(errors))

                if mean_error > 0.1:
                    bias = "positive"
                elif mean_error < -0.1:
                    bias = "negative"
                else:
                    bias = "neutral"

                report["joystick_analysis"][axis] = {
                    "mean_error": round(float(mean_error), 4),
                    "std_error": round(float(std_error), 4),
                    "max_error": round(float(max_error), 4),
                    "bias": bias
                }

        # Button analysis
        for btn, count in sorted(self.button_errors.items(), key=lambda x: -x[1]):
            pct = count / self.total_corrections * 100
            report["button_analysis"]["missed_presses"][btn] = {
                "count": count,
                "percentage": round(pct, 1)
            }

        for btn, count in sorted(self.button_overcalls.items(), key=lambda x: -x[1]):
            pct = count / self.total_corrections * 100
            report["button_analysis"]["unwanted_presses"][btn] = {
                "count": count,
                "percentage": round(pct, 1)
            }

        # Generate insights
        report["insights"] = self._generate_insights(report)
        report["recommendations"] = self._generate_recommendations(report)

        return report

    @staticmethod
    def _joystick_insights(report: dict) -> list:
        """Build insights describing systematic joystick bias."""
        insights = []
        for axis, data in report["joystick_analysis"].items():
            if abs(data["mean_error"]) > 0.15:
                direction = "right/up" if data["mean_error"] > 0 else "left/down"
                stick = "left" if "left" in axis else "right"
                coord = "X" if "_x" in axis else "Y"
                insights.append(
                    f"AI tends to under-steer {stick} stick {coord}-axis. "
                    f"Human corrections push {direction} (avg: {data['mean_error']:+.2f})"
                )
        return insights

    @staticmethod
    def _button_insights(report: dict) -> list:
        """Build insights describing missed and unwanted button presses."""
        insights = []
        missed = report["button_analysis"]["missed_presses"]
        if missed:
            top_missed = list(missed.keys())[:3]
            if top_missed:
                insights.append(
                    f"Most commonly missed buttons: {', '.join(top_missed)}. "
                    f"AI should press these more often."
                )

        unwanted = report["button_analysis"]["unwanted_presses"]
        if unwanted:
            top_unwanted = list(unwanted.keys())[:3]
            if top_unwanted:
                insights.append(
                    f"AI presses these too often: {', '.join(top_unwanted)}. "
                    f"Human rarely wants these in corrected frames."
                )
        return insights

    def _generate_insights(self, report: dict) -> list:
        """Generate human-readable insights."""
        insights = []
        insights.extend(self._joystick_insights(report))
        insights.extend(self._button_insights(report))

        if not insights:
            insights.append("No major systematic errors detected. Corrections may be situational.")

        return insights

    def _generate_recommendations(self, report: dict) -> list:
        """Generate training recommendations."""
        recommendations = []

        # Check if we have enough data
        if self.total_corrections < 100:
            recommendations.append(
                f"Collect more corrections ({self.total_corrections} is low). "
                f"Aim for 500+ for reliable training."
            )

        # Check for button imbalances
        missed = report["button_analysis"]["missed_presses"]
        for btn, data in missed.items():
            if data["percentage"] > 20:
                recommendations.append(
                    f"Consider adding more training data where '{btn}' should be pressed."
                )

        # Check joystick bias
        for axis, data in report["joystick_analysis"].items():
            if abs(data["mean_error"]) > 0.2:
                recommendations.append(
                    f"Strong {axis} bias detected. Training should help correct this."
                )

        if not recommendations:
            recommendations.append("Data looks balanced. Proceed with training.")

        return recommendations

    def print_report(self, report: dict):
        """Print formatted report to console."""
        print("\n" + "="*70)
        print("CORRECTION ANALYSIS REPORT")
        print("="*70)

        print(f"\nTotal Corrections Analyzed: {report['summary']['total_corrections']}")

        print("\n--- JOYSTICK ANALYSIS ---")
        for axis, data in report["joystick_analysis"].items():
            print(f"  {axis:12s}: mean={data['mean_error']:+.3f}, std={data['std_error']:.3f}, max={data['max_error']:.3f} ({data['bias']})")

        print("\n--- BUTTON ANALYSIS ---")
        print("\n  Missed Presses (human wanted, AI didn't press):")
        for btn, data in list(report["button_analysis"]["missed_presses"].items())[:5]:
            print(f"    {btn:20s}: {data['count']:4d} times ({data['percentage']:.1f}%)")

        print("\n  Unwanted Presses (AI pressed, human didn't want):")
        for btn, data in list(report["button_analysis"]["unwanted_presses"].items())[:5]:
            print(f"    {btn:20s}: {data['count']:4d} times ({data['percentage']:.1f}%)")

        print("\n--- INSIGHTS ---")
        for i, insight in enumerate(report["insights"], 1):
            print(f"  {i}. {insight}")

        print("\n--- RECOMMENDATIONS ---")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

        print("\n" + "="*70)


# =============================================================================
# DATASET
# =============================================================================
from nitrogen.shared import BUTTON_ACTION_TOKENS

# Map the recorder's lowercase human button names onto the model's button tokens.
_HUMAN_TO_TOKEN = {b.lower(): b for b in BUTTON_ACTION_TOKENS}


def human_buttons_to_vec(buttons: dict):
    """Build the model's 21-dim button vector (BUTTON_ACTION_TOKENS order) from a
    recorder human-button dict (lowercase names like 'south','left_trigger')."""
    vec = []
    for tok in BUTTON_ACTION_TOKENS:
        vec.append(float(buttons.get(tok.lower(), 0)))
    return vec


class CorrectionDataset(Dataset):
    """Human corrections, tokenized into the exact dict model.forward() expects.

    Each item runs the frame + the human action through the model's tokenizer, so
    the action is laid out as the model wants ([buttons, j_left, j_right], joysticks
    in [0,1]) and the vl/sa token ids + masks are built correctly.
    """

    def __init__(self, corrections_dirs, img_proc, tokenizer, analyzer,
                 frame_per_sample=1, action_horizon=16):
        self.img_proc = img_proc
        self.tokenizer = tokenizer
        if tokenizer is not None:
            tokenizer.training = True  # produce actions/has_real_action keys in encode()
        self.frame_per_sample = frame_per_sample
        self.action_horizon = action_horizon
        self.samples = []

        for corrections_dir in corrections_dirs:
            json_file = Path(corrections_dir) / "corrections.json"
            if not json_file.exists():
                continue
            with open(json_file) as f:
                corrections = json.load(f)
            for c in corrections:
                self.samples.append({
                    "frame_path": c["frame_path"],
                    "j_left": c["human"]["j_left"],
                    "j_right": c["human"]["j_right"],
                    "buttons": c["human"]["buttons"],
                })
                analyzer.add_correction(c["human"], c["ai"])

        print(f"Loaded {len(self.samples)} corrections from {len(corrections_dirs)} sessions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            image = Image.open(s["frame_path"]).convert("RGB")
            pixel_values = self.img_proc(image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 256, 256)

        # frames: (frame_per_sample, C, H, W); only the last frame is real.
        c, h, w = pixel_values.shape
        frames = torch.zeros(self.frame_per_sample, c, h, w, dtype=pixel_values.dtype)
        frames[-1] = pixel_values
        dropped = torch.ones(self.frame_per_sample, dtype=torch.bool)
        dropped[-1] = False

        # The tokenizer packs buttons/j_left/j_right itself (it normalizes the
        # joysticks and orders them as [buttons, j_left, j_right]). Provide each as
        # (chunks=1, action_horizon, dims); joysticks stay raw in [-1, 1].
        T = self.action_horizon
        buttons = np.tile(np.array(human_buttons_to_vec(s["buttons"]), dtype=np.float32), (1, T, 1))
        jl = np.tile(np.array(s["j_left"], dtype=np.float32), (1, T, 1))
        jr = np.tile(np.array(s["j_right"], dtype=np.float32), (1, T, 1))

        data = {"frames": frames, "dropped_frames": dropped,
                "buttons": buttons, "j_left": jl, "j_right": jr, "game": None}
        return self.tokenizer.encode(data)


def collate_tokenized(batch):
    """Stack a list of tokenizer.encode() dicts into a batched dict."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        elif isinstance(v0, np.ndarray):
            out[key] = torch.stack([torch.as_tensor(v) for v in vals], dim=0)
        elif isinstance(v0, (bool, np.bool_)):
            out[key] = torch.tensor([bool(v) for v in vals])
        else:
            out[key] = vals  # e.g. game: list of None
    return out


# =============================================================================
# TRAINING
# =============================================================================
class TrainingTracker:
    """Tracks training metrics."""

    def __init__(self):
        self.epoch_losses = []
        self.batch_losses = []
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.end_time = datetime.now()

    def add_batch_loss(self, loss: float):
        self.batch_losses.append(loss)

    def add_epoch_loss(self, loss: float):
        self.epoch_losses.append(loss)

    def get_summary(self) -> dict:
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "epochs": len(self.epoch_losses),
            "final_loss": self.epoch_losses[-1] if self.epoch_losses else None,
            "best_loss": min(self.epoch_losses) if self.epoch_losses else None,
            "loss_reduction": (self.epoch_losses[0] - self.epoch_losses[-1]) if len(self.epoch_losses) > 1 else 0,
            "training_duration_seconds": duration,
            "loss_curve": self.epoch_losses,
        }


def find_correction_sessions(corrections_dir: str) -> list:
    """Find all correction session directories."""
    sessions = []
    corrections_path = Path(corrections_dir)

    if not corrections_path.exists():
        return sessions

    for item in corrections_path.iterdir():
        if item.is_dir():
            if (item / "corrections.json").exists() or (item / "corrections.parquet").exists():
                sessions.append(str(item))

    return sorted(sessions)


def _to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch with the model's real flow-matching loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        data = _to_device(batch, device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(data)                 # NitroGen.forward computes the FM loss
            loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


def save_training_report(output_dir: Path, analysis_report: dict, training_summary: dict):
    """Save comprehensive training report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "correction_analysis": analysis_report,
        "training_summary": training_summary,
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nTraining report saved to: {report_path}")

    # Also save human-readable version
    txt_path = output_dir / "training_report.txt"
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("NITROGEN DAGGER TRAINING REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("CORRECTION ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total corrections: {analysis_report['summary']['total_corrections']}\n\n")

        f.write("Insights:\n")
        for insight in analysis_report.get('insights', []):
            f.write(f"  - {insight}\n")

        f.write("\nRecommendations:\n")
        for rec in analysis_report.get('recommendations', []):
            f.write(f"  - {rec}\n")

        f.write("\n\nTRAINING SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Epochs: {training_summary['epochs']}\n")
        f.write(f"Final loss: {training_summary['final_loss']}\n")
        f.write(f"Best loss: {training_summary['best_loss']}\n")
        f.write(f"Duration: {training_summary['training_duration_seconds']:.1f} seconds\n")

    print(f"Human-readable report: {txt_path}")


# =============================================================================
# MAIN
# =============================================================================
def _load_model_and_proc(checkpoint, device):
    """Load the full model if possible, else fall back to a bare image processor.

    Returns (model, tokenizer, img_proc, ckpt_config, model_loaded).
    """
    try:
        from nitrogen.inference_session import load_model
        model, tokenizer, img_proc, ckpt_config, _, _ = load_model(checkpoint)
        model.to(device)
        return model, tokenizer, img_proc, ckpt_config, True
    except Exception as e:
        print(f"WARNING: Could not load full model: {e}")
        print("Will use basic image processor for analysis...")
        from transformers import AutoImageProcessor
        img_proc = AutoImageProcessor.from_pretrained("google/siglip-large-patch16-256")
        return None, None, img_proc, None, False


def _run_training_loop(model, dataloader, optimizer, scheduler, device, tracker,
                       epochs, ckpt_config, sessions, output_path):
    """Run the epoch loop, saving the best checkpoint along the way."""
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()
        tracker.add_epoch_loss(loss)

        print(f"Epoch {epoch}: Loss = {loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            torch.save({
                "model": model.state_dict(),
                "ckpt_config": ckpt_config.model_dump() if ckpt_config else {},
                "epoch": epoch,
                "loss": loss,
                "dagger_sessions": sessions,
            }, output_path)
            print(f"  Saved best model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DAgger training with analytics")
    parser.add_argument("--corrections", type=str, default="./corrections",
                        help="Directory containing correction sessions")
    parser.add_argument("--checkpoint", type=str, default="../NitroGen/models/ng.pt",
                        help="Base model checkpoint")
    parser.add_argument("--output", type=str, default="./models/ng_dagger.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--freeze-vision", action="store_true",
                        help="Freeze vision encoder (saves VRAM)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze corrections, don't train")
    args = parser.parse_args()

    print("="*70)
    print("NITROGEN DAGGER TRAINING")
    print("="*70)

    # Find correction sessions
    sessions = find_correction_sessions(args.corrections)
    if not sessions:
        print(f"\nERROR: No correction sessions found in {args.corrections}")
        print("Run play_interactive.py first to collect corrections!")
        return

    print(f"\nFound {len(sessions)} correction sessions:")
    for s in sessions:
        print(f"  - {Path(s).name}")

    # Initialize analyzer
    analyzer = CorrectionAnalyzer()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load model (or just image processor for analysis)
    print(f"\nLoading model from {args.checkpoint}...")

    model, tokenizer, img_proc, ckpt_config, model_loaded = _load_model_and_proc(
        args.checkpoint, device
    )

    # Action/context dims come from the checkpoint config when the model loaded.
    if model_loaded:
        frame_per_sample = ckpt_config.modality_cfg.frame_per_sample
        action_horizon = ckpt_config.model_cfg.action_horizon
    else:
        frame_per_sample, action_horizon = 1, 16

    # Create dataset (this also feeds the analyzer)
    print("\nLoading and analyzing corrections...")
    dataset = CorrectionDataset(sessions, img_proc, tokenizer, analyzer,
                                frame_per_sample=frame_per_sample, action_horizon=action_horizon)

    if len(dataset) == 0:
        print("ERROR: No valid corrections found!")
        return

    # Generate and print analysis report
    analysis_report = analyzer.generate_report()
    analyzer.print_report(analysis_report)

    if args.analyze_only:
        print("\n--analyze-only flag set, skipping training.")
        # Save analysis report
        output_dir = Path(args.corrections)
        save_training_report(output_dir, analysis_report, {"epochs": 0, "final_loss": None, "best_loss": None, "training_duration_seconds": 0, "loss_curve": []})
        return

    if not model_loaded:
        print("\nERROR: Cannot train without loaded model.")
        print(f"Check that checkpoint exists: {args.checkpoint}")
        return

    # Configure trainable parameters
    if args.freeze_vision:
        print("\nFreezing vision encoder...")
        model.set_trainable_parameters(
            tune_vision_tower=False,
            tune_mm_projector=True,
            tune_diffusion_model=True,
            tune_multi_projector=True,
            tune_vl_mixing=True
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_tokenized,
    )

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training tracker
    tracker = TrainingTracker()
    tracker.start()

    # Training loop
    print("\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _run_training_loop(model, dataloader, optimizer, scheduler, device, tracker,
                       args.epochs, ckpt_config, sessions, output_path)

    tracker.end()

    # Save comprehensive report
    training_summary = tracker.get_summary()
    save_training_report(output_path.parent, analysis_report, training_summary)

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nResults:")
    print(f"  Corrections used: {len(dataset)}")
    print(f"  Final loss: {training_summary['final_loss']:.6f}")
    print(f"  Best loss: {training_summary['best_loss']:.6f}")
    print(f"  Duration: {training_summary['training_duration_seconds']:.1f}s")
    print(f"\nModel saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Test: python scripts/serve.py {output_path} --timesteps 2")
    print("  2. Collect more corrections with play_interactive.py")
    print("  3. Train again for further improvement!")
    print("="*70)


if __name__ == "__main__":
    main()
