"""
Validate a prepared training dataset.

Usage:
    python scripts/validate_dataset.py --dataset ./training_data/dataset.parquet
"""
import argparse
import random
from pathlib import Path

import polars as pl


def validate_dataset(parquet_path: str):
    """Validate the training dataset."""
    print(f"Validating: {parquet_path}")
    print("=" * 50)

    try:
        df = pl.read_parquet(parquet_path)
    except Exception as e:
        print(f"ERROR: Could not read dataset: {e}")
        return False

    # Basic statistics
    print("\n[Dataset Statistics]")
    print(f"  Total frames: {len(df):,}")
    print(f"  Columns: {df.columns}")

    if "chunk_id" in df.columns:
        print(f"  Chunks: {df['chunk_id'].n_unique()}")

    if "game_label" in df.columns:
        games = df["game_label"].unique().to_list()
        print(f"  Games: {games}")

    # Duration estimate
    fps = 30  # assumed
    duration_minutes = len(df) / fps / 60
    print(f"  Estimated duration: {duration_minutes:.1f} minutes ({duration_minutes/60:.1f} hours)")

    # Check frame existence
    print("\n[Frame Validation]")
    if "frame_path" in df.columns:
        sample_size = min(100, len(df))
        sample_paths = random.sample(df["frame_path"].to_list(), sample_size)

        missing = 0
        for path in sample_paths:
            if not Path(path).exists():
                missing += 1
                if missing <= 3:
                    print(f"  Missing: {path}")

        if missing > 0:
            print(f"  WARNING: {missing}/{sample_size} sampled frames missing!")
        else:
            print(f"  All {sample_size} sampled frames exist")
    else:
        print("  WARNING: No 'frame_path' column found!")

    # Action statistics
    print("\n[Action Distribution]")

    # Joystick analysis
    if "j_left" in df.columns:
        # Note: j_left is stored as a list [x, y]
        try:
            j_left_vals = df["j_left"].to_list()
            if j_left_vals and isinstance(j_left_vals[0], list):
                x_vals = [v[0] for v in j_left_vals if v]
                y_vals = [v[1] for v in j_left_vals if v]
                print(f"  Left Joystick X: mean={sum(x_vals)/len(x_vals):.3f}")
                print(f"  Left Joystick Y: mean={sum(y_vals)/len(y_vals):.3f}")
        except:
            print("  Could not analyze joystick data")

    # Button analysis
    button_cols = ["south", "east", "west", "north", "left_shoulder", "right_shoulder",
                   "left_trigger", "right_trigger", "dpad_up", "dpad_down", "dpad_left", "dpad_right"]

    for col in button_cols:
        if col in df.columns:
            try:
                mean = df[col].mean()
                if mean is not None:
                    print(f"  {col}: {mean:.2%} pressed")
            except:
                pass

    # Quality checks
    print("\n[Quality Checks]")

    # Check for duplicate frames
    if "frame_path" in df.columns:
        n_unique = df["frame_path"].n_unique()
        if n_unique < len(df):
            print(f"  WARNING: {len(df) - n_unique} duplicate frame paths!")
        else:
            print("  No duplicate frame paths")

    # Check for all-zero actions (player not moving)
    if "j_left" in df.columns and "j_right" in df.columns:
        # This is a simplified check
        print("  (Joystick movement analysis requires manual review)")

    print("\n" + "=" * 50)
    print("Validation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate training dataset")
    parser.add_argument("--dataset", type=str, default="./training_data/dataset.parquet",
                        help="Path to dataset parquet file")
    args = parser.parse_args()

    validate_dataset(args.dataset)


if __name__ == "__main__":
    main()
