"""
Combine training data chunks into a unified dataset for fine-tuning.

Usage:
    python scripts/prepare_training_data.py --input ./training_data --output ./training_data/dataset.parquet
"""
import argparse
from pathlib import Path

import polars as pl


def prepare_dataset(data_dir: str, output_path: str):
    """Combine all recording chunks into a training dataset."""
    chunks = []
    data_path = Path(data_dir)

    for chunk_dir in sorted(data_path.glob("*_chunk_*")):
        actions_file = chunk_dir / "actions_processed.parquet"
        frames_dir = chunk_dir / "frames"

        if not actions_file.exists():
            print(f"Skipping {chunk_dir.name}: no actions file")
            continue

        try:
            df = pl.read_parquet(actions_file)
        except Exception as e:
            print(f"Error reading {actions_file}: {e}")
            continue

        # Get frame paths
        frame_paths = sorted(frames_dir.glob("*.png"))

        if len(frame_paths) < len(df):
            print(f"Warning: {chunk_dir.name} has {len(df)} actions but only {len(frame_paths)} frames")
            df = df.head(len(frame_paths))

        # Add metadata columns
        df = df.with_columns([
            pl.Series("frame_path", [str(p) for p in frame_paths[:len(df)]]),
            pl.Series("frame_idx", list(range(len(df)))),
            pl.lit(chunk_dir.name).alias("chunk_id"),
            pl.lit(chunk_dir.name.split("_")[0]).alias("game_label")
        ])

        chunks.append(df)
        print(f"Loaded {chunk_dir.name}: {len(df)} frames")

    if not chunks:
        print("ERROR: No valid chunks found!")
        return None

    # Combine all chunks
    combined = pl.concat(chunks)

    # Save
    combined.write_parquet(output_path)

    print(f"\n{'='*50}")
    print(f"Dataset created: {output_path}")
    print(f"Total frames: {len(combined):,}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Games: {combined['game_label'].unique().to_list()}")
    print(f"{'='*50}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--input", type=str, default="./training_data",
                        help="Input directory with recording chunks")
    parser.add_argument("--output", type=str, default="./training_data/dataset.parquet",
                        help="Output parquet file path")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
