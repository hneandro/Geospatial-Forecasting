from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "EDA" / "data"
DEFAULT_MODEL_OUTPUT = Path(__file__).resolve().with_name("trained_plume_model.pkl")


def downsample_grid(grid: np.ndarray, factor: int = 4) -> np.ndarray:
    """Average-pool a 2D grid by a given factor."""
    if grid.ndim != 2:
        raise ValueError("Expected a 2D grid for downsampling.")

    h, w = grid.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError("Grid dimensions must be divisible by downsample factor.")

    return grid.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def extract_features(input_window: np.ndarray) -> np.ndarray:
    """Extract vector features from a single input window."""
    if input_window.shape != (3, 10, 64, 64):
        raise ValueError(f"Expected input window shape (3, 10, 64, 64), got {input_window.shape}.")

    concentration_frames = input_window[:, 0, :, :]
    downsampled = np.stack([downsample_grid(frame, factor=4) for frame in concentration_frames], axis=0)
    concentration_flat = downsampled.ravel()

    meteorology = input_window[:, 1:10, :, :]
    meteorology_summary = meteorology.mean(axis=(2, 3)).ravel()

    concentration_stats = []
    for frame in concentration_frames:
        concentration_stats.append(frame.mean())
        concentration_stats.append(frame.max())
    concentration_stats = np.asarray(concentration_stats, dtype=np.float32)

    return np.concatenate((concentration_flat, meteorology_summary, concentration_stats), axis=0)


def extract_target(target_window: np.ndarray) -> np.ndarray:
    """Extract the next-frame plume target from a single target window."""
    if target_window.shape != (1, 10, 64, 64):
        raise ValueError(f"Expected target window shape (1, 10, 64, 64), got {target_window.shape}.")

    target_plume = target_window[0, 0, :, :]
    return downsample_grid(target_plume, factor=4).ravel()


def load_window_paths(data_dir: Path, max_samples: int | None = None, seed: int = 42) -> list[Path]:
    windows_dir = data_dir / "windows"
    if not windows_dir.exists():
        raise FileNotFoundError(f"Data windows folder not found: {windows_dir}")

    window_files = sorted(windows_dir.glob("*.npz"))
    if not window_files:
        raise FileNotFoundError(f"No .npz files found in {windows_dir}")

    if max_samples is not None and len(window_files) > max_samples:
        random.Random(seed).shuffle(window_files)
        window_files = window_files[:max_samples]

    return window_files


def build_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    feature_list = []
    target_list = []

    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            features = extract_features(archive["input"])
            target = extract_target(archive["target"])

        feature_list.append(features)
        target_list.append(target)

    X = np.asarray(feature_list, dtype=np.float32)
    y = np.asarray(target_list, dtype=np.float32)
    return X, y


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def train_and_evaluate(data_dir: Path, max_samples: int | None, model_output: Path) -> None:
    print(f"Loading data from: {data_dir}")
    window_paths = load_window_paths(data_dir, max_samples=max_samples)
    print(f"Using {len(window_paths)} windows for training.")

    X, y = build_dataset(window_paths)
    print(f"Feature shape: {X.shape}, target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred, multioutput="uniform_average")

    print("Training complete.")
    print(f"Test samples: {y_test.shape[0]}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R^2 score: {r2:.6f}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model,
        "downsample_factor": 4,
        "feature_description": {
            "plume_downsampled_frames": "3 frames of plume concentration downsampled to 16x16",
            "meteorology_summary": "3 frames x 9 broadcast meteorological channels averaged over space",
            "concentration_stats": "mean and max for each of the 3 input plume frames",
        },
        "target_description": "Next frame plume concentration downsampled to 16x16",
    }
    with model_output.open("wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved trained model artifact to: {model_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a learned plume prediction model using the EDA ConvLSTM dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the EDA data directory containing windows/*.npz.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8000,
        help="Maximum number of windows to load for training. Use 0 or omit to load all available windows.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help="Path to write the trained model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = None if args.max_samples <= 0 else args.max_samples
    train_and_evaluate(args.data_dir, max_samples, args.model_output)


if __name__ == "__main__":
    main()
