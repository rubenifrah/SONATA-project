#!/usr/bin/env python3
"""Axis 1 shortcut analysis for SONATA-style point features.

This script is intentionally written as a real experiment entry point rather
than a placeholder. It supports two practical workflows:

1. A synthetic sanity check that always works and validates the diagnostics.
2. An NPZ-based evaluation path for real point features exported elsewhere.

The synthetic mode is important because the project plan explicitly asks for a
sanity check before running the full ScanNet study. The NPZ mode is the bridge
to the real project pipeline: once SONATA features are exported for scenes,
this script can score them without changing the analysis logic.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class SceneFeatures:
    """Container for one scene worth of geometry and features.

    The analysis only needs coordinates and a feature matrix. Keeping them in a
    small dataclass makes the downstream code explicit and easier to read.
    """

    name: str
    coord: np.ndarray
    feat: np.ndarray


def parse_args() -> argparse.Namespace:
    """Expose a small but useful CLI.

    The CLI is designed so that the default command runs the synthetic sanity
    check without extra inputs. Real experiments can switch to `npz` mode and
    point the script at exported feature files.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate how strongly point features encode low-level geometry."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory where metrics, plots, and metadata will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "npz"),
        default="synthetic",
        help="Run a synthetic sanity check or score NPZ feature exports.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="",
        help="Glob for NPZ files when --mode=npz, for example 'results/features/*.npz'.",
    )
    parser.add_argument(
        "--feature-key",
        type=str,
        default="feat",
        help="Name of the feature array inside each NPZ file.",
    )
    parser.add_argument(
        "--coord-key",
        type=str,
        default="coord",
        help="Name of the coordinate array inside each NPZ file.",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=0,
        help="Maximum number of scenes to load. Zero means no explicit limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of points used for the linear coordinate probe.",
    )
    parser.add_argument(
        "--synthetic-points",
        type=int,
        default=12000,
        help="Number of points in the synthetic sanity-check cloud.",
    )
    parser.add_argument(
        "--synthetic-semantic-dim",
        type=int,
        default=32,
        help="Feature dimensionality for each synthetic representation.",
    )
    return parser.parse_args()


def ensure_2d(array: np.ndarray, *, name: str) -> np.ndarray:
    """Reject malformed inputs early.

    The experiment assumes a dense matrix shape. Catching shape problems here
    produces cleaner failure modes than letting linear algebra crash later.
    """

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    return array


def rank_columns(array: np.ndarray) -> np.ndarray:
    """Approximate per-column ranking for Spearman correlation.

    This implementation uses a stable double argsort. It is not tie-aware in
    the strict statistical sense, but for continuous features and coordinates it
    is entirely adequate and keeps the script dependency-light.
    """

    order = np.argsort(array, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(array.shape[0], dtype=np.float64)[:, None]
    ranks[order, np.arange(array.shape[1])] = rows
    return ranks


def standardize(array: np.ndarray) -> np.ndarray:
    """Center and scale each column with numerical safeguards."""

    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (array - mean) / std


def fit_linear_regressor(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    """Fit a linear probe with a bias term using least squares.

    A simple linear regressor is exactly the right diagnostic here: if geometry
    is strongly recoverable from the features with a linear map, the shortcut is
    likely still present in a fairly direct form.
    """

    design = np.concatenate(
        [train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1
    )
    weights, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return weights


def predict_linear_regressor(test_x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply a least-squares probe fitted by :func:`fit_linear_regressor`."""

    design = np.concatenate(
        [test_x, np.ones((test_x.shape[0], 1), dtype=test_x.dtype)], axis=1
    )
    return design @ weights


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return per-dimension R^2 scores."""

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum(centered**2, axis=0)
    ss_tot = np.where(ss_tot < 1e-8, 1.0, ss_tot)
    return 1.0 - ss_res / ss_tot


def per_feature_correlations(
    feat: np.ndarray, coord: np.ndarray, *, spearman: bool
) -> np.ndarray:
    """Compute the maximum absolute correlation of each feature dimension.

    Each feature dimension is compared against x, y, and z. We keep the maximum
    absolute value because it is a compact summary of "how geometric" that
    feature dimension is, regardless of the axis or sign.
    """

    feat = standardize(feat)
    coord = standardize(coord)
    if spearman:
        feat = standardize(rank_columns(feat))
        coord = standardize(rank_columns(coord))
    corr = (feat.T @ coord) / feat.shape[0]
    return np.max(np.abs(corr), axis=1)


def pca_projection(feat: np.ndarray, output_dim: int = 2) -> np.ndarray:
    """Project features with a small SVD-based PCA.

    The script only needs a visualization-oriented embedding, so a compact SVD
    is enough and avoids any dependency on sklearn.
    """

    feat_centered = feat - feat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(feat_centered, full_matrices=False)
    basis = vt[:output_dim].T
    return feat_centered @ basis


def write_scatter_ppm(points: np.ndarray, output_path: Path, title: str) -> None:
    """Write a tiny dependency-free scatter plot as a PPM image.

    PPM is deliberately primitive, but it is trivial to generate with pure
    NumPy and can still be opened by common image viewers. This keeps the
    script portable on cluster environments where plotting stacks are fragile.
    """

    width = 640
    height = 480
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # Normalize to a visible range while guarding against degenerate cases.
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.where((maxs - mins) < 1e-8, 1.0, maxs - mins)
    norm = (points - mins) / spans

    # Leave a border so points do not touch the edges of the image.
    xs = np.clip((norm[:, 0] * (width - 40) + 20).astype(int), 0, width - 1)
    ys = np.clip((norm[:, 1] * (height - 40) + 20).astype(int), 0, height - 1)

    # Color points by polar angle in the 2D PCA plane to reveal cluster shape.
    angles = np.arctan2(points[:, 1], points[:, 0])
    red = ((np.sin(angles) * 0.5 + 0.5) * 255).astype(np.uint8)
    green = ((np.cos(angles) * 0.5 + 0.5) * 255).astype(np.uint8)
    blue = np.full_like(red, 110)

    for x, y, r, g, b in zip(xs, ys, red, green, blue):
        # Flip y because image coordinates grow downward.
        canvas[height - 1 - y, x] = np.array([r, g, b], dtype=np.uint8)

    output_path.write_bytes(
        f"P6\n# {title}\n{width} {height}\n255\n".encode("ascii") + canvas.tobytes()
    )


def evaluate_one_representation(
    feat: np.ndarray,
    coord: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> dict[str, object]:
    """Compute the main shortcut diagnostics for one feature matrix."""

    feat = ensure_2d(np.asarray(feat, dtype=np.float64), name="feat")
    coord = ensure_2d(np.asarray(coord, dtype=np.float64), name="coord")

    if coord.shape[1] != 3:
        raise ValueError(f"coord must have shape (N, 3), got {coord.shape}")
    if feat.shape[0] != coord.shape[0]:
        raise ValueError(
            f"feat and coord must agree on N, got {feat.shape[0]} and {coord.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(coord.shape[0])
    rng.shuffle(indices)
    train_size = max(1, min(coord.shape[0] - 1, int(train_fraction * coord.shape[0])))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    probe = fit_linear_regressor(feat[train_idx], coord[train_idx])
    pred = predict_linear_regressor(feat[test_idx], probe)
    per_axis_r2 = r2_score(coord[test_idx], pred)

    pearson_per_dim = per_feature_correlations(feat, coord, spearman=False)
    spearman_per_dim = per_feature_correlations(feat, coord, spearman=True)
    pca_2d = pca_projection(feat, output_dim=2)

    return {
        "num_points": int(coord.shape[0]),
        "feature_dim": int(feat.shape[1]),
        "coordinate_probe_r2_mean": float(np.mean(per_axis_r2)),
        "coordinate_probe_r2_xyz": {
            "x": float(per_axis_r2[0]),
            "y": float(per_axis_r2[1]),
            "z": float(per_axis_r2[2]),
        },
        "pearson_abs_corr_mean": float(np.mean(pearson_per_dim)),
        "pearson_abs_corr_median": float(np.median(pearson_per_dim)),
        "pearson_abs_corr_p95": float(np.quantile(pearson_per_dim, 0.95)),
        "spearman_abs_corr_mean": float(np.mean(spearman_per_dim)),
        "spearman_abs_corr_median": float(np.median(spearman_per_dim)),
        "spearman_abs_corr_p95": float(np.quantile(spearman_per_dim, 0.95)),
        "pca_projection": pca_2d,
    }


def make_semantic_labels(coord: np.ndarray) -> np.ndarray:
    """Create simple synthetic semantic regions that are not axis-aligned.

    The labels are deliberately defined by a combination of radial and angular
    structure so that the synthetic data is not a trivial height-binning toy.
    """

    radius = np.linalg.norm(coord[:, :2], axis=1)
    angle = np.arctan2(coord[:, 1], coord[:, 0])
    labels = np.zeros(coord.shape[0], dtype=np.int64)
    labels += radius > 0.75
    labels += angle > -0.4
    labels += coord[:, 2] > 0.1
    return labels % 4


def build_synthetic_scene(
    *,
    seed: int,
    num_points: int,
    semantic_dim: int,
) -> dict[str, SceneFeatures]:
    """Generate a pair of synthetic representations with different shortcut levels.

    The `shortcut_baseline` representation leaks coordinates directly into the
    features. The `semantic_like` representation mostly follows cluster identity
    with only weak geometric leakage. A good shortcut analysis should clearly
    separate the two.
    """

    rng = np.random.default_rng(seed)
    coord = rng.uniform(low=-1.0, high=1.0, size=(num_points, 3))
    labels = make_semantic_labels(coord)
    centroids = rng.normal(loc=0.0, scale=1.0, size=(4, semantic_dim))
    semantic_core = centroids[labels] + 0.12 * rng.normal(
        size=(num_points, semantic_dim)
    )

    # Shortcut-heavy features expose geometry directly and repeatedly.
    coord_repeated = np.concatenate(
        [coord, coord[:, [2]], coord[:, :2], coord[:, [0]]], axis=1
    )
    shortcut_tiled = np.tile(
        coord_repeated, math.ceil(semantic_dim / coord_repeated.shape[1])
    )[:, :semantic_dim]
    shortcut_feat = 1.6 * shortcut_tiled + 0.4 * semantic_core

    # More semantic features still contain a small amount of geometry, because
    # real models are rarely geometry-free. The point is that the leakage is
    # much weaker and harder to decode linearly.
    weak_geometry = np.tile(coord, math.ceil(semantic_dim / 3))[:, :semantic_dim]
    semantic_feat = 1.4 * semantic_core + 0.12 * weak_geometry

    return {
        "shortcut_baseline": SceneFeatures(
            name="synthetic_shortcut_baseline",
            coord=coord,
            feat=shortcut_feat,
        ),
        "semantic_like": SceneFeatures(
            name="synthetic_semantic_like",
            coord=coord,
            feat=semantic_feat,
        ),
    }


def load_npz_scenes(
    *,
    pattern: str,
    feature_key: str,
    coord_key: str,
    scene_limit: int,
) -> list[SceneFeatures]:
    """Load scene feature exports from NPZ files.

    The expected minimal schema is:
    - `coord`: (N, 3)
    - `feat`: (N, D)

    Additional arrays are fine; this script ignores them.
    """

    if not pattern:
        raise ValueError("--input-glob is required when --mode=npz")

    paths = [Path(path_str) for path_str in sorted(glob.glob(pattern))]
    if scene_limit > 0:
        paths = paths[:scene_limit]
    if not paths:
        raise FileNotFoundError(f"No NPZ files matched pattern: {pattern}")

    scenes: list[SceneFeatures] = []
    for path in paths:
        with np.load(path) as data:
            if coord_key not in data:
                raise KeyError(f"{path} does not contain coord key '{coord_key}'")
            if feature_key not in data:
                raise KeyError(f"{path} does not contain feature key '{feature_key}'")
            coord = np.asarray(data[coord_key], dtype=np.float64)
            feat = np.asarray(data[feature_key], dtype=np.float64)
        scenes.append(SceneFeatures(name=path.stem, coord=coord, feat=feat))
    return scenes


def concatenate_scenes(scenes: Iterable[SceneFeatures]) -> SceneFeatures:
    """Aggregate multiple scenes into one analysis batch.

    Axis 1 should not depend on a single anecdotal room. Concatenating scenes is
    a simple way to produce a stable global score while still keeping the raw
    per-scene metrics around.
    """

    scenes = list(scenes)
    if not scenes:
        raise ValueError("Need at least one scene to concatenate")
    coord = np.concatenate([scene.coord for scene in scenes], axis=0)
    feat = np.concatenate([scene.feat for scene in scenes], axis=0)
    return SceneFeatures(name="aggregate", coord=coord, feat=feat)


def sanitize_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Strip large NumPy arrays before JSON serialization."""

    output = dict(metrics)
    output.pop("pca_projection", None)
    return output


def run_synthetic(args: argparse.Namespace) -> dict[str, object]:
    """Execute the synthetic sanity check and export plots/metrics."""

    synthetic = build_synthetic_scene(
        seed=args.seed,
        num_points=args.synthetic_points,
        semantic_dim=args.synthetic_semantic_dim,
    )

    results: dict[str, object] = {"mode": "synthetic", "representations": {}}
    for representation_name, scene in synthetic.items():
        metrics = evaluate_one_representation(
            scene.feat,
            scene.coord,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        write_scatter_ppm(
            metrics["pca_projection"],
            args.results_dir / f"{representation_name}_pca.ppm",
            title=representation_name,
        )
        results["representations"][representation_name] = sanitize_metrics(metrics)

    shortcut_score = results["representations"]["shortcut_baseline"][
        "coordinate_probe_r2_mean"
    ]
    semantic_score = results["representations"]["semantic_like"][
        "coordinate_probe_r2_mean"
    ]
    results["sanity_check"] = {
        "passes": bool(shortcut_score > semantic_score),
        "expected_order": [
            "shortcut_baseline should have higher coordinate predictability",
            "semantic_like should have lower geometric dependence",
        ],
    }
    return results


def run_npz(args: argparse.Namespace) -> dict[str, object]:
    """Score real scene exports and write both per-scene and aggregate metrics."""

    scenes = load_npz_scenes(
        pattern=args.input_glob,
        feature_key=args.feature_key,
        coord_key=args.coord_key,
        scene_limit=args.scene_limit,
    )

    per_scene: dict[str, object] = {}
    for offset, scene in enumerate(scenes):
        metrics = evaluate_one_representation(
            scene.feat,
            scene.coord,
            train_fraction=args.train_fraction,
            seed=args.seed + offset,
        )
        write_scatter_ppm(
            metrics["pca_projection"],
            args.results_dir / f"{scene.name}_pca.ppm",
            title=scene.name,
        )
        per_scene[scene.name] = sanitize_metrics(metrics)

    aggregate_scene = concatenate_scenes(scenes)
    aggregate_metrics = evaluate_one_representation(
        aggregate_scene.feat,
        aggregate_scene.coord,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    write_scatter_ppm(
        aggregate_metrics["pca_projection"],
        args.results_dir / "aggregate_pca.ppm",
        title="aggregate",
    )

    return {
        "mode": "npz",
        "num_scenes": len(scenes),
        "per_scene": per_scene,
        "aggregate": sanitize_metrics(aggregate_metrics),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be strictly between 0 and 1")

    if args.mode == "synthetic":
        results = run_synthetic(args)
    else:
        results = run_npz(args)

    summary = {
        "axis": "axis1_shortcut",
        "analysis": "coordinate predictability and feature-geometry dependence",
        "config": {
            "mode": args.mode,
            "input_glob": args.input_glob,
            "feature_key": args.feature_key,
            "coord_key": args.coord_key,
            "scene_limit": args.scene_limit,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "synthetic_points": args.synthetic_points,
            "synthetic_semantic_dim": args.synthetic_semantic_dim,
        },
        "results": results,
    }

    (args.results_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(f"Shortcut analysis complete. Results written to {args.results_dir}")


if __name__ == "__main__":
    main()
