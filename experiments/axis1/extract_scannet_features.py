#!/usr/bin/env python3
"""Export dense SONATA features for one ScanNet-style point cloud.

This script is the real-world companion to `shortcut_test.py`.

Its job is intentionally narrow:
1. Load one raw point cloud.
2. Run the official pre-trained SONATA encoder.
3. Up-cast the hierarchical encoder output back to dense point features.
4. Save `coord` and `feat` into an NPZ file that `shortcut_test.py` can score.

The implementation mirrors the upstream SONATA demo code as closely as
possible, while adding the glue needed for project-scale experimentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sonata

try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


def parse_args() -> argparse.Namespace:
    """Expose a CLI suited for both quick tests and cluster jobs."""

    parser = argparse.ArgumentParser(
        description="Extract dense SONATA features from a ScanNet-style point cloud."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to a raw scene file. Supported: .ply, .npz, .npy, .pth.",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="",
        help="Optional built-in SONATA sample name, e.g. sample1, for smoke tests.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Target NPZ path containing at least coord and feat.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sonata",
        help="SONATA checkpoint name or local checkpoint path.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="facebook/sonata",
        help="Hugging Face repo used when model-name is not a local path.",
    )
    parser.add_argument(
        "--download-root",
        type=str,
        default="",
        help="Optional local cache root for checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, typically cuda on Juliet.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional cap on the number of raw input points before preprocessing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=53124,
        help="Random seed used by SONATA utilities and optional point subsampling.",
    )
    parser.add_argument(
        "--estimate-normals",
        action="store_true",
        help="Estimate normals for PLY inputs with Open3D when they are missing.",
    )
    parser.add_argument(
        "--force-disable-flash",
        action="store_true",
        help="Disable FlashAttention even if it is installed.",
    )
    return parser.parse_args()


def require_one_input(args: argparse.Namespace) -> None:
    """Keep invocation mistakes obvious."""

    if bool(args.input_path) == bool(args.sample_name):
        raise ValueError("Provide exactly one of --input-path or --sample-name")


def find_first_key(payload: dict[str, Any], candidates: list[str]) -> Any:
    """Return the first present key from a list of aliases."""

    for key in candidates:
        if key in payload:
            return payload[key]
    raise KeyError(f"None of the keys {candidates} were found in the input payload")


def maybe_limit_points(point: dict[str, np.ndarray], max_points: int, seed: int) -> None:
    """Subsample large scenes before preprocessing if requested.

    This is a pragmatic escape hatch for debugging. It should be disabled for
    final report numbers, but it is valuable when validating the end-to-end path
    on the cluster before scaling up.
    """

    if max_points <= 0 or point["coord"].shape[0] <= max_points:
        return

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(point["coord"].shape[0], size=max_points, replace=False))
    for key, value in list(point.items()):
        if isinstance(value, np.ndarray) and value.shape[0] == point["coord"].shape[0]:
            point[key] = value[keep]


def ensure_float32(name: str, array: np.ndarray, *, second_dim: int) -> np.ndarray:
    """Validate array shape and normalize dtype."""

    if array.ndim != 2 or array.shape[1] != second_dim:
        raise ValueError(f"{name} must have shape (N, {second_dim}), got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def normalize_point_dict(
    point: dict[str, Any], *, estimate_normals: bool, source_path: Path | None
) -> dict[str, np.ndarray]:
    """Convert heterogeneous raw inputs into the schema SONATA expects.

    SONATA's default transform collects features from `coord`, `color`, and
    `normal`. Raw ScanNet assets do not always provide normals in the format we
    need, so this function fills that gap conservatively.
    """

    coord = ensure_float32("coord", np.asarray(point["coord"]), second_dim=3)

    if point.get("color") is not None:
        color = ensure_float32("color", np.asarray(point["color"]), second_dim=3)
    else:
        color = np.zeros_like(coord, dtype=np.float32)

    # SONATA's NormalizeColor transform divides by 255. If colors already arrive
    # in [0, 1], rescale them back to 8-bit-like values first.
    if color.size > 0 and float(np.max(color)) <= 1.5:
        color = color * 255.0

    if point.get("normal") is not None:
        normal = ensure_float32("normal", np.asarray(point["normal"]), second_dim=3)
    elif estimate_normals and source_path and source_path.suffix.lower() == ".ply":
        normal = estimate_normals_from_ply(source_path)
    else:
        normal = np.zeros_like(coord, dtype=np.float32)

    normalized = {
        "coord": coord,
        "color": color,
        "normal": normal,
    }
    return normalized


def estimate_normals_from_ply(path: Path) -> np.ndarray:
    """Estimate normals with Open3D when the input file does not include them."""

    if o3d is None:
        raise RuntimeError("Open3D is required for --estimate-normals on PLY inputs")
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"PLY file contains no points: {path}")
    pcd.estimate_normals()
    return np.asarray(pcd.normals, dtype=np.float32)


def load_from_ply(path: Path) -> dict[str, np.ndarray]:
    """Load a point cloud from PLY using Open3D."""

    if o3d is None:
        raise RuntimeError("Open3D is required to read PLY files")

    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"PLY file contains no points: {path}")

    payload: dict[str, np.ndarray] = {
        "coord": np.asarray(pcd.points, dtype=np.float32),
    }
    if pcd.has_colors():
        payload["color"] = np.asarray(pcd.colors, dtype=np.float32)
    if pcd.has_normals():
        payload["normal"] = np.asarray(pcd.normals, dtype=np.float32)
    return payload


def load_from_npz(path: Path) -> dict[str, np.ndarray]:
    """Load a scene dictionary from a NumPy archive."""

    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    return {
        "coord": find_first_key(payload, ["coord", "xyz", "points"]),
        "color": payload.get("color", payload.get("rgb")),
        "normal": payload.get("normal", payload.get("normals")),
    }


def load_from_npy(path: Path) -> dict[str, np.ndarray]:
    """Load a scene from an `.npy` file.

    Two layouts are supported:
    - a dictionary saved with `allow_pickle=True`
    - a plain numeric matrix with columns interpreted as xyz[ rgb[ normals]]
    """

    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        payload = raw.item()
        return {
            "coord": find_first_key(payload, ["coord", "xyz", "points"]),
            "color": payload.get("color", payload.get("rgb")),
            "normal": payload.get("normal", payload.get("normals")),
        }

    matrix = np.asarray(raw)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError(
            "Plain NPY inputs must be a 2D matrix with at least 3 columns for xyz"
        )
    payload: dict[str, np.ndarray] = {"coord": matrix[:, :3]}
    if matrix.shape[1] >= 6:
        payload["color"] = matrix[:, 3:6]
    if matrix.shape[1] >= 9:
        payload["normal"] = matrix[:, 6:9]
    return payload


def load_from_pth(path: Path) -> dict[str, np.ndarray]:
    """Load a scene dictionary from Torch serialization."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {path}, got {type(payload)}")
    return {
        "coord": find_first_key(payload, ["coord", "xyz", "points"]),
        "color": payload.get("color", payload.get("rgb")),
        "normal": payload.get("normal", payload.get("normals")),
    }


def load_raw_point_cloud(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], str]:
    """Load a raw point cloud either from local storage or a bundled sample."""

    if args.sample_name:
        point = sonata.data.load(args.sample_name)
        if "segment200" in point:
            point.pop("segment200")
        if "segment20" in point:
            point["segment"] = point.pop("segment20")
        source_name = args.sample_name
        normalized = normalize_point_dict(
            point, estimate_normals=False, source_path=None
        )
        return normalized, source_name

    assert args.input_path is not None
    path = args.input_path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".ply":
        point = load_from_ply(path)
    elif suffix == ".npz":
        point = load_from_npz(path)
    elif suffix == ".npy":
        point = load_from_npy(path)
    elif suffix == ".pth":
        point = load_from_pth(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    normalized = normalize_point_dict(
        point, estimate_normals=args.estimate_normals, source_path=path
    )
    return normalized, path.stem


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load the pre-trained SONATA model with a FlashAttention fallback."""

    custom_config = None
    if flash_attn is None or args.force_disable_flash:
        custom_config = {
            "enc_patch_size": [1024 for _ in range(5)],
            "enable_flash": False,
        }

    model = sonata.load(
        args.model_name,
        repo_id=args.repo_id,
        download_root=args.download_root or None,
        custom_config=custom_config,
    )
    model = model.to(device)
    model.eval()
    return model


def move_tensors_to_device(point: dict[str, Any], device: torch.device) -> None:
    """Move transformed tensors in-place to the inference device."""

    for key, value in point.items():
        if isinstance(value, torch.Tensor):
            point[key] = value.to(device=device, non_blocking=True)


def upcast_to_dense_grid(point: Any) -> Any:
    """Mirror the official SONATA demo's feature up-casting path.

    After the encoder finishes, SONATA returns the lowest-resolution point
    structure. The demo code then walks back through pooling parents to produce
    dense point features suitable for visualization or downstream analysis.
    """

    for _ in range(2):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent

    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent

    return point


def run_inference(
    point: dict[str, np.ndarray], model: torch.nn.Module, device: torch.device
) -> dict[str, np.ndarray]:
    """Run SONATA on one scene and return arrays ready for NPZ export."""

    original_coord = point["coord"].copy()
    original_color = point["color"].copy()

    transform = sonata.transform.default()
    transformed = transform(point)
    move_tensors_to_device(transformed, device)

    with torch.inference_mode():
        coarse_point = model(transformed)
        coarse_feat = coarse_point.feat.detach().cpu().numpy().astype(np.float32)
        dense_grid_point = upcast_to_dense_grid(coarse_point)

        # `dense_grid_point` is still in the grid-sampled space created by the
        # default transform. `inverse` maps those features back to the original
        # raw point set so the exported `coord` and `feat` align one-to-one.
        dense_grid_feat = dense_grid_point.feat.detach().cpu().numpy().astype(np.float32)
        dense_original_feat = (
            dense_grid_point.feat[dense_grid_point.inverse]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        grid_coord = dense_grid_point.coord.detach().cpu().numpy().astype(np.float32)
        inverse = dense_grid_point.inverse.detach().cpu().numpy()

    return {
        "coord": original_coord.astype(np.float32),
        "color": original_color.astype(np.float32),
        "feat": dense_original_feat,
        "feat_grid": dense_grid_feat,
        "feat_coarse": coarse_feat,
        "grid_coord": grid_coord,
        "inverse": inverse,
    }


def write_outputs(
    exported: dict[str, np.ndarray],
    *,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write the NPZ file and a sidecar JSON metadata file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **exported)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    require_one_input(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    device = torch.device(args.device)
    sonata.utils.set_seed(args.seed)

    point, source_name = load_raw_point_cloud(args)
    maybe_limit_points(point, args.max_points, args.seed)

    model = load_model(args, device)
    exported = run_inference(point, model, device)

    metadata = {
        "source_name": source_name,
        "input_path": str(args.input_path.resolve()) if args.input_path else "",
        "sample_name": args.sample_name,
        "model_name": args.model_name,
        "repo_id": args.repo_id,
        "device": str(device),
        "num_input_points": int(exported["coord"].shape[0]),
        "dense_feature_dim": int(exported["feat"].shape[1]),
        "grid_feature_dim": int(exported["feat_grid"].shape[1]),
        "coarse_feature_dim": int(exported["feat_coarse"].shape[1]),
        "flash_attention_enabled": bool(flash_attn is not None and not args.force_disable_flash),
        "estimate_normals": bool(args.estimate_normals),
        "max_points": int(args.max_points),
    }
    write_outputs(exported, output_path=args.output_path, metadata=metadata)

    print(f"Feature export complete: {args.output_path}")
    print(
        f"Saved {exported['coord'].shape[0]} points with feature dim {exported['feat'].shape[1]}"
    )


if __name__ == "__main__":
    main()
