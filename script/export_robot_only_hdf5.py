import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


DEFAULT_RENDERER_DIR = Path("/mnt/data/zq/FlowWAM_WorldArena/inference")


def load_robot_only_renderer(renderer_dir):
    renderer_dir = Path(renderer_dir).resolve()
    sys.path.insert(0, str(renderer_dir))
    from robot_only_renderer import RobotOnlyRenderer

    return RobotOnlyRenderer


def encode_rgb_frames(frames):
    encoded = []
    max_len = 0

    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG")
        payload = buf.getvalue()
        encoded.append(payload)
        max_len = max(max_len, len(payload))

    if max_len == 0:
        raise ValueError("No RGB frames to encode.")

    return [payload.ljust(max_len, b"\0") for payload in encoded], max_len


def write_robot_only_hdf5(output_path, camera, frames):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded, max_len = encode_rgb_frames(frames)
    with h5py.File(output_path, "w") as f:
        group = f.create_group("observation").create_group(camera)
        group.create_dataset("rgb", data=encoded, dtype=f"S{max_len}")


def normalize_input_path(path):
    path = Path(path).resolve()
    if (path / "data").is_dir():
        return [path]

    variant_dirs = []
    for data_dir in sorted(path.glob("*/*/data")):
        variant_dirs.append(data_dir.parent)

    if not variant_dirs:
        raise FileNotFoundError(
            f"Cannot find dataset episodes under {path}. Expected either "
            f"{path}/data/episode*.hdf5 or {path}/<task>/<variant>/data/episode*.hdf5."
        )

    return variant_dirs


def find_episode_hdf5_files(variant_dir):
    data_dir = Path(variant_dir) / "data"
    files = sorted(data_dir.glob("episode*.hdf5"), key=episode_sort_key)
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")
    return files


def episode_sort_key(path):
    stem = Path(path).stem
    suffix = stem[len("episode"):] if stem.startswith("episode") else stem
    return int(suffix) if suffix.isdigit() else suffix


def convert_episode(renderer, episode_path, output_path, camera, overwrite=False, dry_run=False):
    if output_path.exists() and not overwrite:
        print(f"SKIP {episode_path.name}: existing {output_path}")
        return "skipped"

    T = renderer.get_episode_length(str(episode_path))
    indices = list(range(T))

    print(f"RENDER {episode_path}")
    print(f"  frames: {T}")
    print(f"  output: {output_path}")

    if dry_run:
        return "dry_run"

    frames = renderer.render_episode(str(episode_path), indices, camera=camera)
    write_robot_only_hdf5(output_path, camera, frames)
    return "converted"


def export_robot_only(
    dataset_path,
    renderer_dir,
    robot_variant,
    camera,
    render_resolution,
    overwrite=False,
    dry_run=False,
    limit=None,
):
    RobotOnlyRenderer = load_robot_only_renderer(renderer_dir)
    variant_dirs = normalize_input_path(dataset_path)
    renderer = RobotOnlyRenderer(
        embodiment_dir=str(Path(renderer_dir).resolve()),
        variant=robot_variant,
        render_resolution=render_resolution,
    )

    counts = {"converted": 0, "skipped": 0, "dry_run": 0, "failed": 0}
    processed = 0

    try:
        for variant_dir in variant_dirs:
            episode_paths = find_episode_hdf5_files(variant_dir)
            output_dir = Path(variant_dir) / "robot_only" / "data"

            for episode_path in episode_paths:
                if limit is not None and processed >= limit:
                    print(f"Limit reached: {limit}")
                    return counts

                output_path = output_dir / episode_path.name
                try:
                    status = convert_episode(
                        renderer,
                        episode_path,
                        output_path,
                        camera,
                        overwrite=overwrite,
                        dry_run=dry_run,
                    )
                    counts[status] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    print(f"FAIL {episode_path}: {repr(exc)}")
                processed += 1
    finally:
        renderer.close()

    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render robot-only HDF5 files for RoboTwin episodes."
    )
    parser.add_argument(
        "dataset_path",
        help=(
            "Either a variant directory containing data/episode*.hdf5, "
            "or a root containing <task>/<variant>/data/episode*.hdf5."
        ),
    )
    parser.add_argument(
        "--renderer-dir",
        default=str(DEFAULT_RENDERER_DIR),
        help="Directory containing robot_only_renderer.py and embodiments/.",
    )
    parser.add_argument(
        "--robot-variant",
        default="aloha-agilex_clean_50",
        help="Robot variant string understood by RobotOnlyRenderer.",
    )
    parser.add_argument(
        "--camera",
        default="head_camera",
        help="Camera to render and write under observation/<camera>/rgb.",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        nargs=2,
        default=(320, 256),
        metavar=("WIDTH", "HEIGHT"),
        help="Robot-only render resolution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing robot_only/data/episode*.hdf5 files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rendered without writing files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most this many episodes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    counts = export_robot_only(
        args.dataset_path,
        args.renderer_dir,
        args.robot_variant,
        args.camera,
        tuple(args.render_resolution),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print(
        "Done: "
        f"converted={counts['converted']}, "
        f"skipped={counts['skipped']}, "
        f"dry_run={counts['dry_run']}, "
        f"failed={counts['failed']}"
    )
    if counts["failed"] > 0:
        raise SystemExit(1)
