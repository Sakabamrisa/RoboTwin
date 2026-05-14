import argparse
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "envs" / "utils"


def load_pkl2hdf5_module():
    # Load the utility module without importing envs/__init__.py, which pulls in SAPIEN.
    envs_pkg = types.ModuleType("envs")
    utils_pkg = types.ModuleType("envs.utils")
    envs_pkg.__path__ = [str(REPO_ROOT / "envs")]
    utils_pkg.__path__ = [str(UTILS_DIR)]
    sys.modules.setdefault("envs", envs_pkg)
    sys.modules.setdefault("envs.utils", utils_pkg)

    module_name = "envs.utils.pkl2hdf5"
    module_path = UTILS_DIR / "pkl2hdf5.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pkl2hdf5_module = load_pkl2hdf5_module()


def episode_id(cache_dir):
    name = cache_dir.name
    if not name.startswith("episode"):
        raise ValueError(f"Invalid episode cache directory name: {cache_dir}")
    suffix = name[len("episode"):]
    if not suffix.isdigit():
        raise ValueError(f"Invalid episode cache directory name: {cache_dir}")
    return int(suffix)


def normalize_rollout_dir(path):
    path = Path(path).resolve()

    if (path / ".cache").is_dir():
        return path

    rollout_dir = path / "rollout_data"
    if (rollout_dir / ".cache").is_dir():
        return rollout_dir

    raise FileNotFoundError(
        f"Cannot find rollout cache under {path}. Expected either "
        f"{path}/.cache or {path}/rollout_data/.cache."
    )


def find_episode_cache_dirs(rollout_dir):
    cache_root = rollout_dir / ".cache"
    cache_dirs = [
        path for path in cache_root.iterdir()
        if path.is_dir() and path.name.startswith("episode")
    ]
    return sorted(cache_dirs, key=episode_id)


def list_episode_pkl_files(cache_dir):
    pkl_files = []
    for path in cache_dir.iterdir():
        if path.suffix == ".pkl" and path.stem.isdigit():
            pkl_files.append((int(path.stem), path))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {cache_dir}")

    pkl_files.sort()
    for expected, (frame_idx, path) in enumerate(pkl_files):
        if frame_idx != expected:
            raise ValueError(f"Missing file {expected}.pkl before {path}")

    return [path for _, path in pkl_files]


def convert_episode(
    cache_dir,
    rollout_dir,
    overwrite=False,
    remove_cache=False,
    dry_run=False,
    stride=1,
):
    ep_id = episode_id(cache_dir)
    hdf5_path = rollout_dir / "data" / f"episode{ep_id}.hdf5"
    video_path = rollout_dir / "video" / f"episode{ep_id}.mp4"

    if not overwrite and hdf5_path.exists() and video_path.exists():
        print(f"SKIP episode{ep_id}: existing hdf5 and video")
        return "skipped"

    pkl_files = list_episode_pkl_files(cache_dir)
    selected_pkl_files = pkl_files[::stride]
    if not selected_pkl_files:
        raise ValueError(f"No frames selected for episode{ep_id} with stride={stride}")

    print(f"CONVERT episode{ep_id}: {cache_dir}")
    print(f"  stride: {stride}, frames: {len(pkl_files)} -> {len(selected_pkl_files)}")
    print(f"  hdf5:  {hdf5_path}")
    print(f"  video: {video_path}")

    if dry_run:
        return "dry_run"

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    pkl2hdf5_module.pkl_files_to_hdf5_and_video(
        [str(path) for path in selected_pkl_files],
        str(hdf5_path),
        str(video_path),
    )

    if remove_cache:
        shutil.rmtree(cache_dir)
        print(f"  removed cache: {cache_dir}")

    return "converted"


def convert_rollout_dir(
    rollout_dir,
    overwrite=False,
    remove_cache=False,
    dry_run=False,
    stride=1,
):
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    rollout_dir = normalize_rollout_dir(rollout_dir)
    cache_dirs = find_episode_cache_dirs(rollout_dir)

    if not cache_dirs:
        raise FileNotFoundError(f"No episode cache directories found under {rollout_dir / '.cache'}")

    counts = {"converted": 0, "skipped": 0, "dry_run": 0, "failed": 0}
    for cache_dir in cache_dirs:
        try:
            status = convert_episode(
                cache_dir,
                rollout_dir,
                overwrite=overwrite,
                remove_cache=remove_cache,
                dry_run=dry_run,
                stride=stride,
            )
            counts[status] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAIL {cache_dir}: {repr(exc)}")

    print(
        "Done: "
        f"converted={counts['converted']}, "
        f"skipped={counts['skipped']}, "
        f"dry_run={counts['dry_run']}, "
        f"failed={counts['failed']}, "
        f"rollout_dir={rollout_dir}"
    )

    if counts["failed"] > 0:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert rollout pkl caches to hdf5 and mp4."
    )
    parser.add_argument(
        "rollout_dir",
        help=(
            "Path to rollout_data, or to an eval run directory containing rollout_data. "
            "Expected cache layout: rollout_data/.cache/episode*/0.pkl."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data/episode*.hdf5 and video/episode*.mp4.",
    )
    parser.add_argument(
        "--remove-cache",
        action="store_true",
        help="Remove each .cache/episode* directory after successful conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without writing files.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth pkl frame when writing hdf5 and mp4. Default: 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_rollout_dir(
        args.rollout_dir,
        overwrite=args.overwrite,
        remove_cache=args.remove_cache,
        dry_run=args.dry_run,
        stride=args.stride,
    )
