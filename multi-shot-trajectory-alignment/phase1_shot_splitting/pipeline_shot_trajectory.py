#!/usr/bin/env python3
"""

Input:
    - video.mp4: source video containing multiple shots
    - trajectory.json: shot boundary info
      {
        "shots": ["name1.mp4", "name2.mp4", ...],
        "shot_cut": [0, 109, 249, ...],
        "total_frames": 249
      }

Output:
    output_root/
    ├── shot_name_1/
    │   ├── shot_name_1.mp4
    │   └── shot_name_1_pose.json   (w2c 4x4 matrices)
    └── shot_name_2/
        ├── shot_name_2.mp4
        └── shot_name_2_pose.json

Requires: pip install -e <path-to-Pi3-repo>
"""

from contextlib import nullcontext

import argparse
import json
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np



def get_total_frames(video_path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    return int(subprocess.check_output(cmd).decode().strip())


def shot_ranges(manifest, total_frames):
    shots = manifest["shots"]
    cuts = manifest["shot_cut"]
    if not isinstance(shots, list) or not shots:
        raise ValueError("shots must be a nonempty list")
    if not isinstance(cuts, list) or len(cuts) not in (len(shots), len(shots) + 1):
        raise ValueError("Expected one start per shot and an optional end boundary")
    bounds = cuts + [total_frames] if len(cuts) == len(shots) else cuts
    if any(type(n) is not int or not 0 <= n <= total_frames for n in bounds):
        raise ValueError(f"Shot boundaries must be integers in [0, {total_frames}]")
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Shot boundaries must increase strictly")

    names = []
    seen = set()
    for filename in shots:
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"Invalid shot filename: {filename!r}")
        name = filename[:-4] if filename.lower().endswith(".mp4") else filename
        if (not name or name.startswith(".") or "\\" in name or ":" in name
                or any(ord(c) < 32 for c in name)):
            raise ValueError(f"Invalid shot filename: {filename!r}")
        key = name.casefold()
        if key in seen or key in {"trajectory.json", "global_prompt.txt", "stage1_sam2",
                                  "stage1_reorganized", "stage2_pi3", "stage3_aligned", "stage4_merged"}:
            raise ValueError(f"Duplicate or reserved shot name: {name!r}")
        seen.add(key)
        names.append(name)
    return list(zip(names, bounds, bounds[1:]))


def checked_shots(manifest, video_path, output_root):
    shots = shot_ranges(manifest, get_total_frames(video_path))
    # Validate every destination before creating or replacing any clips.
    for name, _, _ in shots:
        folder = output_root / name
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ValueError(f"Invalid shot directory: {folder}")
        for filename in (f"{name}.mp4", f"{name}_pose.json", ".phase1-pending"):
            if (folder / filename).is_symlink():
                raise ValueError(f"Unexpected shot output symlink: {folder / filename}")
    return shots


def split_video_by_shots(video_path: Path, trajectory_path: Path, output_root: Path):
    """Split a video into validated per-shot clips and return their locations."""
    manifest = json.loads(trajectory_path.read_text())
    results = []
    for name, start, end in checked_shots(manifest, video_path, output_root):
        folder = output_root / name
        folder.mkdir(parents=True, exist_ok=True)
        clip = folder / f"{name}.mp4"
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            staged = Path(tmp) / clip.name
            split_clip(video_path, staged, start, end)
            staged.replace(clip)
        results.append((name, folder, clip))
    return results


def complete_shot(folder, name, frame_count, interval):
    clip = folder / f"{name}.mp4"
    poses = folder / f"{name}_pose.json"
    if (folder / ".phase1-pending").exists() or not clip.is_file() or not poses.is_file():
        return False
    try:
        if get_total_frames(clip) != frame_count:
            return False
        frames = json.loads(poses.read_text())["frames"]
        expected = range(0, frame_count, interval)
        if len(frames) != len(expected):
            return False
        for i in expected:
            pose = np.asarray(frames[f"frame_{i:04d}"], dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                return False
        return True
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError):
        return False


def split_clip(video_path, clip_path, start, end):
    vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vf", vf,
        "-vsync", "0", "-an", "-c:v", "libx264", str(clip_path),
    ], check=True)


_model = None


def get_model(device):
    global _model
    if _model is None:
        from pi3.models.pi3 import Pi3

        _model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    return _model


def extract_trajectory(video_path: Path, device: str = "cuda", interval: int = 1) -> np.ndarray:
    """Run Pi3 on a video clip. Returns (N, 4, 4) c2w poses."""
    import torch
    from pi3.utils.basic import load_images_as_tensor

    dev = torch.device(device)
    model = get_model(dev)
    imgs = load_images_as_tensor(str(video_path), interval=interval).to(dev)

    autocast = nullcontext()
    if dev.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
        autocast = torch.amp.autocast("cuda", dtype=dtype)
    with torch.no_grad(), autocast:
        res = model(imgs[None])

    return res["camera_poses"][0].cpu().numpy()


def save_trajectory_w2c(poses_c2w: np.ndarray, output_path: Path, interval=1):
    frames = {}
    for i, c2w in enumerate(poses_c2w):
        frames[f"frame_{i * interval:04d}"] = np.linalg.inv(c2w).tolist()
    with open(output_path, "w") as f:
        json.dump({"frames": frames}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Shot splitting + trajectory extraction")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Reuse complete shots for the same manifest")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    trajectory_path = Path(args.trajectory_path)
    output_root = Path(args.output_root)

    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    if not trajectory_path.exists():
        sys.exit(f"Trajectory not found: {trajectory_path}")

    if args.interval <= 0:
        parser.error("--interval must be positive")
    manifest = json.loads(trajectory_path.read_text())
    shots = checked_shots(manifest, video_path, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    saved_manifest = output_root / "trajectory.json"
    reuse = args.resume and saved_manifest.is_file() and saved_manifest.read_bytes() == trajectory_path.read_bytes()
    rebuilt = 0
    for name, start, end in shots:
        folder = output_root / name
        if reuse and complete_shot(folder, name, end - start, args.interval):
            continue
        if rebuilt == 0:
            for stage in ("stage1_sam2", "stage2_pi3", "stage3_aligned", "stage4_merged"):
                (output_root / stage / ".complete").unlink(missing_ok=True)
        folder.mkdir(exist_ok=True)
        pending = folder / ".phase1-pending"
        pending.touch()
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            clip = Path(tmp) / f"{name}.mp4"
            poses = Path(tmp) / f"{name}_pose.json"
            split_clip(video_path, clip, start, end)
            trajectory = extract_trajectory(clip, device=args.device, interval=args.interval)
            save_trajectory_w2c(trajectory, poses, args.interval)
            if not complete_shot(Path(tmp), name, end - start, args.interval):
                raise ValueError(f"Incomplete clip or trajectory for {name}")
            clip.replace(folder / clip.name)
            poses.replace(folder / poses.name)
        pending.unlink()
        rebuilt += 1

    print(f"Done. {rebuilt} rebuilt, {len(shots) - rebuilt} reused -> {output_root}")


if __name__ == "__main__":
    main()
