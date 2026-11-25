#!/usr/bin/env python3
"""
Input:
  - Pi3 poses: *_poses.json (c2w, sampled frames only)
  - Full poses: {clip_name}_pose.json (w2c, all frames)

Output:
  - {clip_name}_all.json (c2w, all frames, aligned + scaled)
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_np(mat):
    return np.array(mat, dtype=np.float32)


def extract_frame_number(frame_name):
    """Extract integer frame index from names like frame_0010_black_mask.jpg."""
    match = re.search(r'frame_(\d+)', frame_name)
    return int(match.group(1)) if match else None


def find_matching_full_frame_key(pi3_image_name, full_frames):
    """Find the full-frame dict key corresponding to a Pi3 image name."""
    frame_num = extract_frame_number(pi3_image_name)
    if frame_num is None:
        raise ValueError(f"Cannot extract frame number from: {pi3_image_name}")

    candidates = [
        f"frame_{frame_num:04d}",
        f"frame_{frame_num:04d}.jpg",
        f"frame_{frame_num:04d}.png",
        f"frame_{frame_num}",
        pi3_image_name,
        Path(pi3_image_name).stem,
    ]
    for key in candidates:
        if key in full_frames:
            return key

    available = sorted(full_frames.keys())
    raise KeyError(
        f"No matching key for frame {frame_num} in full poses. "
        f"Tried: {candidates}. Available range: {available[0]} ~ {available[-1]}"
    )


def extract_clip_name_from_pi3_file(pi3_filename):
    """
    Extract clip name from Pi3 output filename.
    e.g. 'scene_0137_t0040cm6ddq_scene_0542_poses.json' -> 't0040cm6ddq_scene_0542'
    """
    stem = Path(pi3_filename).stem
    if stem.endswith('_poses'):
        stem = stem[:-6]

    parts = stem.split('_')

    scene_idx = -1
    for i, part in enumerate(parts):
        if part == 'scene' and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                scene_idx = i
                break
            except ValueError:
                continue

    if scene_idx >= 0 and scene_idx + 2 < len(parts):
        return '_'.join(parts[scene_idx + 2:])

    if len(parts) > 2:
        return '_'.join(parts[2:])

    return stem


def align_single_shot(pi3_json_path, full_pose_json_path, output_path, pi3_data=None):
    """Align a single shot: Pi3 sparse c2w + full w2c -> aligned full c2w."""
    if pi3_data is None:
        pi3_data = load_json(pi3_json_path)
    full_data = load_json(full_pose_json_path)

    pi3_frames = pi3_data["frames"]
    full_frames = full_data.get("frames", full_data)

    if len(pi3_frames) == 0:
        raise ValueError("Pi3 poses are empty")

    keys = [find_matching_full_frame_key(frame["image"], full_frames) for frame in pi3_frames]
    pi3_poses = np.asarray([frame["pose"] for frame in pi3_frames], dtype=np.float64)
    # The first matched camera fixes rotation and translation.
    transform = pi3_poses[0] @ np.asarray(full_frames[keys[0]], dtype=np.float64)
    aligned_tmp = {name: transform @ np.linalg.inv(np.asarray(w2c, dtype=np.float64))
                   for name, w2c in full_frames.items()}
    origin = aligned_tmp[keys[0]][:3, 3]
    local = np.stack([aligned_tmp[key][:3, 3] - origin for key in keys])
    joint = pi3_poses[:, :3, 3] - pi3_poses[0, :3, 3]

    # Fit the lengths of all corresponding displacements relative to the anchor.
    local_radii = np.linalg.norm(local, axis=1)
    joint_radii = np.linalg.norm(joint, axis=1)
    if not (np.isfinite(local_radii).all() and np.isfinite(joint_radii).all()):
        raise ValueError("Non-finite sampled camera positions")
    estimated = local_radii.max() > 1e-6 and joint_radii.max() > 1e-6
    if estimated:
        scale = np.dot(local_radii, joint_radii) / np.dot(local_radii, local_radii)
        if scale <= 0:
            raise ValueError("Inconsistent motion between matching samples")
    else:
        if max(local_radii.max(), joint_radii.max()) > 1e-6:
            raise ValueError("Only one trajectory has measurable sampled translation")
        scale = 1.0
    # Retain the original endpoint-distance fields for existing readers.
    pi3_dist, full_dist = joint_radii[-1], local_radii[-1]
    aligned = {}

    for frame_name, c2w_mat in aligned_tmp.items():
        R = c2w_mat[:3, :3]
        t = c2w_mat[:3, 3]
        t_scaled = origin + (t - origin) * scale

        c2w_out = np.eye(4, dtype=np.float32)
        c2w_out[:3, :3] = R
        c2w_out[:3, 3] = t_scaled
        aligned[frame_name] = c2w_out.tolist()

    # Save
    clip_name = pi3_data.get("clip") or extract_clip_name_from_pi3_file(pi3_json_path)
    out_obj = {
        "clip_name": clip_name,
        "num_frames": len(aligned),
        "pi3_sampled_frames": len(pi3_frames),
        "scale_factor": float(scale),
        "scale_estimated": bool(estimated),
        "pi3_distance": float(pi3_dist),
        "full_distance": float(full_dist),
        "coordinate_system": "c2w",
        "frames": aligned,
    }
    with open(output_path, "w") as f:
        json.dump(out_obj, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Align Pi3 poses to full-frame GT poses")
    parser.add_argument("--pi3-root", type=str, required=True, help="Dir with Pi3 *_poses.json files")
    parser.add_argument("--full-pose-root", type=str, required=True, help="Dir with per-shot full pose JSONs")
    parser.add_argument("--output-root", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    pi3_root = Path(args.pi3_root)
    full_pose_root = Path(args.full_pose_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pi3_files = list(pi3_root.glob("*_poses.json"))
    if not pi3_files:
        raise SystemExit(f"No Pi3 pose files found in {pi3_root}")

    success, fail = 0, 0

    for pi3_file in tqdm(pi3_files, desc="Aligning"):
        pi3_data = load_json(pi3_file)
        clip_name = pi3_data.get("clip") or extract_clip_name_from_pi3_file(pi3_file.name)
        if not clip_name or Path(clip_name).name != clip_name or clip_name.startswith("."):
            raise ValueError(f"Invalid clip identifier: {clip_name!r}")
        full_pose_file = full_pose_root / clip_name / f"{clip_name}_pose.json"

        if not full_pose_file.exists():
            fail += 1
            continue

        output_file = output_root / f"{clip_name}_all.json"

        try:
            align_single_shot(pi3_file, full_pose_file, output_file, pi3_data)
            success += 1
        except Exception as e:
            print(f"Failed: {pi3_file.name} ({e})")
            fail += 1

    print(f"Done. success={success} failed={fail} total={len(pi3_files)}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
