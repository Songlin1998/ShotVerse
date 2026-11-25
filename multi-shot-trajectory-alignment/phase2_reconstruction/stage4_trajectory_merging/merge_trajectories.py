#!/usr/bin/env python3
"""
Merge multiple aligned per-shot trajectories into a single trajectory.

Reads all *_all.json from input dir (stage3 output), re-numbers frames
sequentially, and writes a single merged JSON.

Input:  stage3_aligned/{clip_name}_all.json  (c2w per-frame poses)
Output: stage4_merged/{scene_name}_merged_all_poses.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def extract_frame_number(frame_name):
    match = re.search(r'frame_(\d+)', str(frame_name))
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', str(frame_name))
    if match:
        return int(match.group(1))
    return None


def sort_trajectory_files(file_paths: List[Path]) -> List[Path]:
    def sort_key(path):
        numbers = re.findall(r'\d+', path.stem)
        return int(numbers[-1]) if numbers else 0
    return sorted(file_paths, key=sort_key)


def merge_multiple_trajectories(file_paths: List[Path], output_path: Path, scene_name: str = "merged"):
    """Load, re-number, and merge all trajectory files sequentially."""
    if not file_paths:
        return

    all_data = []
    for fp in file_paths:
        data = load_json(fp)
        frames = data["frames"]
        if not frames:
            raise ValueError(f"Empty trajectory: {fp}")
        all_data.append((fp, data, frames))

    if not all_data:
        return

    merged_frames = {}
    current_frame_num = 0
    merge_info = []

    for fp, data, frames in all_data:
        # Sort frames by original number
        frame_items = []
        for frame_name, pose in frames.items():
            num = extract_frame_number(frame_name)
            if num is not None:
                frame_items.append((num, frame_name, pose))
        frame_items.sort(key=lambda x: x[0])

        start_frame = current_frame_num

        for _, _, pose in frame_items:
            merged_frames[f"frame_{current_frame_num:04d}"] = pose
            current_frame_num += 1

        end_frame = current_frame_num - 1

        clip_name = data.get("clip_name", data.get("clip", fp.stem))
        merge_info.append({
            "file": fp.name,
            "clip_name": clip_name,
            "original_frames": len(frames),
            "frame_range": f"frame_{start_frame:04d} ~ frame_{end_frame:04d}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "scale_factor": data.get("scale_factor", None),
            "scale_estimated": data.get("scale_estimated"),
            "pi3_distance": data.get("pi3_distance", None),
            "full_distance": data.get("full_distance", None),
        })

    first_data = all_data[0][1]
    merged_data = {
        "scene_name": scene_name,
        "video_folder": first_data.get("video_folder", ""),
        "num_clips": len(all_data),
        "num_frames": len(merged_frames),
        "coordinate_system": "c2w",
        "merged_clips": merge_info,
        "frames": merged_frames,
    }

    save_json(merged_data, output_path)


def process_scene(input_dir: Path, output_dir: Path, shot_manifest=None):
    scene_name = input_dir.parent.name
    if shot_manifest:
        shots = load_json(shot_manifest)["shots"]
        names = [shot[:-4] if shot.lower().endswith(".mp4") else shot for shot in shots]
        if len(set(names)) != len(names) or any(not n or n.startswith(".") or Path(n).name != n for n in names):
            raise ValueError("Expected distinct shot filenames in the manifest")
        trajectory_files = [input_dir / f"{name}_all.json" for name in names]
    else:
        trajectory_files = sort_trajectory_files(input_dir.glob("*_all.json"))
    if not trajectory_files:
        raise ValueError(f"No shot trajectories in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{scene_name}_merged_all_poses.json"
    merge_multiple_trajectories(trajectory_files, output_file, scene_name)


def main():
    parser = argparse.ArgumentParser(description="Merge aligned per-shot trajectories")
    parser.add_argument("--input-dir", type=str, required=True, help="stage3_aligned directory")
    parser.add_argument("--output-dir", type=str, required=True, help="stage4_merged output directory")
    parser.add_argument("--shot-manifest", type=Path, help="trajectory.json with shots in playback order")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    process_scene(input_dir, output_dir, args.shot_manifest)
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
