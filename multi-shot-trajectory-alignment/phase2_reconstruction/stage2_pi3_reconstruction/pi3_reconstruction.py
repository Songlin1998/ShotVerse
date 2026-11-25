#!/usr/bin/env python3
"""
Pi3 multi-shot 3D reconstruction.

Expects directory structure:
    root_dir/
    └── video_folder/
        ├── clip_xxx/*.jpg          (flat mode)
        └── scene_xxxx/clip_xxx/*.jpg  (hierarchical mode)

Outputs per clip:
    - *_poses.json   (per-clip camera poses, c2w 4x4)
    - *.glb          (3D point cloud + camera frustums)
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import trimesh
from tqdm import tqdm

from pi3.models.pi3 import Pi3


# ------------------------------------------------------------------ #
#  Camera frustum visualization
# ------------------------------------------------------------------ #

def create_camera_frustum(pose, scale=0.1, color=(255, 0, 0, 255)):
    w, h, z = scale, scale * 0.75, scale * 1.5
    vertices = np.array([
        [0, 0, 0], [-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z],
    ], dtype=np.float32)

    ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
    world_verts = (pose @ np.hstack([vertices, ones]).T).T[:, :3]

    edges = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    lines = np.array([[world_verts[s], world_verts[e]] for s, e in edges])
    frustum = trimesh.load_path(lines)
    for ent in frustum.entities:
        ent.color = list(color)
    return frustum


# ------------------------------------------------------------------ #
#  Image loading with resolution normalization
# ------------------------------------------------------------------ #

PIXEL_LIMIT = 255000


def compute_target_resolution(image_paths):
    """Adapted from Pi3 image sizing in pi3/utils/basic.py (see third_party_licenses)."""
    first_img = Image.open(image_paths[0]).convert("RGB")
    W_orig, H_orig = first_img.size

    scale = (PIXEL_LIMIT / (W_orig * H_orig)) ** 0.5 if W_orig * H_orig > 0 else 1.0
    W_target, H_target = W_orig * scale, H_orig * scale

    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def load_images_with_target_resolution(image_paths, device):
    target_w, target_h = compute_target_resolution(image_paths)
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs, dim=0).to(device)


# ------------------------------------------------------------------ #
#  Inference helper
# ------------------------------------------------------------------ #

def run_inference(model, imgs, device):
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    model = model.to(device).eval()
    with torch.no_grad(), ctx:
        return model(imgs[None])


# ------------------------------------------------------------------ #
#  Build trimesh scene (point cloud + frustums + trajectory)
# ------------------------------------------------------------------ #

CLIP_PALETTE = [
    (255,0,0,255), (0,255,0,255), (0,0,255,255), (255,255,0,255),
    (255,0,255,255), (0,255,255,255), (255,128,0,255), (128,0,255,255),
]


def export_glb(camera_poses, imgs, points_raw, frame_clip_idx, glb_path, point_mask=None):
    scene = trimesh.Scene()

    pts = points_raw.reshape(-1, 3)
    imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
    colors = imgs_np.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if point_mask is not None:
        valid &= point_mask.reshape(-1)
    pts, colors = pts[valid], colors[valid]
    rgba = np.hstack([(colors * 255).astype(np.uint8),
                      np.full((colors.shape[0], 1), 255, dtype=np.uint8)])

    if pts.shape[0] > 0:
        scene.add_geometry(trimesh.points.PointCloud(vertices=pts, colors=rgba))

    extent = np.ptp(pts, axis=0).max() if len(pts) else 0.0
    cam_scale = 0.05 * extent if extent > 0 else 0.1
    cam_centers = []
    for i, pose in enumerate(camera_poses):
        c_idx = int(frame_clip_idx[i])
        color = CLIP_PALETTE[c_idx % len(CLIP_PALETTE)]
        scene.add_geometry(create_camera_frustum(pose, scale=cam_scale, color=color))
        cam_centers.append(pose[:3, 3])

    if len(cam_centers) >= 2:
        centers = np.asarray(cam_centers)
        segs = np.stack([centers[:-1], centers[1:]], axis=1)
        path = trimesh.load_path(segs)
        for ent in path.entities:
            ent.color = (255, 255, 255, 255)
        scene.add_geometry(path)

    scene.export(str(glb_path))


# ------------------------------------------------------------------ #
#  Export per-clip pose JSON
# ------------------------------------------------------------------ #

def export_clip_poses(clip_ranges, camera_poses, all_image_paths,
                      video_folder_name, scene_name):
    for c_idx, (clip_dir, s, e) in enumerate(clip_ranges):
        clip_poses = camera_poses[s:e]
        clip_images = all_image_paths[s:e]
        out_name = f"{video_folder_name}_{scene_name}_{clip_dir.name}_poses.json"
        out_path = clip_dir / out_name

        frames_data = [{"image": p.name, "pose": pose.tolist()}
                       for pose, p in zip(clip_poses, clip_images)]
        payload = {
            "video_folder": video_folder_name,
            "scene": scene_name,
            "clip": clip_dir.name,
            "num_frames": len(frames_data),
            "frames": frames_data,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)


# ------------------------------------------------------------------ #
#  Collect images from clip directories
# ------------------------------------------------------------------ #

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def collect_clips(clip_dirs):
    all_image_paths = []
    clip_ranges = []
    for clip_dir in clip_dirs:
        imgs = sorted([p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
                      key=lambda p: p.name)
        if not imgs:
            continue
        start = len(all_image_paths)
        all_image_paths.extend(imgs)
        clip_ranges.append((clip_dir, start, len(all_image_paths)))
    return all_image_paths, clip_ranges


# ------------------------------------------------------------------ #
#  Process a scene (hierarchical: video_folder/scene/clip/*.jpg)
# ------------------------------------------------------------------ #

def process_scene(scene_dir, video_folder_name, model, device, conf_threshold):
    scene_name = scene_dir.name
    clip_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not clip_dirs:
        return

    all_image_paths, clip_ranges = collect_clips(clip_dirs)
    if not all_image_paths:
        return

    N = len(all_image_paths)
    imgs = load_images_with_target_resolution(all_image_paths, device=device)
    results = run_inference(model, imgs, device)

    camera_poses = results["camera_poses"][0].cpu().numpy()
    points = results["points"][0].cpu().numpy()

    frame_clip_idx = np.empty(N, dtype=np.int32)
    for c_idx, (_, s, e) in enumerate(clip_ranges):
        frame_clip_idx[s:e] = c_idx

    # Export GLB
    glb_path = scene_dir / f"{video_folder_name}_{scene_name}.glb"
    point_mask = (results["conf"][0, ..., 0].sigmoid() > conf_threshold).cpu().numpy()
    export_glb(camera_poses, imgs, points, frame_clip_idx, glb_path, point_mask)

    # Export per-clip poses
    export_clip_poses(clip_ranges, camera_poses, all_image_paths,
                      video_folder_name, scene_name)


# ------------------------------------------------------------------ #
#  Process flat structure (video_folder/clip/*.jpg, no scene layer)
# ------------------------------------------------------------------ #

def process_scene_flat(video_folder_dir, video_folder_name, clip_dirs,
                       model, device, conf_threshold):
    scene_name = "all_clips"

    all_image_paths, clip_ranges = collect_clips(clip_dirs)
    if not all_image_paths:
        return

    N = len(all_image_paths)
    imgs = load_images_with_target_resolution(all_image_paths, device=device)
    results = run_inference(model, imgs, device)

    camera_poses = results["camera_poses"][0].cpu().numpy()
    points = results["points"][0].cpu().numpy()

    frame_clip_idx = np.empty(N, dtype=np.int32)
    for c_idx, (_, s, e) in enumerate(clip_ranges):
        frame_clip_idx[s:e] = c_idx

    # Export GLB
    glb_path = video_folder_dir / f"{video_folder_name}.glb"
    point_mask = (results["conf"][0, ..., 0].sigmoid() > conf_threshold).cpu().numpy()
    export_glb(camera_poses, imgs, points, frame_clip_idx, glb_path, point_mask)

    # Export per-clip poses
    export_clip_poses(clip_ranges, camera_poses, all_image_paths,
                      video_folder_name, scene_name)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Pi3 multi-shot 3D reconstruction")
    parser.add_argument("--root-dir", type=str, default="outputs/sam2_scene_clips",
                        help="Root: video_folder/[scene/]clip/*.jpg")
    parser.add_argument("--ckpt", type=str, default="", help="Custom Pi3 checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 <= args.conf_threshold <= 1:
        parser.error("--conf-threshold must be between 0 and 1")

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    if args.ckpt:
        model = Pi3.from_pretrained(args.ckpt).to(device).eval()
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    video_folders = sorted([d for d in root_dir.iterdir() if d.is_dir()], key=lambda p: p.name)

    for vf in tqdm(video_folders, desc="Processing"):
        video_folder_name = vf.name
        subdirs = sorted([d for d in vf.iterdir() if d.is_dir()], key=lambda p: p.name)

        if not subdirs:
            continue

        # Detect structure: flat (clip/*.jpg) vs hierarchical (scene/clip/*.jpg)
        has_images = any(p.suffix.lower() in IMG_EXTS for sub in subdirs
                         for p in sub.iterdir() if p.is_file())

        if has_images:
            # Flat: subdirs are clips directly
            process_scene_flat(vf, video_folder_name, subdirs, model, device, args.conf_threshold)
        else:
            # Hierarchical: subdirs are scenes containing clips
            for scene_dir in subdirs:
                process_scene(scene_dir, video_folder_name, model, device, args.conf_threshold)

    print("Done.")


if __name__ == "__main__":
    main()
