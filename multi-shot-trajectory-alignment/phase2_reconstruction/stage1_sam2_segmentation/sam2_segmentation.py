#!/usr/bin/env python3
"""
Person detection, SAM 2 prompting and RLE encoding adapt the Grounded SAM 2
Hugging Face demo to sample frames from shot clips and mask people.
See README.md and third_party_licenses/Grounded-SAM-2.txt.

Output per video:
    output_dir/shot_name/
    ├── frame_0010_black_mask.jpg
    ├── frame_0020_black_mask.jpg
    └── ...
"""

import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import pycocotools.mask as mask_util
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 person segmentation for multi-shot videos")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="person.")
    parser.add_argument("--video-dir", required=True, help="Directory containing video files")
    parser.add_argument("--output-base-dir", required=True, help="Output directory")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to sample per video")
    parser.add_argument("--skip-first-frames", type=int, default=10, help="Frames to skip at the start")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l")
    parser.add_argument("--no-dump-json", action="store_true", help="Skip saving per-frame JSON results")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Models
# ------------------------------------------------------------------ #

def load_models(args, device):
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_pred = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)

    return sam2_pred, grounding, processor


# ------------------------------------------------------------------ #
#  Frame sampling
# ------------------------------------------------------------------ #

def get_sample_indices(total_frames: int, num_frames: int, skip_first: int) -> list:
    if total_frames <= skip_first:
        return []
    available = total_frames - skip_first
    if available < num_frames:
        return list(range(skip_first, total_frames))
    return np.linspace(skip_first, total_frames - 1, num_frames, dtype=int).tolist()


# ------------------------------------------------------------------ #
#  Single frame processing
# ------------------------------------------------------------------ #

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_frame(frame_bgr, frame_idx, sam2_pred, grounding, processor,
                  text_prompt, device, output_dir, dump_json):
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    sam2_pred.set_image(np.array(image))

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=0.4, text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    boxes = results[0]["boxes"].cpu().numpy()

    if len(boxes) == 0:
        masks = np.array([])
        scores = np.array([])
        class_names, confidences = [], []
    else:
        masks, scores, _ = sam2_pred.predict(
            point_coords=None, point_labels=None,
            box=boxes, multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["text_labels"]

    # Black out detected regions
    if len(masks) > 0:
        combined = np.any(masks, axis=0)[:, :, np.newaxis]
        out_frame = np.where(combined, 0, frame_bgr)
    else:
        out_frame = frame_bgr.copy()

    out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_black_mask.jpg")
    if not cv2.imwrite(out_path, out_frame):
        raise OSError(f"Failed to write {out_path}")

    if dump_json and len(masks) > 0:
        ann = [
            {
                "class_name": cn,
                "bbox": bx.tolist() if hasattr(bx, "tolist") else bx,
                "segmentation": single_mask_to_rle(m),
                "score": float(s),
            }
            for cn, bx, m, s in zip(class_names, boxes, masks, scores)
        ]
        json_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_results.json")
        with open(json_path, "w") as f:
            json.dump({
                "frame_index": frame_idx,
                "annotations": ann,
                "box_format": "xyxy",
                "img_width": image.width,
                "img_height": image.height,
            }, f, indent=2)


# ------------------------------------------------------------------ #
#  Video processing
# ------------------------------------------------------------------ #

def process_video(video_path, output_dir, sam2_pred, grounding, processor, args, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = get_sample_indices(total, args.num_frames, args.skip_first_frames)
    if not indices:
        cap.release()
        raise ValueError(f"No sample frames in {video_path}. Check skip_first_frames.")

    dump_json = not args.no_dump_json
    written = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            process_frame(frame, idx, sam2_pred, grounding, processor,
                          args.text_prompt, device, output_dir, dump_json)
            written += 1
        idx += 1

    cap.release()
    if written != len(indices):
        raise OSError(f"Decoded {written} of {len(indices)} sampled frames in {video_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    video_dir = Path(args.video_dir)
    output_base = Path(args.output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    sam2_pred, grounding, processor = load_models(args, device)

    manifest = video_dir / "trajectory.json"
    if manifest.is_file():
        videos = []
        for filename in json.loads(manifest.read_text())["shots"]:
            name = filename[:-4] if filename.lower().endswith(".mp4") else filename
            if not name or name.startswith(".") or Path(name).name != name:
                raise ValueError(f"Invalid shot filename: {filename!r}")
            videos.append(video_dir / name / f"{name}.mp4")
    else:
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        videos = sorted(v for v in video_dir.rglob("*") if v.suffix.lower() in exts
                        and not any(part.startswith(".") for part in v.relative_to(video_dir).parts))

    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    for video_file in videos:
        rel = video_file.relative_to(video_dir)
        if rel.parent != Path("."):
            out_dir = output_base / rel.parent.name
        else:
            out_dir = output_base / video_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        process_video(video_file, out_dir, sam2_pred, grounding, processor, args, device)

    print(f"Done. {len(videos)} videos processed -> {output_base}")


if __name__ == "__main__":
    main()
