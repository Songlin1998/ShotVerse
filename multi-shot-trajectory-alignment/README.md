# Multi-shot trajectory alignment

Estimate a camera trajectory for each shot in a video, then express every frame in one scene-wide coordinate system. Each output pose is a 4 × 4 camera-to-world matrix. Scale is relative to the joint Pi3 reconstruction.

## Run on a scene

Follow the [environment setup](environments/README.md), then run these commands from this directory with the `pi3` environment active. The launcher uses Python and PyYAML before switching environments.

```bash
cp configs/pipeline_config.yaml.example configs/pipeline_config.yaml
# Set input_root and output_root in the new file.
python tools/check_environment.py
bash scripts/run_full_pipeline.sh
```

Place each video at `data/input_scenes/scene_XXXX/camera_1_recon.mp4`. Its sibling `trajectory.json` lists the shot filenames in playback order and their zero-based starting frame indices. An optional final boundary is the exclusive end. For a 100-frame, two-shot video:

```json
{"shots": ["shot_001.mp4", "shot_002.mp4"], "shot_cut": [0, 48, 100]}
```

| Step | Operation | Output under `data/output_scenes/scene_XXXX/` |
| --- | --- | --- |
| Phase 1 | Split with FFmpeg and run Pi3 on each shot | Per-shot MP4 and full-frame world-to-camera poses |
| Phase 2, stage 1 | Detect and mask people on sampled frames | `stage1_sam2/` |
| Stage 2 | Reconstruct all sampled shots together with Pi3 | `stage2_pi3/` with per-shot poses and a GLB scene |
| Stage 3 | Anchor each trajectory to its first sample and fit scale across all matched samples | `stage3_aligned/` |
| Stage 4 | Concatenate the aligned frames in shot order | `stage4_merged/scene_XXXX_merged_all_poses.json` |

The default sampler skips the first 10 frames of each shot and takes up to five of the remaining frames. Set `sampling.skip_first_frames` below the length of your shortest shot. At least two sampled frames with distinct camera positions are needed to estimate relative scale from translation. When both sampled trajectories have no measurable translation, alignment keeps unit scale and records `scale_estimated: false`. A measurable translation in only one trajectory raises an error. The `pi3.conf_threshold` setting filters point-cloud vertices by their Pi3 confidence probability.

To resume from split clips and per-shot pose files, run `bash phase2_reconstruction/run.sh`. Use `--input DIR` on that script for a separate directory of prepared scenes. Run `bash phase2_reconstruction/run.sh --force` to recompute cached stages after changing sampling, reconstruction settings, or the alignment implementation. Use a fresh output directory after replacing the source video.

## Acknowledgements

We thank the authors and maintainers of the following projects:

- **Pi3** provides both the per-shot poses and the joint multi-shot reconstruction. Stage 2 adapts its image-sizing routine for the sampled frames. [Code](https://github.com/yyfz/Pi3) · [Original image loader](https://github.com/yyfz/Pi3/blob/a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99/pi3/utils/basic.py) · [Model weights](https://huggingface.co/yyfz233/Pi3) · [License](third_party_licenses/Pi3.txt)
- **Grounded SAM 2** provides the reference implementation for combining text-conditioned detections and segmentation. Stage 1 adapts its Hugging Face demo, including the mask encoding, to process frames from each shot. [Project](https://github.com/IDEA-Research/Grounded-SAM-2) · [Reference demo](https://github.com/IDEA-Research/Grounded-SAM-2/blob/b7a9c29f196edff0eb54dbe14588d7ae5e3dde28/grounded_sam2_hf_model_demo.py) · [License](third_party_licenses/Grounded-SAM-2.txt)
- **Grounding DINO** locates people from the text prompt. **SAM 2** converts the detected boxes into masks. [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) · [Detector weights](https://huggingface.co/IDEA-Research/grounding-dino-tiny) · [SAM 2](https://github.com/facebookresearch/sam2)

For the ShotVerse paper and citation, see the [repository README](../README.md).

## License

The ShotVerse alignment code is licensed under MIT. Adapted upstream portions retain the notices linked above.
