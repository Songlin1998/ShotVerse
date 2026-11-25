#!/usr/bin/env python3
"""Verify project setup."""

import subprocess, os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
errors = 0


def check(name, ok, hint=""):
    global errors
    status = "ok" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        if hint:
            print(f"         -> {hint}")
        errors += 1


def section(title):
    print(f"\n--- {title} ---")


def main():
    global errors

    print("=" * 50)
    print("  Environment Check")
    print("=" * 50)

    # Config
    section("Config")
    config = os.path.join(PROJECT_ROOT, "configs", "pipeline_config.yaml")
    check("pipeline_config.yaml", os.path.isfile(config),
          "cp configs/pipeline_config.yaml.example configs/pipeline_config.yaml")

    # Conda envs
    section("Conda Environments")
    r = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    for env in ["gd-sam2", "pi3"]:
        found = any(env in l.split()[0] for l in r.stdout.strip().split("\n")
                    if l.strip() and not l.startswith("#"))
        check(f"conda env: {env}", found)

    # Scripts (4 stages, no stage 5)
    section("Core Scripts")
    scripts = [
        "phase1_shot_splitting/pipeline_shot_trajectory.py",
        "phase2_reconstruction/stage1_sam2_segmentation/sam2_segmentation.py",
        "phase2_reconstruction/stage2_pi3_reconstruction/pi3_reconstruction.py",
        "phase2_reconstruction/stage3_pose_alignment/align_all_poses.py",
        "phase2_reconstruction/stage4_trajectory_merging/merge_trajectories.py",
    ]
    for s in scripts:
        check(s, os.path.isfile(os.path.join(PROJECT_ROOT, s)))

    # Import check
    section("Python Imports (current env)")
    r = subprocess.run([sys.executable, "-c", "from pi3.models.pi3 import Pi3"],
                       capture_output=True, text=True)
    check("import pi3", r.returncode == 0, "pip install -e external/Pi3")

    # GPU
    section("GPU")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free",
                            "--format=csv,noheader"], capture_output=True, text=True)
        check("nvidia-smi", r.returncode == 0, r.stdout.strip())
    except FileNotFoundError:
        check("nvidia-smi", False, "not found")

    # External deps
    section("External Dependencies")
    if os.path.isfile(config):
        try:
            import yaml
            with open(config) as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            sam2 = paths.get("sam2_repo", "")
            if sam2:
                check("Grounded-SAM-2 repo", os.path.isdir(sam2), sam2)
            ckpt = paths.get("sam2_checkpoint", "")
            if ckpt:
                check("SAM2 checkpoint", os.path.isfile(ckpt), ckpt)
        except ImportError:
            print("  [skip] PyYAML not available")

    # Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("  All checks passed.")
    else:
        print(f"  {errors} issue(s) found.")
    print("=" * 50)
    return errors


if __name__ == "__main__":
    sys.exit(main())
