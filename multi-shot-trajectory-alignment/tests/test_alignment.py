"""CPU regression tests for shot manifests and camera alignment."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
shot_ranges = runpy.run_path(str(ROOT / "phase1_shot_splitting/pipeline_shot_trajectory.py"))["shot_ranges"]
align = runpy.run_path(str(ROOT / "phase2_reconstruction/stage3_pose_alignment/align_all_poses.py"))["align_single_shot"]


class ManifestTests(unittest.TestCase):
    def test_optional_end(self):
        for cuts in ([0, 10], [0, 10, 20]):
            self.assertEqual(shot_ranges({"shots": ["a.mp4", "b.mp4"], "shot_cut": cuts}, 20),
                             [("a", 0, 10), ("b", 10, 20)])

    def test_invalid_inputs(self):
        cases = [(["a", "A.mp4"], [0, 10]), (["a", "b"], [0]), (["a", "b"], [0, 0]),
                 (["a"], [0, 21]), (["../a.mp4"], [0]), (["stage2_pi3"], [0])]
        for names, cuts in cases:
            with self.subTest(shots=names, cuts=cuts), self.assertRaises(ValueError):
                shot_ranges({"shots": names, "shot_cut": cuts}, 20)


class AlignmentTests(unittest.TestCase):
    def aligned(self, local_x, joint_x):
        rigid = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]])
        full, sparse, expected = {}, [], []
        for i, (x, y) in enumerate(zip(local_x, joint_x)):
            key = f"frame_{i:04d}"
            pose = np.eye(4)
            pose[0, 3] = x
            full[key] = np.linalg.inv(pose).tolist()
            pose[0, 3] = y
            expected.append(rigid @ pose)
            sparse.append({"image": key + ".jpg", "pose": expected[-1].tolist()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "full.json").write_text(json.dumps({"frames": full}))
            target = root / "aligned.json"
            align(root / "sample_poses.json", root / "full.json", target, {"clip": "shot", "frames": sparse})
            result = json.loads(target.read_text())
        np.testing.assert_allclose(list(result["frames"].values()), expected, atol=1e-6)
        return result

    def test_round_trip(self):
        result = self.aligned([0, 1, 0], [0, 3, 0])
        self.assertTrue(result["scale_estimated"])
        self.assertAlmostEqual(result["scale_factor"], 3)

    def test_stationary_samples(self):
        result = self.aligned([0, 0], [0, 0])
        self.assertFalse(result["scale_estimated"])
        self.assertEqual(result["scale_factor"], 1)

    def test_inconsistent_motion(self):
        with self.assertRaisesRegex(ValueError, "Only one trajectory"):
            self.aligned([0, 0], [0, 3])


if __name__ == "__main__":
    unittest.main()
