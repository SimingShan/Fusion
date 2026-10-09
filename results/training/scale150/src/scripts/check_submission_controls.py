#!/usr/bin/env python3
"""Offline checks for automatic submission evidence and public metadata guards."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.periodic_submission import candidate_evidence, digest, submission_priority, preserves_best_diii_d


class SubmissionControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.checkpoints = []
        for name in ("a.pt", "b.pt"):
            path = self.run / name
            path.write_bytes(name.encode())
            self.checkpoints.append({"frozen_file": str(path), "sha256": digest(path)})
        reports = {}
        for key, value in (("native", .996), ("baseline_native", .995), ("legacy", .9935), ("baseline_legacy", .993)):
            split = "legacy_val" if "legacy" in key else "val"
            report = {"S": value, "evaluation": {"type": "local_held_out_validation", "split": split,
                "sampled": False, "evaluated_shots": 128, "evaluated_frames": 28000,
                "scorer_sha256": "fixed", "data_directory": "same"}}
            path = self.run / f"{key}.json"
            path.write_text(json.dumps(report))
            reports[key] = {"path": str(path), "sha256": digest(path)}
        with zipfile.ZipFile(self.run / "submission.zip", "w") as z:
            z.writestr("diii_d_public_test.npz", b"test fixture")
        self.decision = {"decision": "accept", "test_labels_read": False, "audit_labels_read": False,
            "public_name": "Submission 20260920-123", "validation": reports,
            "checkpoints": self.checkpoints, "primary_weight": .5, "gate_threshold": .1}
        self.save("decision.json", self.decision)
        self.save("validation.json", {"status": "passed", "submission_zip": str(self.run / "submission.zip"),
            "zip_sha256": digest(self.run / "submission.zip"), "input_sha256": "same-predictions"})
        self.save("export.json", {"status": "complete", "primary_weight": .5, "gate_threshold": .1,
            "source_checkpoint_sha256": dict(zip(("primary", "secondary"), (x["sha256"] for x in self.checkpoints))),
            "prediction_sha256": "same-predictions"})
        self.save("export_parity.json", {"passed": True})

    def save(self, name, data):
        (self.run / name).write_text(json.dumps(data))

    def test_valid_evidence(self):
        self.assertEqual(candidate_evidence(self.run)["native"], .996)

    def test_checkpoint_tamper(self):
        (self.run / "a.pt").write_bytes(b"changed")
        with self.assertRaises(ValueError): candidate_evidence(self.run)

    def test_archive_tamper(self):
        with (self.run / "submission.zip").open("ab") as f: f.write(b"changed")
        with self.assertRaises(ValueError): candidate_evidence(self.run)

    def test_public_method_title_rejected(self):
        self.decision["public_name"] = "Private implementation title"
        self.save("decision.json", self.decision)
        with self.assertRaises(ValueError): candidate_evidence(self.run)

    def test_nonfinite_score_rejected(self):
        path = self.run / "native.json"
        value = json.loads(path.read_text())
        value["S"] = float("nan")
        path.write_text(json.dumps(value))
        self.decision["validation"]["native"]["sha256"] = digest(path)
        self.save("decision.json", self.decision)
        with self.assertRaises(ValueError): candidate_evidence(self.run)

    def test_missing_parity_waits(self):
        (self.run / "export_parity.json").unlink()
        self.assertIsNone(candidate_evidence(self.run))

    def test_prior_posts_precede_new_candidates(self):
        new = submission_priority(self.run, {})
        self.assertLess(submission_priority(self.run, {"submission_id": 1}), new)
        self.assertLess(submission_priority(self.run, {"submission_post_started": True}), new)
        self.assertLess(submission_priority(self.run, {"dataset_post_started": True}), new)

    def test_stronger_candidates_precede_weaker_candidates(self):
        self.decision["validation"]["native"]["S"] = .996
        self.save("decision.json", self.decision)
        strong = submission_priority(self.run, {})
        self.decision["validation"]["native"]["S"] = .995
        self.save("decision.json", self.decision)
        self.assertLess(strong, submission_priority(self.run, {}))

    def test_different_validation_selection_rejected(self):
        path = self.run / "native.json"
        value = json.loads(path.read_text())
        value["evaluation"]["evaluated_frames"] = 100
        path.write_text(json.dumps(value))
        self.decision["validation"]["native"]["sha256"] = digest(path)
        self.save("decision.json", self.decision)
        with self.assertRaises(ValueError): candidate_evidence(self.run)

    def test_uploader_rejects_title_before_authentication(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/codabench_submit.py"),
            "--name", "Private implementation title"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("neutral numeric identifier", result.stderr)

    def verified_state(self):
        return {"entries": {self.run.name: {"verified_success": True, "submission_id": 7,
                    "sha256": digest(self.run / "submission.zip")}},
                "historical_best": {"d3d_S": {"submission_id": 7, "score": .996}}}

    def test_transfer_keeps_verified_prediction(self):
        state = self.verified_state()
        self.assertTrue(preserves_best_diii_d(self.run.parent, state, "same-predictions"))
        self.assertFalse(preserves_best_diii_d(self.run.parent, state, "older-predictions"))

    def test_transfer_requires_verified_reference(self):
        state = self.verified_state()
        state["entries"][self.run.name]["verified_success"] = False
        with self.assertRaises(ValueError): preserves_best_diii_d(self.run.parent, state, "same-predictions")

    def test_changed_official_reference_rejected(self):
        state = self.verified_state()
        self.save("export.json", {"status": "complete", "prediction_sha256": "changed"})
        with self.assertRaises(ValueError): preserves_best_diii_d(self.run.parent, state, "changed")

    def test_transfer_reference_uses_official_best(self):
        state = self.verified_state()
        state["entries"][self.run.name]["native"] = .995
        state["entries"]["another"] = {"verified_success": True, "submission_id": 8, "native": .999}
        self.assertTrue(preserves_best_diii_d(self.run.parent, state, "same-predictions"))


if __name__ == "__main__":
    unittest.main()
