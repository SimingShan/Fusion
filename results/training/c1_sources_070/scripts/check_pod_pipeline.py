#!/usr/bin/env python3
"""Small regression checks for POD physical units, missing labels, and splits."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from fusion.pod import validate_splits
from scripts.train_pod import batch_loss, loss_matrix, reconstruct, torch_normalization, validation


class PODIntegrity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        rng = np.random.default_rng(17)
        q, _ = np.linalg.qr(rng.normal(size=(4225, 4)))
        self.norm = {"basis": q.T.astype(np.float32), "coefficient_std": np.array([3., 1., .1, .01], dtype=np.float32),
                     "psi_mean": rng.normal(0, .02, (65, 65)).astype(np.float32), "psi_std": np.float32(.2)}
        self.tnorm = torch_normalization(self.norm, "cpu")
        self.model = build_advanced_model("pod_deeponet", 5, 4, width=16, depth=1)
        self.Q = loss_matrix(self.tnorm, 1., .25)

    def test_physical_reconstruction_and_zero_fourier_transfer(self):
        fourier = build_advanced_model("pod_fourier_deeponet", 5, 4, width=16, depth=1, spectral_width=4, spectral_layers=1, modes=2)
        with torch.no_grad():
            self.model.linear_coefficients.weight.normal_()
        fourier.load_state_dict(self.model.state_dict(), strict=False)
        x = torch.randn(3, 5)
        outputs = self.model(x)
        expected = ((outputs["coefficients"].detach().numpy() * self.norm["coefficient_std"]) @ self.norm["basis"]).reshape(-1, 65, 65) + self.norm["psi_mean"]
        for model in (self.model, fourier):
            actual = reconstruct(model, model(x), self.tnorm)
            np.testing.assert_allclose(actual.detach(), expected, atol=1e-7)
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(model.linear_coefficients.weight.grad).all())
            self.assertGreater(float(model.linear_coefficients.weight.grad.abs().sum()), 0)

    def test_missing_scalar_masks_are_finite_and_batch_independent(self):
        arrays = {"X": torch.randn(5, 5), "coefficients": torch.randn(5, 4),
                  "scalars": torch.tensor([[1., 2.], [float("nan"), 3.], [float("inf"), float("-inf")], [4., 5.], [6., float("nan")]])}
        a = validation(self.model, arrays, self.tnorm, self.Q, .25, 2)
        b = validation(self.model, arrays, self.tnorm, self.Q, .25, 5)
        self.assertAlmostEqual(a, b, places=6)
        loss = batch_loss(self.model, arrays, self.tnorm, self.Q, .25)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_train_audit_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"train": [{"shot_index": 1}], "val": [{"shot_index": 2}], "audit": [{"shot_index": 1}]}
            Path(directory, "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                validate_splits(directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
