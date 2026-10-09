#!/usr/bin/env python3
"""Small CPU checks for leakage, Ip correction, operator gradients, and export.

Run with the project Python environment. --data-dir additionally checks the real
prepared train/validation split; no raw shots or training jobs are loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.data import extract_features, feature_names, interpolate_signal
from fusion.models import SpectralConv2d, build_model
from scripts.train import fit_normalization, normalized, prediction_entries, validate_shot_split, validation, write_predictions


class InputIntegrity(unittest.TestCase):
    @staticmethod
    def row():
        return {
            "efit_times": np.array([0., 500., 1000.]),
            "magnetics_time": np.array([-1000., 0., 1000., 2000.]),
            "magnetics_plasma_current_times": np.array([3000., 4000., 5000., 6000.]),
            "magnetics_plasma_current": np.array([0., 1.e6, 1.e6, 0.]),
            "magnetics_F1A": np.array([1., 2., 3., 4.]),
            "thomson_core_times": np.array([0., 1000.]),
            "thomson_core_Te": np.array([[2., 4.], [4., 6.]]),
        }

    def test_features_independent_of_all_target_columns(self):
        inputs = self.row()
        expected = extract_features(inputs)
        poisoned = copy.deepcopy(inputs)
        for column in ("efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis",
                       "efit_z_axis", "efit_boundary", "efit_pressure", "efit_fpol"):
            poisoned[column] = np.full((3, 65, 65), np.nan)
        poisoned["shot_id"] = "must-not-be-an-input"
        np.testing.assert_array_equal(expected, extract_features(poisoned))
        selected = extract_features(poisoned, np.array([2, 0]))
        np.testing.assert_array_equal(selected, expected[[2, 0]])
        self.assertTrue(np.isfinite(expected).all())
        self.assertEqual(expected.shape, (3, len(feature_names())))

    def test_ip_erratum_shifts_only_affected_axis_and_preserves_current_sign(self):
        row = self.row()
        names = feature_names()
        current = names.index("magnetics_plasma_current")
        np.testing.assert_allclose(extract_features(row)[:, current], 1.e6)
        row["magnetics_plasma_current_times"] = np.array([-1000., 0., 1000., 2000.])
        row["magnetics_time"] = np.array([-4000., -3000., -2000., -1000.])
        row["magnetics_plasma_current"] *= -1
        # Correct timestamps must be retained even when the shared axis differs.
        np.testing.assert_allclose(extract_features(row)[:, current], -1.e6)

    def test_interpolation_marks_missing_coverage(self):
        values, observed = interpolate_signal(
            [2., 1., 1., np.nan, 0.], [20., 10., 99., 5., 0.],
            np.array([-1., .5, 1., 3.]),
        )
        np.testing.assert_allclose(values, [0., 5., 10., 20.])
        np.testing.assert_array_equal(observed, [0., 1., 1., 0.])


class OperatorIntegrity(unittest.TestCase):
    def test_both_operator_encoders_receive_finite_gradients(self):
        for name in ("deeponet", "fourier_deeponet"):
            with self.subTest(model=name):
                torch.manual_seed(13)
                model = build_model(name, 5, width=12, latent_dim=8, depth=2,
                                    grid_shape=(9, 11), spectral_width=4,
                                    spectral_layers=2, modes=3, padding=1)
                psi, scalars = model(torch.randn(2, 5))
                self.assertEqual(tuple(psi.shape), (2, 9, 11))
                self.assertEqual(tuple(scalars.shape), (2, 2))
                (psi.square().mean() + scalars.square().mean()).backward()
                for module in (model.branch, model.trunk):
                    magnitude = sum(p.grad.abs().sum().item() for p in module.parameters())
                    self.assertGreater(magnitude, 0.)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_spectral_decoder_transmits_information_to_distant_grid_points(self):
        layer = SpectralConv2d(channels=1, modes=2)
        with torch.no_grad():
            layer.weight_positive.zero_()
            layer.weight_negative.zero_()
            layer.weight_positive[0, 0, 0, 0, 0] = 1.
        impulse = torch.zeros(1, 1, 8, 8)
        impulse[0, 0, 0, 0] = 1.
        # Keeping only the DC Fourier coefficient returns the global mean at
        # every pixel, including points far from the impulse.
        torch.testing.assert_close(layer(impulse), torch.full_like(impulse, 1./64.))


class TrainingIntegrity(unittest.TestCase):
    def test_public_test_export_requires_complete_canonical_shot_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = [{"path": f"test{index}.npz", "shot_index": index,
                         "n_frames": 2, "raw_frames": 2} for index in range(2)]
            for entry in selected:
                (root/entry["path"]).touch()
            manifest = {"expected_shots": {"diii_d_public_test": 2}, "test": selected[:1]}
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Incomplete public-test cache"):
                prediction_entries(root, "test")
            manifest["test"] = selected[::-1]
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "canonical order"):
                prediction_entries(root, "test")
            manifest["test"] = selected
            (root/"manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(len(prediction_entries(root, "test")), 2)
            manifest["test"][0]["n_frames"] = 1
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "every frame"):
                prediction_entries(root, "test")

    def test_nonfinite_scalar_labels_are_masked_before_loss(self):
        train = {"X": np.array([[0.], [2.], [4.]], dtype=np.float32),
                 "psi": np.zeros((3, 3, 3), dtype=np.float32),
                 "scalars": np.array([[1., 2.], [3., np.nan], [np.inf, 6.]], dtype=np.float32)}
        norm = fit_normalization(train)
        np.testing.assert_allclose(norm["x_mean"], [2.])
        np.testing.assert_allclose(norm["scalar_mean"], [2., 4.])
        arrays = normalized(train, norm, torch.device("cpu"))
        self.assertTrue(torch.isfinite(arrays[2]).all())
        self.assertEqual(int(arrays[3].sum()), 4)

    def test_validation_missing_label_weighting_independent_of_batch_size(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), 3, 3), torch.ones(len(x), 2)

        arrays = (torch.zeros(3, 1), torch.zeros(3, 3, 3),
                  torch.tensor([[0., 0.], [3., 0.], [5., 0.]]),
                  torch.tensor([[True, True], [True, False], [True, False]]))
        self.assertAlmostEqual(validation(Constant(), arrays, 1, 1.), .15*22./4.)
        self.assertAlmostEqual(validation(Constant(), arrays, 3, 1.), .15*22./4.)

    def test_shot_holdout_rejects_duplicate_identity_across_distinct_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"train": [{"path": "train/a.npz", "shot_id": "shot123", "shot_index": 1}],
                        "val": [{"path": "val/b.npz", "shot_id": "shot123", "shot_index": 2}]}
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "shot_id overlap"):
                validate_shot_split(root)
            manifest["val"][0]["shot_id"] = "shot456"
            (root/"manifest.json").write_text(json.dumps(manifest))
            validate_shot_split(root)

    def test_submission_denormalization_and_shot_order(self):
        class Known(torch.nn.Module):
            def forward(self, x):
                return x[:, :1, None].expand(-1, 65, 65), torch.cat((x, 2*x), dim=1)

        norm = {"x_mean": np.array([0.], np.float32), "x_std": np.array([1.], np.float32),
                "psi_mean": np.float32(2.), "psi_std": np.float32(3.),
                "scalar_mean": np.array([4., 5.], np.float32),
                "scalar_std": np.array([2., 3.], np.float32)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = []
            expected = []
            for shot_id, values in enumerate(([1., 2.], [3.])):
                x = np.asarray(values, np.float32)[:, None]
                psi = np.broadcast_to(x[:, :1, None]*3+2, (len(x), 65, 65))
                scalars = np.concatenate((x, 2*x), axis=1)*norm["scalar_std"]+norm["scalar_mean"]
                path = f"shot{shot_id}.npz"
                np.savez(root/path, X=x, psi=psi, scalars=scalars)
                selected.append({"path": path})
                expected.append((psi, scalars))
            (root/"manifest.json").write_text(json.dumps({"val": selected}))
            output = root/"submission.npz"
            metrics = write_predictions(Known(), norm, root, "val", output, 1, torch.device("cpu"))
            self.assertEqual(metrics["frames"], 3)
            self.assertEqual(metrics["r2_psi"], 1.)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(len(archive.files), 6)
                for index, (psi, scalars) in enumerate(expected):
                    for key, target in (("psirz", psi), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                        actual = archive[f"shot_{index:04d}_{key}"]
                        self.assertEqual(actual.dtype, np.float32)
                        np.testing.assert_allclose(actual, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.data_dir:
        validate_shot_split(args.data_dir)
        print(f"Real prepared training/validation shot identities are disjoint: {args.data_dir}")


if __name__ == "__main__":
    main()
