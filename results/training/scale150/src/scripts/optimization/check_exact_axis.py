#!/usr/bin/env python3
"""Check differentiable axis extraction against the official functional."""
from pathlib import Path
import argparse
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.geometry_loss import GeometryTargetStore
from fusion.pod import load_arrays
from scripts.codabench_submit import write_json
from starter import local_score as official
from derive import magnetic_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    store = GeometryTargetStore(ROOT / "data/geometry_history_v4")
    module = store.exact_axis_loss_module()
    indices = store.arrays["global_frame_indices"][store.arrays["valid"]]
    indices = indices[np.linspace(0, len(indices)-1, 128).astype(int)]
    fields = np.array(load_arrays(ROOT / "data/advanced_history_v2", "train", ("psi",))["psi"][indices])
    positions, targets = store.batch(indices)
    if len(positions) != len(indices):
        raise ValueError("Parity check requires valid training references")
    R, Z, mask = store.R, store.Z, module.mask.numpy()
    monotone = np.broadcast_to(R[None]+Z[:, None], fields.shape[1:]).astype(np.float32)
    flat = np.ones_like(monotone)
    combined = np.concatenate((fields, monotone[None], flat[None]))
    reference = np.array([magnetic_axis(x, R, Z, official.MACHINE, mask)[:2] for x in combined])
    predicted = module.predict(torch.from_numpy(combined)).detach().numpy()
    np.testing.assert_allclose(predicted, reference, atol=1e-12, rtol=1e-12)
    exact = module(torch.from_numpy(fields), targets)
    if float(exact) > 1e-20:
        raise ValueError("Reference field has nonzero supervised axis loss")
    shifted = module(torch.from_numpy(fields).double()+.125, targets)
    if float(shifted) > 1e-20:
        raise ValueError("Axis supervision lost flux-gauge invariance")
    perturbation = np.broadcast_to(R[None]**2, fields.shape[1:]).copy()
    perturbed = (torch.from_numpy(fields).double()+1e-3*torch.from_numpy(perturbation)).requires_grad_(True)
    loss = module(perturbed, targets)
    loss.backward()
    if not torch.isfinite(perturbed.grad).all() or not torch.any(perturbed.grad != 0):
        raise ValueError("Axis supervision requires finite nonzero gradients away from truth")
    small = perturbed[:1].detach().clone().requires_grad_(True)
    gradcheck = torch.autograd.gradcheck(lambda x: module.predict(x), (small,),
                                       eps=1e-7, atol=1e-5, rtol=1e-4, fast_mode=True)
    report = {"passed": True, "training_frames": len(fields), "special_fallback_fields": 2,
              "maximum_axis_error": float(np.max(np.abs(predicted-reference))),
              "reference_loss": float(exact), "gauge_invariance": True,
              "finite_nonzero_gradient": True, "numerical_gradient_check": bool(gradcheck),
              "audit_labels_read": False, "test_labels_read": False}
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
