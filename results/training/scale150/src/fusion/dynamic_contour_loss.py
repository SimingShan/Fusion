"""Experimental scorer-consistent current-prediction kappa value and gradient.

CPU workers refresh all discrete choices on the current field. Only that local
derivative is attached to the GPU coefficient graph. Failures retain an explicit
mean-substitution penalty and are counted; they are never silently dropped.
"""
from collections import Counter
import atexit
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch

from .contour_replay import trace, replay, shape

_WORKER = {}


def _init_worker(decoder, grid_path, gradient_cap):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    grid = np.load(grid_path)
    _WORKER.update(decoder=np.asarray(decoder, dtype=np.float64),
                   R=grid["grid_R"], Z=grid["grid_Z"], mask=grid["mask_coarse"].astype(bool),
                   gradient_cap=float(gradient_cap))


def _evaluate_field(field):
    try:
        w = _WORKER
        if not np.isfinite(field).all():
            raise ValueError("nonfinite_field")
        tr = trace(field, w["R"], w["Z"], w["mask"])
        pt = torch.tensor(field, dtype=torch.float64, requires_grad=True)
        value = shape(replay(pt, tr))[0]
        gradient = torch.autograd.grad(value, pt)[0].numpy().ravel()
        coefficient_gradient = w["decoder"] @ gradient
        scalar = float(value.detach())
        # This is the scorer's exact kappa plausibility check. Such predictions
        # fail extraction; they do not become smaller/cleaner training samples.
        if not np.isfinite(scalar) or not .8 <= scalar <= 4.0:
            raise ValueError("implausible_kappa")
        if not np.isfinite(coefficient_gradient).all():
            raise ValueError("nonfinite_gradient")
        norm = float(np.linalg.norm(coefficient_gradient))
        factor = min(1.0, w["gradient_cap"] / max(norm, 1e-30))
        return {"ok": True, "value": scalar, "gradient": coefficient_gradient * factor,
                "gradient_norm": norm, "clipped": factor < 1, "event": tr.event_kind}
    except (ValueError, IndexError, RuntimeError, FloatingPointError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


class DynamicKappaLoss:
    def __init__(self, norm, target_dir, grid_path, workers=2, gradient_cap=1.0,
                 status_path=None, failure_limit=.01, failure_min_attempts=1000):
        target_dir = Path(target_dir)
        self.metadata = json.loads((target_dir / "metadata.json").read_text())
        if self.metadata["source_split"] != "train" or not self.metadata["raw_psi_verified"]:
            raise ValueError("Only verified raw-training kappa targets are permitted")
        self.targets = np.load(target_dir / "train_kappa.npy", mmap_mode="r")
        self.variance = self.metadata["kappa_variance"]
        self.mean = self.metadata["kappa_mean"]
        if not self.variance > 0 or workers < 1 or gradient_cap <= 0:
            raise ValueError("Invalid dynamic kappa configuration")
        decoder = np.asarray(norm["coefficient_std"], dtype=float)[:,None] * np.asarray(norm["basis"], dtype=float)
        # Spawn avoids inheriting a live CUDA context; workers remain alive across
        # optimizer batches and use no RNG from the training process.
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_init_worker,
                                        initargs=(decoder, str(grid_path), gradient_cap))
        self.status_path = None if status_path is None else Path(status_path)
        self.failure_limit = failure_limit
        self.failure_min_attempts = failure_min_attempts
        self.counts = Counter()
        self.errors = Counter()
        self.failure_examples = []
        self.worker_seconds = 0.0
        self.max_gradient_norm = 0.0
        self._closed = False
        atexit.register(self.close)

    def __call__(self, coefficients, fields, indices, frame_weights=None, batch_weight_mean=None):
        global_indices = indices.detach().cpu().numpy()
        target = np.asarray(self.targets[global_indices])
        valid = np.isfinite(target)
        self.counts["selected"] += len(target)
        self.counts["invalid_target"] += int((~valid).sum())
        if not valid.any():
            return coefficients.sum() * 0
        selected = np.flatnonzero(valid)
        cpu_fields = fields.detach().double().cpu().numpy()[selected]
        start = time.monotonic()
        results = list(self.pool.map(_evaluate_field, cpu_fields, chunksize=1))
        self.worker_seconds += time.monotonic() - start
        values = np.full(len(selected), self.mean)
        gradients = np.zeros((len(selected), coefficients.shape[1]))
        for j, result in enumerate(results):
            self.counts["attempts"] += 1
            if result["ok"]:
                values[j], gradients[j] = result["value"], result["gradient"]
                self.counts["successes"] += 1
                self.counts["clipped"] += int(result["clipped"])
                self.counts[result["event"]] += 1
                self.max_gradient_norm = max(self.max_gradient_norm, result["gradient_norm"])
            else:
                self.counts["failures"] += 1
                self.errors[result["error"]] += 1
                if len(self.failure_examples) < 20:
                    self.failure_examples.append({"global_index":int(global_indices[selected[j]]),"error":result["error"]})
        if (self.counts["attempts"] >= self.failure_min_attempts and
                self.counts["failures"] / self.counts["attempts"] > self.failure_limit):
            self.write_status()
            raise RuntimeError(f"Dynamic kappa failure fraction exceeds {self.failure_limit}: {self.counts}")
        position = torch.as_tensor(selected, dtype=torch.long, device=coefficients.device)
        c = coefficients[position]
        g = torch.as_tensor(gradients, dtype=c.dtype, device=c.device)
        exact_forward = torch.as_tensor(values, dtype=c.dtype, device=c.device)
        local = exact_forward + ((c - c.detach()) * g).sum(1)
        truth = torch.as_tensor(target[selected], dtype=c.dtype, device=c.device)
        terms = (local - truth).square() / self.variance
        if frame_weights is not None:
            # Normalize by the FULL batch's mean amplitude weight. The first
            # eight shuffled examples then estimate its weighted mean without
            # the ratio bias of renormalizing eight weights independently.
            terms = terms * frame_weights[position] / batch_weight_mean
        return terms.sum() / len(target)

    def status(self):
        return {"counts":dict(self.counts),"errors":dict(self.errors),
                "failure_examples":self.failure_examples,"worker_seconds":self.worker_seconds,
                "max_unclipped_gradient_norm":self.max_gradient_norm,
                "target_variance_train_only":self.variance}

    def write_status(self):
        if self.status_path is not None:
            self.status_path.write_text(json.dumps(self.status(),indent=2)+"\n")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.write_status()
        self.pool.shutdown(wait=True, cancel_futures=True)
