"""Input-only missing-diagnostic augmentation for operator training."""
from __future__ import annotations

import numpy as np
import torch


def thomson_dropout_plan(feature_names, norm, device):
    selected_names = [feature_names[i] for i in norm["feature_indices"]]
    plans = []
    for system in ("core", "edge"):
        columns = np.asarray([i for i, name in enumerate(selected_names)
                              if name.startswith((f"thomson_{system}_Te_", f"thomson_{system}_ne_"))], dtype=np.int64)
        if len(columns):
            missing = np.clip(-norm["x_mean"][columns] / norm["x_std"][columns], -12, 12).astype(np.float32)
            plans.append((torch.as_tensor(columns, device=device), torch.as_tensor(missing, device=device)))
    return plans


def drop_thomson_modalities(X, plans, probability):
    """Independently remove core/edge values and masks; retain current/geometry."""
    if probability == 0:
        return X
    result = X.clone()
    for columns, missing in plans:
        selected = torch.rand((len(X), 1), device=X.device) < probability
        result[:, columns] = torch.where(selected, missing[None, :], result[:, columns])
    return result
