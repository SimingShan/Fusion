"""Deterministic interactions of permitted, already-normalized input signals."""
from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn


def feature_schema(names, norm, constant_observations):
    selected = [names[i] for i in norm["feature_indices"]]
    columns = {name: i for i, name in enumerate(selected)}
    def observation(name):
        if name in columns:
            return columns[name]
        if constant_observations.get(name) not in (0., 1.):
            raise ValueError(f"Unrepresented observation mask is not a verified training constant: {name}")
        return -1 if constant_observations[name] == 1. else -2
    pressure = []
    for name in selected:
        if not re.fullmatch(r"thomson_(core|edge)_Te_log1p_channel_\d+", name):
            continue
        partners = [name, name.replace("_Te_", "_ne_"),
                    name.replace("_log1p_", "_observed_"),
                    name.replace("_Te_log1p_", "_ne_observed_")]
        if not all(key in columns for key in partners):
            continue
        indices = [columns[key] for key in partners]
        raw_indices = norm["feature_indices"][indices[:2]]
        if not np.array_equal(norm["raw_log_scale"][raw_indices], [1e3, 1e19]):
            raise ValueError("Physical products require the established Thomson unit scaling")
        pressure.append(indices)
    ip = columns["magnetics_plasma_current"]
    ipmask = observation("magnetics_plasma_current_observed")
    coils, history = [], []
    for name, index in columns.items():
        if re.fullmatch(r"magnetics_(ECOILA|bcoil|F[1-9][AB])", name):
            observed = name + "_observed"
            coils.append([index, observation(observed)])
        match = re.fullmatch(r"(magnetics_(?:ECOILA|bcoil|F[1-9][AB]|plasma_current))_value_offset_([+-]\d+)ms", name)
        if match:
            base, offset = match.groups()
            partners = [base, base + "_observed", base + "_observed_offset_" + offset + "ms"]
            if base in columns:
                history.append([index, columns[base], observation(partners[1]), observation(partners[2])])
    if not pressure or not coils or not history:
        raise ValueError("Required physical input groups are missing")
    return {"mean": norm["x_mean"].tolist(), "std": norm["x_std"].tolist(),
            "ip": ip, "ipmask": ipmask, "pressure": pressure, "coils": coils, "history": history,
            "constant_observations": constant_observations,
            "ip_floor_kA": 100., "pressure_units_Pa": 1.602176634e3}


class DerivedInputs(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.register_buffer("input_mean", torch.tensor(schema["mean"], dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor(schema["std"], dtype=torch.float32))
        for key in ("pressure", "coils", "history"):
            self.register_buffer(key + "_columns", torch.tensor(schema[key], dtype=torch.long))
        self.ip, self.ipmask = int(schema["ip"]), int(schema["ipmask"])
        self.ip_floor_kA = float(schema["ip_floor_kA"])
        self.width = 2*len(schema["pressure"]) + len(schema["coils"]) + len(schema["history"]) + 1
        self.register_buffer("mean", torch.zeros(self.width))
        self.register_buffer("std", torch.ones(self.width))

    def physical(self, x):
        raw = x*self.input_std + self.input_mean
        def observed(columns):
            columns = torch.as_tensor(columns, dtype=torch.long, device=raw.device).reshape(-1)
            # Negative indices encode verified constants excluded by the old
            # normalization; they never index another input signal.
            return torch.where(columns[None] >= 0, raw[:, columns.clamp_min(0)] > .5,
                               columns[None] == -1).to(raw)
        ip = raw[:, self.ip:self.ip+1].abs().clamp_min(self.ip_floor_kA)
        ip_valid = observed([self.ipmask])*(raw[:, self.ip:self.ip+1].abs() > 0).to(raw)
        p = self.pressure_columns
        te = torch.expm1(raw[:, p[:, 0]].clamp(0., 20.))
        ne = torch.expm1(raw[:, p[:, 1]].clamp(0., 20.))
        pressure_observed = ((raw[:, p[:, 2]] > .5) & (raw[:, p[:, 3]] > .5)).to(raw)
        electron = te*ne*pressure_observed
        pressure = torch.log1p(electron)
        normalized_pressure = torch.log1p(electron/(ip/1000.).square())*ip_valid
        c = self.coils_columns
        coil = torch.asinh(raw[:, c[:, 0]]/ip)*observed(c[:, 1])*ip_valid
        h = self.history_columns
        history = torch.asinh((raw[:, h[:, 0]]-raw[:, h[:, 1]])/ip)
        history = history*observed(h[:, 2])*observed(h[:, 3])*ip_valid
        return torch.cat((pressure, normalized_pressure, coil, history, torch.log1p(ip/1000.)*ip_valid), dim=1)

    def forward(self, x):
        return ((self.physical(x)-self.mean)/self.std).clamp(-12., 12.)
