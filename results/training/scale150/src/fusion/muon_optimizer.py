"""Fixed six-matrix upstream Muon plus unchanged native Torch AdamW.

The composite deliberately schedules the real AdamW optimizer, then copies its
relative learning-rate schedule to Muon. A shared absolute cosine floor would
break the calibrated ratio and matched multiplicative weight decay.
"""
from __future__ import annotations

import math
import torch
from .vendor.muon_upstream import SingleDeviceMuon

MUON_NAMES = tuple(f"blocks.{i}.layers.{j}.weight" for i in range(3) for j in (0, 3))


import re
import os
# experiment override: MUON_PATTERN env = regex of parameter names optimised by Muon (default = residual-block hidden matrices only)
MUON_PATTERN = re.compile(os.environ.get("MUON_PATTERN", r"blocks\.\d+\.layers\.\d+\.weight"))


def partition_parameters(model):
    """Generic: every 2-D hidden matrix of the residual blocks goes to Muon (any width/depth); the rest to AdamW."""
    named = dict(model.named_parameters())
    hidden = [p for n, p in named.items() if p.requires_grad and MUON_PATTERN.fullmatch(n) and p.ndim == 2]
    auxiliary = [p for n, p in named.items() if p.requires_grad and not (MUON_PATTERN.fullmatch(n) and p.ndim == 2)]
    if not hidden:
        raise ValueError("no hidden Muon matrices found")
    global MUON_NAMES
    MUON_NAMES = tuple(n for n, p in named.items() if p.requires_grad and MUON_PATTERN.fullmatch(n) and p.ndim == 2)
    return hidden, auxiliary


class MuonAdamW:
    """Disjoint real optimizers with one zero/step/state interface; no closures."""
    def __init__(self, model, lr, weight_decay, lr_ratio):
        if not math.isfinite(lr_ratio) or lr_ratio <= 0:
            raise ValueError("Muon learning-rate ratio must be positive and finite")
        self.lr_ratio = float(lr_ratio)
        self.hidden, auxiliary = partition_parameters(model)
        self.all_parameters = [p for p in model.parameters() if p.requires_grad]
        self.adam = torch.optim.AdamW(auxiliary, lr=lr, betas=(.9, .999), eps=1e-8,
                                      weight_decay=weight_decay)
        import os   # experiment overrides (default = shipped behaviour): MUON_MOMENTUM, MUON_WD_MULT (x the coupled wd/ratio)
        self.wd_mult = float(os.environ.get("MUON_WD_MULT", "1"))
        self.muon = SingleDeviceMuon(self.hidden, lr=lr * self.lr_ratio,
                                     weight_decay=weight_decay / self.lr_ratio * self.wd_mult,
                                     momentum=float(os.environ.get("MUON_MOMENTUM", ".95")))
        self.sync_learning_rates()

    @property
    def param_groups(self):
        return self.adam.param_groups + self.muon.param_groups

    def sync_learning_rates(self):
        if len(self.adam.param_groups) != 1 or len(self.muon.param_groups) != 1:
            raise ValueError("Unexpected optimizer group count")
        adam, muon = self.adam.param_groups[0], self.muon.param_groups[0]
        muon["lr"] = adam["lr"] * self.lr_ratio
        if not math.isclose(muon["weight_decay"] * self.lr_ratio, adam["weight_decay"] * self.wd_mult, rel_tol=1e-12):
            raise ValueError("Per-step multiplicative decay no longer matches AdamW")

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Composite Muon step requires precomputed gradients")
        if any(p.grad is None for p in self.all_parameters):
            raise ValueError("Missing intended gradient; upstream Muon would silently insert zeros")
        self.sync_learning_rates()
        self.adam.step()
        # Upstream mutates hidden .grad; clipping/logging must precede this call.
        self.muon.step()

    def state_dict(self):
        return {"version": 1, "lr_ratio": self.lr_ratio, "muon_names": MUON_NAMES,
                "adam": self.adam.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state):
        if state["version"] != 1 or state["lr_ratio"] != self.lr_ratio or tuple(state["muon_names"]) != MUON_NAMES:
            raise ValueError("Composite optimizer checkpoint contract changed")
        self.adam.load_state_dict(state["adam"])
        self.muon.load_state_dict(state["muon"])
        self.sync_learning_rates()


class MuonLRScheduler:
    """Delegate exact AdamW scheduler/state, retaining Muon's calibrated ratio."""
    def __init__(self, optimizer, adam_scheduler):
        if adam_scheduler.optimizer is not optimizer.adam:
            raise ValueError("Scheduler is not attached to the native auxiliary AdamW")
        self.optimizer = optimizer
        self.adam_scheduler = adam_scheduler
        self.optimizer.sync_learning_rates()

    def step(self, *args, **kwargs):
        self.adam_scheduler.step(*args, **kwargs)
        self.optimizer.sync_learning_rates()

    def get_last_lr(self):
        return self.adam_scheduler.get_last_lr()

    def state_dict(self):
        return {"version": 1, "lr_ratio": self.optimizer.lr_ratio,
                "adam_scheduler": self.adam_scheduler.state_dict()}

    def load_state_dict(self, state):
        if state["version"] != 1 or state["lr_ratio"] != self.optimizer.lr_ratio:
            raise ValueError("Composite scheduler checkpoint contract changed")
        self.adam_scheduler.load_state_dict(state["adam_scheduler"])
        self.optimizer.sync_learning_rates()


def make_optimizer_and_scheduler(model, args):
    """Production factory also exercised directly by paired optimizer smokes."""
    if args.muon_lr_ratio:
        optimizer = MuonAdamW(model, args.lr, args.weight_decay, args.muon_lr_ratio)
        scheduled = optimizer.adam
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduled = optimizer
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(scheduled, cosine_epochs, eta_min=args.lr * .03)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(scheduled, start_factor=args.warmup_start_factor,
                                                   total_iters=args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(scheduled, [warmup, cosine],
                                                          milestones=[args.warmup_epochs])
    else:
        scheduler = cosine
    return optimizer, MuonLRScheduler(optimizer, scheduler) if args.muon_lr_ratio else scheduler
