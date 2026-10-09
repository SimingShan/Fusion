"""PyTorch baseline models — imported only when torch is installed.

These baselines predict the 2-D flux map (`efit_psirz`). The scored EFIT scalar
labels (`efit_beta_n`, `efit_li`, `efit_q95`, `efit_r_axis`, `efit_z_axis`,
`magnetics_dsep`) are predicted with a Ridge baseline in ``experiments.py`` and run
regardless of the PyTorch install — extending these networks with a scalar head is a
natural next step (add an extra output branch trained against ``data["S_train"]``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import (
    EFIT_GRID_SIZE, compute_metrics, plot_flux_comparison, plot_residual_heatmap,
    plot_sample_grid, plot_training_curves, typical_sample_idx,
)

if TYPE_CHECKING:
    from experiments import ExperimentConfig


class SimpleMLP(nn.Module):
    """Fully connected network: features -> 65x65 flux map."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, EFIT_GRID_SIZE * EFIT_GRID_SIZE),
        )

    def forward(self, x):
        return self.net(x).view(-1, EFIT_GRID_SIZE, EFIT_GRID_SIZE)


class ConvDecoder(nn.Module):
    """FC encoder to latent, then ConvTranspose2d decoder to 65x65."""

    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 4 * 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )
        self.to_target = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        z = self.encoder(x).view(-1, 32, 4, 4)
        out = self.decoder(z)
        out = self.to_target(out)
        return out.squeeze(1)


class UNetLite(nn.Module):
    """Lightweight decoder with skip-connection-like structure for 65x65 output."""

    def __init__(self, n_features: int):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 64 * 8 * 8)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=2, stride=1, padding=0),
        )
        self.skip_fc = nn.Linear(n_features, 128 * 16 * 16)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h)).view(-1, 64, 8, 8)
        h = self.up1(h)
        skip = torch.relu(self.skip_fc(x)).view(-1, 128, 16, 16)
        h = h + skip
        h = self.up2(h)
        h = self.up3(h)
        return self.final(h).squeeze(1)


def resolve_device(config: ExperimentConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    config: ExperimentConfig,
) -> dict:
    device = resolve_device(config)
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()),
        batch_size=config.batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
                n_val += 1
        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {epoch + 1:3d}/{config.epochs}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.1e}"
            )

    return {"model": model, "train_losses": train_losses, "val_losses": val_losses, "device": device}


def predict_pytorch(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_tensor), 512):
            preds.append(model(X_tensor[i : i + 512]).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_pytorch_experiments(config: ExperimentConfig, data: dict[str, Any]) -> dict[str, dict]:
    print("\n=== Training PyTorch Models ===")
    device = resolve_device(config)
    print(f"  Device: {device}")
    n_features = data["n_features"]

    pytorch_models = [
        ("Simple MLP (PyTorch)", SimpleMLP(n_features)),
        ("Conv Decoder", ConvDecoder(n_features)),
        ("UNet Lite", UNetLite(n_features)),
    ]

    # Shared model-agnostic "typical" test frame (matches the sklearn path; avoids frame 0).
    rep_idx = typical_sample_idx(data["Y_test"])

    results: dict[str, dict] = {}
    for name, model in pytorch_models:
        print(f"\n  [{name}]")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

        t0 = time.time()
        result = train_pytorch_model(
            model, data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], config,
        )
        fit_time = time.time() - t0

        Y_pred = predict_pytorch(result["model"], data["X_test"], result["device"])
        metrics = compute_metrics(data["Y_test"], Y_pred)
        metrics["fit_time"] = fit_time
        results[name] = metrics

        print(
            f"    MSE={metrics['MSE']:.6f}, R2={metrics['R2']:.4f}, "
            f"SSIM={metrics.get('SSIM', 'N/A')}, time={fit_time:.1f}s"
        )

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_flux_comparison(
            data["Y_test"], Y_pred, sample_idx=rep_idx,
            title=f"{name} - typical test frame (idx {rep_idx})",
            output_path=config.output_dir / f"flux_{safe_name}.png",
        )
        plot_sample_grid(
            data["Y_test"], Y_pred, name,
            output_path=config.output_dir / f"grid_{safe_name}.png",
        )
        plot_residual_heatmap(
            metrics["residual_map"], name,
            output_path=config.output_dir / f"residual_{safe_name}.png",
        )
        plot_training_curves(
            result["train_losses"], result["val_losses"], name,
            output_path=config.output_dir / f"training_{safe_name}.png",
        )

    return results
