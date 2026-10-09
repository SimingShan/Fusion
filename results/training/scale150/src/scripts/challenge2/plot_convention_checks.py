"""Plot the controlled training-label magnetic convention diagnostic."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/challenge2"


def main():
    tags = ["d3d_physics_conventions_v1", "d3d_physics_conventions_rectangle_v2",
            "d3d_physics_conventions_oriented_v3"]
    data = [json.loads((BASE / tag / "summary.json").read_text()) for tag in tags]
    errors = [np.median([r["exterior_centered_RMSE_Wb_per_rad"] for r in d["records"]]) for d in data]
    old, new = data[1]["records"], data[2]["records"]
    assert [(r["training_shot_index"],r["frame_index"]) for r in old] == [
        (r["training_shot_index"],r["frame_index"]) for r in new]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), constrained_layout=True)
    colors = ["#94a3b8", "#7c8ea3", "#157a6e"]
    axes[0].bar(range(3), errors, color=colors, width=.65)
    axes[0].set_xticks(range(3), ["Center\nfilaments", "Finite\nsolenoid", "Finite solenoid\n+ orientation"])
    axes[0].set_ylabel("Median exterior RMSE [Wb/rad]")
    axes[0].set_title("Controlled convention changes")
    axes[0].set_ylim(0, .12)
    for i, value in enumerate(errors):
        axes[0].text(i, value+.004, f"{value:.5f}", ha="center", fontsize=10)
    current = np.array([r["Ip_measured_A"] for r in old])/1e6
    axes[1].scatter(current, [r["per_frame_PF_diagnostic_factor"] for r in old],
                    s=36, label="Raw current orientation", color="#b35d43")
    axes[1].scatter(current, [r["per_frame_PF_diagnostic_factor"] for r in new],
                    s=30, marker="x", label="Orientation normalized", color="#157a6e")
    axes[1].axhline(1, color=".5", linewidth=.8, linestyle="--")
    axes[1].axhline(-1, color=".5", linewidth=.8, linestyle=":")
    axes[1].set_xlabel("Measured plasma current [MA]")
    axes[1].set_ylabel("Target-fitted PF response factor")
    axes[1].set_title("Target convention differs from physical sign")
    axes[1].legend(fontsize=8, loc="lower right")
    offsets = np.array([r["required_frame_gauge_offset"] for r in new])
    axes[2].scatter(range(len(new)), offsets, color="#5965a8", s=32)
    axes[2].axhline(0, color=".5", linewidth=.8)
    axes[2].set_xlabel("Diagnostic training frame")
    axes[2].set_ylabel("Required constant offset [Wb/rad]")
    axes[2].set_title("Absolute flux offset remains unresolved")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle("DIII-D training-label convention checks — not prediction accuracy", fontsize=15)
    out = BASE / tags[-1]
    fig.savefig(out / "convention_checks.png", dpi=180)
    fig.savefig(out / "convention_checks.pdf")
    plt.close(fig)
    print(out / "convention_checks.png")


if __name__ == "__main__":
    main()
