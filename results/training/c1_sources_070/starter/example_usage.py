#!/usr/bin/env python3
"""
Example: load the Fusion Equilibrium Challenge dataset from the Hugging Face Hub.

    pip install -r requirements.txt
    python example_usage.py

The primary target is the 2-D flux map (`efit_psirz`). Alongside it, `diii_d_train`
now ships scored EFIT scalar labels (`efit_beta_n`, `efit_li`, `efit_q95`,
`efit_r_axis`, `efit_z_axis`) plus the EFIT-derived x-point gap `magnetics_dsep` —
all withheld on the test splits.

Tip: if you have the dataset downloaded locally (same layout as the Hub repo,
`<root>/data/<config>/*.parquet`), you can read it without the Hub via pandas, e.g.
`pandas.read_parquet(".../data/diii_d_train/d3d_shot_203702.parquet").iloc[0]`.
See `experiments.py --source local` for a full offline pipeline.
"""

import numpy as np
from datasets import load_dataset

REPO_ID = "Sophelio/fusion-equilibrium-challenge"

# Scored EFIT scalar labels (present in train, withheld on test)
SCALAR_TARGETS = ["efit_beta_n", "efit_li", "efit_q95", "efit_r_axis", "efit_z_axis", "magnetics_dsep"]


def main() -> None:
    # --- DIII-D training data (includes the target flux maps + scalar labels) ---
    train = load_dataset(REPO_ID, "diii_d_train", split="train", streaming=True)
    shot = next(iter(train))

    times = np.asarray(shot["efit_times"])      # (T,) ms
    psirz = np.asarray(shot["efit_psirz"])      # (T, 65, 65) — the primary target
    ip = np.asarray(shot["magnetics_plasma_current"])

    print("DIII-D train example")
    print(f"  source              : {shot['source']}")
    print(f"  EFIT timesteps (T)  : {len(times)}")
    print(f"  flux map shape      : {psirz.shape}")
    print(f"  |Ip| peak (A)       : {np.nanmax(np.abs(ip)):.0f}")

    print("  EFIT scalar targets :")
    for name in SCALAR_TARGETS:
        if name in shot:
            arr = np.asarray(shot[name], dtype=float)
            print(f"    {name:<14} shape={arr.shape}  mean={np.nanmean(arr):.4g}")
    dsep = np.asarray(shot["magnetics_dsep"], dtype=float)  # EFIT-derived target
    print(f"    magnetics_dsep shape={dsep.shape}  "
          f"diverted frac={np.nanmean(dsep > 0):.2f} (x-point gap, a target)")

    # --- Public test (inputs only — all EFIT-derived targets are withheld) ---
    test = load_dataset(REPO_ID, "diii_d_public_test", split="public_test", streaming=True)
    t0 = next(iter(test))
    assert "efit_psirz" not in t0, "test split should not contain the target"
    withheld = ["efit_psirz", *SCALAR_TARGETS, "magnetics_dsep"]
    print("\nDIII-D public_test example")
    print(f"  has efit_psirz?     : {'efit_psirz' in t0}  (expected False)")
    print(f"  withheld targets    : {[k for k in withheld if k not in t0]}")
    print(f"  predict flux maps at: {len(t0['efit_times'])} timesteps")

    # --- MAST is zero-shot: test config only ---
    mast = load_dataset(REPO_ID, "mast_public_test", split="public_test", streaming=True)
    m0 = next(iter(mast))
    print("\nMAST public_test example")
    print(f"  source              : {m0['source']}")
    print(f"  grid R / Z present  : {'efit_grid_R' in m0} / {'efit_grid_Z' in m0}")


if __name__ == "__main__":
    main()
