"""Drop-in fixes for the two known data errata in dataset release v1.1.0.

Both issues were participant-reported (thank you!) and affect *input* columns
only -- the flux-map targets, ``efit_times``, and all scoring are unaffected,
and every participant has the same data. A corrected v1.1.1 dataset release
will match these fixes exactly, so anything you build on top of them carries
over unchanged.

1.  **DIII-D ``magnetics_plasma_current_times`` is wrong on ~69% of shots**
    (`issue #6 <https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/6>`_).
    Every DIII-D file shipped with the *same* Ip time axis. That axis is
    genuinely correct for ~31% of shots, but on the rest the Ip trace actually
    starts at ``magnetics_time[0]`` -- about three seconds earlier -- so
    interpolating Ip onto ``efit_times`` with the shipped axis reads the
    pre-plasma noise floor (~2 kA) instead of the real current (~1 MA).
    The Ip *values* are correct on every shot; only the timestamps are wrong.
    Use :func:`fix_d3d_ip_times`. MAST is unaffected (its Ip sits on
    ``magnetics_time``).

2.  **MAST ``thomson_core_R`` has 130 entries but Te/ne have 131 channels**
    (`issue #5 <https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/5>`_).
    Channel 0 of the raw Thomson system has no radius calibration and is NaN
    in every shot; its coordinate was dropped at conversion time but its
    (empty) data column was kept. The alignment is therefore
    ``R[j] <-> Te[t][j + 1]``. Use :func:`align_mast_thomson_core`.

Run ``python data_fixes.py`` to see both fixes applied to the demo shots in
``parquet_data/`` (requires ``git lfs pull``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["fix_d3d_ip_times", "align_mast_thomson_core"]

# On DIII-D, t = 0 is plasma breakdown by convention, so the correct Ip axis
# is the one that puts the first significant current near t = 0. Across all
# 9,113 DIII-D source shots the winning axis puts it within [-118, +348] ms
# while the losing axis lands seconds away, so this window is unambiguous.
_BREAKDOWN_WINDOW_MS = (-500.0, 1000.0)


def fix_d3d_ip_times(row) -> np.ndarray:
    """Return the corrected time axis for ``magnetics_plasma_current`` (DIII-D).

    Accepts a Hugging Face dataset row (dict) or a pandas row from a demo
    parquet. It needs only input columns, so it works identically on the
    train and the inputs-only test splits.

    Per shot, this keeps the shipped ``magnetics_plasma_current_times`` if it
    already places plasma breakdown (first |Ip| crossing of 10% of the shot's
    peak) near t = 0, and otherwise re-origins the axis to start at
    ``magnetics_time[0]``, which is where the affected shots' Ip acquisition
    actually began. Validated on all 9,113 DIII-D source shots: the decision
    is unambiguous on every one, correcting 6,322 shots (4,930 of 7,041 in
    ``diii_d_train``, 586 of 874 in ``diii_d_public_test``) and leaving the
    already-correct 2,791 untouched.

    Do NOT blanket-shift every shot instead -- ~31% of shots ship with the
    correct axis, and shifting those would corrupt them.
    """
    ipt = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    ip = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64))
    threshold = 0.10 * np.nanpercentile(ip, 99.5)
    breakdown = float(ipt[int(np.argmax(ip > threshold))])
    lo, hi = _BREAKDOWN_WINDOW_MS
    if lo <= breakdown <= hi:
        return ipt
    return float(row["magnetics_time"][0]) + (ipt - ipt[0])


def align_mast_thomson_core(row) -> tuple:
    """Return ``(R, Te, ne)`` for a MAST row with the ghost channel dropped.

    MAST rows ship 130 coordinates in ``thomson_core_R`` but 131 channels in
    ``thomson_core_Te`` / ``thomson_core_ne``; the extra data channel is
    channel 0, which is NaN in every shot. This drops it, so column ``j`` of
    the returned ``Te`` / ``ne`` (shape ``(T, 130)``) sits at radius ``R[j]``.
    Rows whose counts already match (all DIII-D shots) pass through unchanged.
    """
    R = np.asarray(row["thomson_core_R"], dtype=np.float64)
    Te = np.stack([np.asarray(p) for p in row["thomson_core_Te"]])
    ne = np.stack([np.asarray(p) for p in row["thomson_core_ne"]])
    if Te.shape[1] == R.shape[0]:
        return R, Te, ne
    if Te.shape[1] != R.shape[0] + 1:
        raise ValueError(
            f"unexpected Thomson core layout: {R.shape[0]} coordinates vs "
            f"{Te.shape[1]} data channels"
        )
    return R, Te[:, 1:], ne[:, 1:]


def _demo() -> None:
    from pathlib import Path

    import pandas as pd

    data_dir = Path(__file__).parent / "parquet_data"
    d3d = sorted(data_dir.glob("d3d_shot_*.parquet"))
    mast = sorted(data_dir.glob("mast_shot_*.parquet"))
    if not d3d or d3d[0].stat().st_size < 10_000:
        raise SystemExit(
            "Demo shots missing or still LFS pointers -- run: git lfs install && git lfs pull"
        )

    print("DIII-D Ip time-axis fix (issue #6):")
    for path in d3d:
        row = pd.read_parquet(
            path,
            columns=[
                "magnetics_plasma_current_times",
                "magnetics_plasma_current",
                "magnetics_time",
                "efit_times",
            ],
        ).iloc[0]
        shipped = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
        fixed = fix_d3d_ip_times(row)
        ip = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)
        et = np.asarray(row["efit_times"], dtype=np.float64)
        before = np.median(np.abs(np.interp(et, shipped, ip)))
        after = np.median(np.abs(np.interp(et, fixed, ip)))
        changed = "corrected" if fixed[0] != shipped[0] else "kept as shipped"
        print(
            f"  {path.name}: {changed}; median |Ip| at efit_times "
            f"{before:,.0f} -> {after:,.0f}"
        )

    print("MAST Thomson core alignment fix (issue #5):")
    for path in mast:
        row = pd.read_parquet(
            path, columns=["thomson_core_R", "thomson_core_Te", "thomson_core_ne"]
        ).iloc[0]
        n_before = len(row["thomson_core_Te"][0])
        R, Te, ne = align_mast_thomson_core(row)
        print(
            f"  {path.name}: Te {n_before} channels -> {Te.shape[1]}, "
            f"aligned to {R.shape[0]} coordinates"
        )


if __name__ == "__main__":
    _demo()
