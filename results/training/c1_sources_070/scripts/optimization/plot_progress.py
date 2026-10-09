#!/usr/bin/env python3
"""Refresh local experiment reports and scientific figures from saved JSON files.

Run from any directory:
    .venv/bin/python scripts/optimization/plot_progress.py

Only persisted val/legacy_val/audit scores, selected blends, and the original
baseline scores are compared. Different selections/scorers get separate panels.
Oracle reconstructions and perfect self-checks never enter learned-model tables.
This script reads no prediction arrays, checkpoints, training data, or test labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = {
    "S": 0.995440,
    "participant": "kimsunjoong",
    "observed_utc": "2026-09-13T01:59:00+00:00",
    "observed_local": "2026-09-12 21:59 EDT",
    "url": "https://www.codabench.org/api/leaderboards/19740/",
    "qualification": "Historical public-test score; local validation does not establish leaderboard rank.",
}
COLORS = ["#287D8E", "#E79938", "#6F60A7", "#CF6272"]
COMPONENTS = ["Flux", "Submitted scalars", "Boundary", "Derived consistency"]


def read_json(path, warnings, optional=False):
    if optional and not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        warnings.append(f"Skipped {path}: {type(exc).__name__}: {exc}")
        return None


def relative(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def diagnostic(path, score):
    evaluation = score.get("evaluation", {})
    text = " ".join([str(path).lower(), str(evaluation.get("type", "")).lower()])
    markers = ("oracle", "perfect", "self_check", "ground_truth", "compression_diagnostic")
    return evaluation.get("model_prediction") is False or any(word in text for word in markers)


@lru_cache(maxsize=16)
def manifest_identities(directory):
    """Resolve human-readable cache aliases through pinned raw-shot provenance."""
    try:
        manifest = json.loads((Path(directory)/"manifest.json").read_text())
    except (OSError,ValueError,TypeError):
        return {}, None
    splits = manifest.get("splits",manifest)
    entries = {}
    for split in ("train","val","legacy_val","audit"):
        for entry in splits.get(split,[]):
            if entry.get("shot_index") is not None and entry.get("shard"):
                entries[entry["shot_index"]] = f"{entry['shard']}#row={entry.get('row',0)}"
    dataset = manifest.get("dataset",{})
    return entries,[dataset.get("repo"),dataset.get("revision")]


def selection_identity(score, path):
    """Never merge folds based only on an ambiguous filename or shot count."""
    evaluation = score.get("evaluation", {})
    shots = evaluation.get("shots", [])
    provenance,dataset = manifest_identities(evaluation.get("data_directory", ""))
    selected = []
    for shot in shots:
        frames = shot.get("evaluated_original_frame_indices")
        identity = provenance.get(shot.get("shot_index"),shot.get("shot_id",shot.get("shot_index")))
        if identity is None or frames is None:
            return "unverified_" + hashlib.sha256(str(path).encode()).hexdigest()[:12], False, []
        selected.append([shot.get("shot_index"), str(identity), sorted(int(i) for i in frames)])
    if not selected:
        return "unverified_" + hashlib.sha256(str(path).encode()).hexdigest()[:12], False, []
    selected.sort(key=lambda shot: str(shot[:2]))
    contract = [dataset,evaluation.get("machine"), evaluation.get("scorer_version"),
                evaluation.get("scorer_sha256"), selected]
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    return digest, True, [shot[:2] for shot in selected]


def display_name(directory, root):
    names = {"deeponet": "DeepONet (initial)", "fourier_deeponet": "Fourier-DeepONet (initial)"}
    if directory.parent == root:
        return names.get(directory.name, directory.name)
    return directory.name.replace("_", " ")


def plot_name(row):
    names={"deeponet":"Initial DeepONet","fourier_deeponet":"Initial Fourier",
           "pod_legacy":"POD + Ridge","pod_fourier_legacy":"POD + Fourier",
           "pod_legacy_fluxloss":"POD, flux loss","pod_legacy_noridge":"POD, no Ridge",
           "pod_legacy_flux_noridge":"POD, flux/no Ridge","pod_full_v1":"Full-data POD",
           "pod_fourier_full_v1":"Full-data Fourier","pod_currents64_cpu":"Currents-only POD",
           "pod_history64_cpu":"History POD","pod_li_refined_cpu":"POD + li refinement",
           "pod_scaled_v2":"Scaled-feature POD"}
    if row.get("selection_provenance"):return "Mean/shape blend*"
    if row.get("quality_provenance"):
        if row["quality_provenance"].get("secondary_expert")=="robust-history-POD-DeepONet":
            return "Scaled/robust gate" if row["quality_provenance"]["method"]=="gate" else "Scaled/robust 75/25"
        p=row["quality_provenance"];prefix={"Fourier":"Fourier","li-refined_POD_DeepONet":"li POD"}.get(p.get("expert"),"POD")
        suffix={"gate":"quality gate","blend75":"75/25 blend","gated_blend75":"gated 75/25"}[p["method"]]
        return f"{prefix} {suffix}"
    return names.get(Path(row["run_directory"]).name,row["name"])


def official_submissions(results,warnings):
    public=results/"optimization/leaderboard_review"
    observations=[]
    for path in public.glob("public_snapshot_*.json"):
        item=read_json(path,warnings,True) or {}
        for row in item.get("d3d_S_user_ranks",[]):
            observations.append({**row,"observed_utc":item.get("observed_utc")})
    rows=[]
    for path in results.glob("optimization/*/codabench_submission.json"):
        saved=read_json(path,warnings,True) or {};official=saved.get("official",{})
        if not saved.get("submission_id"):continue
        matches=sorted((row for row in observations if row["submission_id"]==saved["submission_id"]),key=lambda row:row["observed_utc"])
        rows.append({"submission_id":saved["submission_id"],"status":official.get("status"),
            "scores":official.get("scores",[]),"created_utc":official.get("created_when"),
            "archive_sha256":saved.get("sha256"),"method":saved.get("method_name",path.parent.name),
            "status_source":str(path.resolve()),"first_rank_observation":matches[0] if matches else None,
            "latest_rank_observation":matches[-1] if matches else None})
    return sorted(rows,key=lambda row:row.get("created_utc") or "")


def discover(results, output, warnings):
    optimization = results / "optimization"
    # Candidate snapshots are deliberately excluded: each run's retained score is
    # val_score.json, while oracle reports are collected in their own table.
    paths = list(results.glob("*/score.json"))
    paths.extend(optimization.glob("native*/*/val_score.json"))
    for name in ("val_score.json", "legacy_val_score.json", "audit_score.json",
                 "oracle_compression_score.json"):
        paths.extend(optimization.glob(f"*/{name}"))
    paths.extend(path for path in optimization.glob("*/*score.json")
                 if any(marker in path.name.lower() for marker in ("oracle", "perfect", "self_check")))
    blends, quality = {}, {}
    summaries=list(optimization.glob("*/summary.json"))+list(optimization.glob("native*/*/summary.json"))
    for summary_path in summaries:
        summary = read_json(summary_path, warnings)
        if not isinstance(summary,dict):
            continue
        policy=read_json(summary_path.parent/"policy.json",warnings,True) or {}
        if (policy.get("fourier_gate") or policy.get("primary_gate")) and policy.get("gated_blend75") and policy.get("threshold_search") is False:
            for split,result in summary.get("splits",{}).items():
                for method in ("gate","gated_blend75"):
                    selected_path=summary_path.parent/method/f"{split}_score.json"
                    if not selected_path.exists():continue
                    paths.append(selected_path)
                    quality[str(selected_path.resolve())]={
                        "summary_path":str(summary_path.resolve()),"policy_sha256":summary.get("policy_sha256"),
                        "method":method,"expert":policy.get("primary_expert","Fourier"),"split":split,"threshold_searched":False,
                        "secondary_expert":policy.get("secondary_expert","currents-only POD"),
                        "threshold":policy.get("threshold"),"fixed_blend_full_weight":policy.get("fourier_weight"),
                        "gated_frames":result.get("coverage",{}).get("gated_frames"),
                        "frames":result.get("coverage",{}).get("frames"),
                        "audit_labels_read":summary.get("audit_labels_read"),
                        "public_test_input_only_coverage":summary.get("public_test_input_only_coverage")}
            continue
        if summary.get("evaluation_type") == "predeclared_input_quality_gate_and_fixed_blend":
            for split,result in summary.get("splits",{}).items():
                for method in ("gate","blend75"):
                    raw_path = result.get("score_paths",{}).get(method)
                    if not raw_path:
                        continue
                    selected_path = Path(raw_path)
                    if not selected_path.is_absolute():
                        selected_path = summary_path.parent/selected_path
                    paths.append(selected_path)
                    quality[str(selected_path.resolve())] = {
                        "summary_path":str(summary_path.resolve()),"policy_sha256":summary.get("policy_sha256"),
                        "method":method,"expert":"Full POD","split":split,"threshold_searched":False,
                        "threshold":.1,"fixed_blend_full_weight":.75,
                        "gated_frames":result.get("coverage",{}).get("gated_frames"),
                        "frames":result.get("coverage",{}).get("frames"),
                        "audit_labels_read":summary.get("audit_labels_read"),
                    }
            continue
        if summary.get("evaluation_type") != "validation_tuned_prediction_blend":
            continue
        best = summary.get("best", {})
        if not best.get("score_path"):
            warnings.append(f"Blend summary has no selected score path: {summary_path}")
            continue
        selected_path = Path(best["score_path"])
        if not selected_path.is_absolute():
            selected_path = summary_path.parent / selected_path
        paths.append(selected_path)
        blends[str(selected_path.resolve())] = {
            "summary_path": str(summary_path.resolve()), "blend_name": summary_path.parent.name,
            "evaluation_type": summary["evaluation_type"],
            "independent_holdout": summary.get("independent_holdout", False),
            "source_a": summary.get("source_a"), "source_b": summary.get("source_b"),
            "formula": summary.get("formula"), "label_use": summary.get("label_use"),
            "candidate_count": len(summary.get("candidates", [])),
            "selected_weights": {key: best.get(key) for key in ("mean_weight_a", "shape_weight_a", "scalar_weight_a")},
        }
    rows, oracle = [], []
    runs = {}
    for path in sorted(set(paths)):
        if output.resolve() in path.resolve().parents:
            continue
        score = read_json(path, warnings)
        if not isinstance(score, dict) or not number(score.get("S")):
            if score is not None:
                warnings.append(f"Skipped {path}: no finite composite S")
            continue
        if any(not number(score.get(key)) for key in ("r2_psi", "r2_qb", "dlcfs", "consistency")):
            warnings.append(f"Skipped {path}: incomplete metric component values")
            continue
        run = runs.setdefault(path.parent, read_json(path.parent / "run.json", warnings, True) or {})
        if run and path.name=="val_score.json":
            history=read_json(path.parent/"history.json",warnings,True)
            if isinstance(history,list):
                scored=[record for record in history if isinstance(record,dict) and number(record.get("S"))]
                if scored and not any(record.get("epoch",0)>0 for record in scored):
                    # The run remains visible in histories, but its initialization
                    # control is not a trained-model validation result.
                    continue
        evaluation = score.get("evaluation", {})
        fold, verified, shot_ids = selection_identity(score, path)
        created = evaluation.get("created_utc") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        row = {
            "name": display_name(path.parent, results), "run_directory": str(path.parent.resolve()),
            "source": str(path.resolve()), "created_utc": created, "fold_id": fold,
            "selection_verified": verified, "shot_ids": shot_ids,
            "split": evaluation.get("split", "val"),
            "n_shots": evaluation.get("evaluated_shots", score.get("n_shots")),
            "n_frames": evaluation.get("evaluated_frames"),
            "scorer_version": evaluation.get("scorer_version", "unknown"),
            "metrics": {key: value for key, value in score.items() if key != "evaluation"},
            "train_shots": run.get("train_shots", run.get("basis", {}).get("train_shots")),
            "train_frames": run.get("train_frames"),
            "model": run.get("model", path.parent.name), "status": run.get("status", "score saved"),
            "diagnostic_type": evaluation.get("type", "unspecified"),
        }
        if number(score.get("compression_diagnostic", {}).get("rank")):
            row["compression_rank"] = score["compression_diagnostic"]["rank"]
            row["name"] += f" · rank {score['compression_diagnostic']['rank']}"
        if str(path.resolve()) in blends:
            row["selection_provenance"] = blends[str(path.resolve())]
            row["name"] = blends[str(path.resolve())]["blend_name"].replace("_", " ") + " (validation-tuned ensemble)"
            row["model"] = "validation_tuned_prediction_blend"
        if str(path.resolve()) in quality:
            row["quality_provenance"]=quality[str(path.resolve())]
            method=quality[str(path.resolve())]["method"]
            row["name"]="Fixed input-quality gate (10%)" if method=="gate" else "Fixed 75% full / 25% currents blend"
            if quality[str(path.resolve())].get("expert")=="Fourier":
                row["name"]="Fixed Fourier quality gate (10%)" if method=="gate" else "Fixed gated 75% Fourier / 25% currents blend"
            elif quality[str(path.resolve())].get("expert")=="li-refined_POD_DeepONet":
                row["name"]="Fixed li-refined POD quality gate (10%)" if method=="gate" else "Fixed gated 75% li-refined POD / 25% currents blend"
            else:
                p=quality[str(path.resolve())]
                if p.get("expert") not in ("Full POD",None):
                    row["name"]=f"Fixed {p['expert']} / {p.get('secondary_expert','currents-only POD')} " + ("quality gate" if method=="gate" else "gated 75/25 blend")
            row["model"]="fixed_input_quality_gate" if method=="gate" else "fixed_prediction_blend"
        row["deficits"] = [.55 * (1 - max(0, score["r2_psi"])),
                            .15 * (1 - max(0, score["r2_qb"])),
                            .10 * min(1, score["dlcfs"]), .20 * (1 - score["consistency"])]
        if abs(sum(row["deficits"]) - (1 - score["S"])) > 3e-6:
            warnings.append(f"Metric weights do not reproduce S within rounding tolerance: {path}")
        (oracle if diagnostic(path, score) else rows).append(row)
    rows.sort(key=lambda row: (row["created_utc"], row["source"]))
    oracle.sort(key=lambda row: (row["fold_id"], row["run_directory"], row.get("compression_rank", 0), row["name"]))

    original = next((row for row in rows if Path(row["source"]).parent == results / "deeponet"), None)
    legacy_ids = original["shot_ids"] if original else None
    groups = {}
    for row in rows + oracle:
        if not row["selection_verified"]:
            base = "Unverified selection — kept separate"
        elif legacy_ids is not None and row["shot_ids"] == legacy_ids:
            base = "Legacy 32-shot validation"
        elif row["split"] == "audit":
            base = f"Independent audit — {row['n_shots']} shots"
        elif row["n_shots"] == 128:
            base = "Fresh 128-shot validation — " + ("4,068 sampled native frames" if row["n_frames"]==4068 else f"all {row['n_frames']:,} native frames")
        else:
            base = f"Other validation — {row['n_shots']} shots"
        group = groups.setdefault(row["fold_id"], {
            "fold_id": row["fold_id"], "label": base,
            "n_shots": row["n_shots"], "n_frames": row["n_frames"],
            "scorer_version": row["scorer_version"], "selection_verified": row["selection_verified"],
        })
        row["fold_label"] = group["label"]
    # Distinguish equal shot sets evaluated on different native frames or scorers.
    labels = defaultdict(list)
    for group in groups.values():
        labels[group["label"]].append(group)
    for duplicates in labels.values():
        if len(duplicates) > 1:
            for group in duplicates:
                group["label"] += f" [{group['fold_id'][:6]}]"
    for row in rows + oracle:
        row["fold_label"] = groups[row["fold_id"]]["label"]
    return rows, oracle, groups, runs


def ordered_groups(groups, rows):
    used = {row["fold_id"] for row in rows}
    def key(group):
        label = group["label"]
        priority = 0 if label.startswith("Legacy") else 1 if label.startswith("Fresh") else 2
        return priority, label, group["fold_id"]
    return sorted((group for fold, group in groups.items() if fold in used), key=key)


def save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_panels(groups, rows):
    panels = ordered_groups(groups, rows)
    if not any(group["label"].startswith("Fresh") for group in panels):
        panels.append({"fold_id": "awaiting_fresh", "label": "Fresh 128-shot validation", "n_frames": None})
    if not panels:
        panels = [{"fold_id": "awaiting", "label": "Awaiting learned-model scores", "n_frames": None}]
    return panels


def style_axis(ax):
    ax.grid(axis="y", color="#DDE3E8", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def score_progression(rows, groups, output):
    panels = plot_panels(groups, rows)
    largest=max((sum(row["fold_id"]==group["fold_id"] for row in rows) for group in panels),default=1)
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(11,.8*largest), 4.7 * len(panels)), squeeze=False, layout="constrained")
    for ax, group in zip(axes[:, 0], panels):
        selected = [row for row in rows if row["fold_id"] == group["fold_id"]]
        ax.set_title(group["label"] + (f" · {group['n_frames']:,} frames" if number(group.get("n_frames")) else ""), loc="left")
        ax.axhline(REFERENCE["S"], color="#A94442", linestyle=":", lw=1.5,
                   label="Public-test reference 0.995440 (Sep 12/13); different data")
        if selected:
            x = np.arange(len(selected))
            y = np.asarray([row["metrics"]["S"] for row in selected])
            ax.plot(x, y, "o", color=COLORS[0], ms=7, label="Saved local model score")
            ax.step(x, np.maximum.accumulate(y), where="post", color=COLORS[0], alpha=.4, lw=1.5,
                    label="Best among displayed saved evaluations")
            for index, value in zip(x, y):
                ax.annotate(f"{value:.6f}", (index, value), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8)
            ticks = np.unique(np.linspace(0, len(selected) - 1, min(len(selected), 18), dtype=int))
            ax.set_xticks(ticks, [textwrap.fill(plot_name(selected[i]), 19) for i in ticks], rotation=35, ha="right")
            ax.set_ylim(max(0, float(y.min()) - .04), 1.02)
            ax.set_xlim(-.5, max(.5, len(selected) - .5))
        else:
            ax.text(.5, .45, "No learned-model result saved for this fold yet", transform=ax.transAxes,
                    ha="center", color="#657582")
            ax.set_xticks([])
            ax.set_ylim(0, 1.04)
        ax.set_ylabel("Local composite S")
        ax.set_xlabel("Saved run evaluations in UTC order; separate folds are not pooled")
        ax.legend(loc="lower right", fontsize=8)
        style_axis(ax)
    fig.suptitle("Optimization progress on held-out DIII-D shots\nLocal scores do not establish leaderboard rank", fontsize=14, fontweight="bold")
    save_figure(fig, output, "score_progression")


def deficit_plot(rows, groups, output, max_models):
    panels = plot_panels(groups, rows)
    heights = []
    for group in panels:
        count = sum(row["fold_id"] == group["fold_id"] for row in rows)
        heights.append(max(2.6, .43 * min(count, max_models) + 1.1))
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, sum(heights) + 1), squeeze=False,
                             gridspec_kw={"height_ratios": heights}, layout="constrained")
    for ax, group in zip(axes[:, 0], panels):
        selected = sorted((row for row in rows if row["fold_id"] == group["fold_id"]),
                          key=lambda row: row["metrics"]["S"], reverse=True)[:max_models]
        ax.set_title(group["label"], loc="left")
        if selected:
            values = np.asarray([row["deficits"] for row in selected])
            left = np.zeros(len(selected))
            for column, (label, color) in enumerate(zip(COMPONENTS, COLORS)):
                ax.barh(np.arange(len(selected)), values[:, column], left=left, color=color, label=label)
                left += values[:, column]
            ax.set_yticks(np.arange(len(selected)), [textwrap.fill(row["name"], 31) for row in selected])
            for i, row in enumerate(selected):
                ax.text(left[i] + max(float(left.max()) * .015, .0001), i,
                        f"S={row['metrics']['S']:.6f}", va="center", fontsize=8)
            ax.set_xlim(0, max(float(left.max()) * 1.35, .01))
            ax.invert_yaxis()
        else:
            ax.text(.5, .5, "Awaiting learned-model scores", ha="center", transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xlim(0, .15)
        ax.set_xlabel("Weighted deficit from perfect score (1 − S); smaller is better")
        ax.grid(axis="x", color="#DDE3E8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    fig.legend(handles, COMPONENTS, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Which errors still limit local scores?\nUp to {max_models} strongest saved models per exact validation fold", fontsize=14, fontweight="bold")
    save_figure(fig, output, "score_deficits")


def training_history(results, rows, output, warnings, max_history):
    paths = list(results.glob("*/history.json")) + list((results / "optimization").glob("*/history.json"))
    available = []
    for path in sorted(set(paths), key=lambda p: p.stat().st_mtime):
        history = read_json(path, warnings)
        if isinstance(history, dict):
            history = history.get("history", history.get("epochs", []))
        if not isinstance(history, list) or not history:
            continue
        records = [item for item in history if isinstance(item, dict) and number(item.get("epoch"))]
        if not records or diagnostic(path, {}):
            continue
        candidates = [row for row in rows if row["run_directory"] == str(path.parent.resolve()) and row["split"] == "val"]
        label = candidates[0]["fold_label"] if candidates else "Fold not yet verified by a saved evaluation"
        available.append((path, records, label))
    selected = available[-max_history:]
    n = max(1, len(selected))
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(12, 3.4 * math.ceil(n / 2)), squeeze=False, layout="constrained")
    for ax, item in zip(axes.ravel(), selected):
        path, records, fold = item
        for key, label, linestyle in (("train_loss", "Training", "--"), ("val_loss", "Validation", "-")):
            points = [(row["epoch"], row[key]) for row in records if number(row.get(key)) and row[key] > 0
                      and not (key == "train_loss" and row["epoch"] == 0)]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, linestyle=linestyle, color=COLORS[0] if key == "val_loss" else "#91A8B0", label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recorded objective (log)")
        ax.set_title(textwrap.fill(display_name(path.parent, results), 45) + "\n" + fold, fontsize=10, loc="left")
        score_points = [(row["epoch"], row["S"]) for row in records if number(row.get("S"))]
        if score_points:
            second = ax.twinx()
            x, y = zip(*score_points)
            second.plot(x, y, "o-", color=COLORS[1], ms=3, lw=1, label="Local S")
            second.set_ylabel("Local S", color=COLORS[1])
            second.tick_params(axis="y", colors=COLORS[1])
            second.spines["top"].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")
        style_axis(ax)
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    if not selected:
        axes[0, 0].axis("on")
        axes[0, 0].text(.5, .5, "No readable training histories yet", ha="center", transform=axes[0, 0].transAxes)
    fig.suptitle("Training histories\nObjectives and scales may differ between runs; compare scores only within the same fold", fontsize=13, fontweight="bold")
    save_figure(fig, output, "training_histories")
    return {"available": len(available), "plotted": [relative(path, results) for path, _, _ in selected],
            "scope": f"Most recently updated {max_history} histories; each run has its own loss axis"}


def md_number(value, digits=6):
    return f"{value:.{digits}f}" if number(value) else "—"


def report(rows, oracle, groups, results, output, warnings, history_scope):
    stamp = datetime.now(timezone.utc).isoformat()
    submissions=official_submissions(results,warnings)
    submission=submissions[-1] if submissions else {"submission_id":None}
    lines = ["# Fusion optimization progress", "", f"Generated: {stamp}", "",
             "These are local held-out DIII-D scores. Separate shot/frame selections and scorer versions are reported separately; oracle reconstructions are excluded from all learned-model comparisons.", "",
             f"Historical public-test reference: **S = {REFERENCE['S']:.6f}** ({REFERENCE['participant']}, observed {REFERENCE['observed_local']} / September 13 UTC). This is different data and does not establish a local model's leaderboard rank. [Public leaderboard API]({REFERENCE['url']}).", "",
             "Each learned-model row is a persisted score file, normally the run's retained best checkpoint. Plots show saved run evaluations ordered by their evaluation timestamp; they are not a complete archive of every training attempt. Active runs may improve on the next refresh.", ""]
    for submission in submissions:
        official_values={row.get("column_key"):row.get("score") for row in submission["scores"] if isinstance(row,dict)}
        official_note=f"Official DIII-D **S = {float(official_values['d3d_S']):.6f}**. No rank is inferred from this score." if "d3d_S" in official_values else "No official DIII-D score is recorded yet."
        rank=submission.get("first_rank_observation")
        rank_text=f" First verified public rank: **#{rank['rank']}** at {rank['observed_utc']}." if rank else ""
        lines += [f"Official **{submission['method']}** submission **{submission['submission_id']}**: **{submission['status']}**, created {submission['created_utc']}. {official_note}{rank_text} Local scores remain in the fold-specific tables below. [Codabench competition](https://www.codabench.org/competitions/17739/).",""]
    coverage=next((row.get("quality_provenance",{}).get("public_test_input_only_coverage") for row in rows
                   if row.get("quality_provenance",{}).get("public_test_input_only_coverage")),None)
    if coverage:
        lines += [f"The fixed diagnostic-quality rule activates on **{coverage['gated_frames']:,}/{coverage['frames']:,} public-test input frames**, across **{coverage['gated_shots']}/{coverage['shots']} shots**. These are input-only coverage counts, not prediction scores; no public-test labels are used.",""]
    for group in ordered_groups(groups, rows):
        selected = sorted((row for row in rows if row["fold_id"] == group["fold_id"]), key=lambda row: row["metrics"]["S"], reverse=True)
        lines += [f"## {group['label']}", "", f"{group['n_shots']} shots; {group['n_frames']} evaluated native frames. Scorer {group['scorer_version']}. Selection fingerprint `{group['fold_id']}`.", "",
                  "| Learned model | S | Flux R² | q95 R² | βN R² | D_LCFS ↓ | Consistency | Training frames | Source |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in selected:
            score = row["metrics"]
            each = score.get("r2_each", {})
            frame_count = f"{row['train_frames']:,}" if isinstance(row["train_frames"], int) else "—"
            source = Path(os.path.relpath(row["source"], output)).as_posix()
            lines.append("| " + " | ".join([row["name"].replace("|", "\\|"), md_number(score["S"]),
                md_number(score["r2_psi"]), md_number(each.get("q95")), md_number(each.get("betaN")),
                md_number(score["dlcfs"]), md_number(score["consistency"]), frame_count,
                f"[JSON]({source})"]) + " |")
        lines.append("")
        if group["label"].startswith("Independent audit"):
            lines += ["This is the single authorized evaluation of the frozen v5 candidate. It was not used for fitting, checkpoint or blend selection. The gate had zero activations, so this audit does not independently test the low-quality fallback. [Protocol](../audit_frozen_v5/protocol.json).",""]
        for row in selected:
            if "selection_provenance" in row:
                provenance = row["selection_provenance"]
                source = Path(os.path.relpath(provenance["summary_path"], output)).as_posix()
                weights = provenance["selected_weights"]
                lines += [f"**{row['name']}** selects global weights on this same validation fold from {provenance['candidate_count']} candidates; this is not an independent holdout result. Selected source-A fractions: mean {weights['mean_weight_a']}, centered shape {weights['shape_weight_a']}, scalars {weights['scalar_weight_a']}. [Blend provenance]({source}).", ""]
            if "quality_provenance" in row:
                provenance=row["quality_provenance"]
                source=Path(os.path.relpath(provenance["summary_path"],output)).as_posix()
                if provenance["method"]=="gate":
                    text=f"The fixed gate activates on {provenance['gated_frames']} of {provenance['frames']} frames."
                    if provenance["gated_frames"]==0:
                        text+=" It does not test the low-quality branch on this fold."
                else:
                    expert=provenance.get("expert","Full POD")
                    secondary=provenance.get("secondary_expert","currents-only POD")
                    text=f"The 75% {expert} / 25% {secondary} weight was fixed before scoring; it was not selected through a grid search."
                    if provenance["method"]=="gated_blend75":
                        text+=f" The quality rule overrides the blend with {secondary} predictions on {provenance['gated_frames']} of {provenance['frames']} frames."
                lines += [f"**{row['name']}:** {text} [Experiment provenance]({source}).",""]
    if not any(group["label"].startswith("Fresh") for group in ordered_groups(groups, rows)):
        lines += ["## Fresh 128-shot validation", "", "No learned-model result has been saved for this fold yet.", ""]
    lines += ["## Oracle compression diagnostics — not model predictions", "",
              "These diagnostics use held-out labels to compute reconstruction coefficients. They measure representation capacity and cannot be submitted as learned predictions.", ""]
    if oracle:
        lines += ["| Diagnostic | Evaluation fold | S | Flux R² | Consistency | Source |", "|---|---|---:|---:|---:|---|"]
        for row in oracle:
            source = Path(os.path.relpath(row["source"], output)).as_posix()
            lines.append(f"| {row['name']} | {row['fold_label']} | {row['metrics']['S']:.6f} | {row['metrics']['r2_psi']:.6f} | {row['metrics']['consistency']:.6f} | [JSON]({source}) |")
        lines.append("")
    else:
        lines += ["No oracle compression scores saved yet.", ""]
    compression_plot = results / "optimization/pod_compression_curve/compression_curve.png"
    if compression_plot.exists():
        link = Path(os.path.relpath(compression_plot, output)).as_posix()
        lines += [f"[Separate oracle compression curve]({link}) · [Diagnostic details]({Path(link).with_name('COMPRESSION.md').as_posix()}).", ""]
    lines += ["## Figures", "", "- [Score progression](score_progression.png) · [PDF](score_progression.pdf)",
              "- [Weighted error components](score_deficits.png) · [PDF](score_deficits.pdf)",
              "- [Training histories](training_histories.png) · [PDF](training_histories.pdf)", "",
              "History plots use separate axes because normalized objectives may differ by model. Epoch-zero placeholder training loss is omitted. A score history, when available, is shown on the right axis.", "",
              "Refresh with `.venv/bin/python scripts/optimization/plot_progress.py`.", ""]
    if warnings:
        lines += ["## Read/consistency warnings", "", "Files can be temporarily incomplete while training writes them. Rerun to refresh.", ""]
        lines += [f"- {warning}" for warning in warnings]
    (output / "PROGRESS.md").write_text("\n".join(lines) + "\n")
    payload = {"generated_utc": stamp, "scope": "local validation; not official leaderboard scores",
               "historical_public_reference": REFERENCE, "folds": list(groups.values()),
               "official_submission_status":submission,
               "official_submission_history":submissions,
               "learned_models": [{key: value for key, value in row.items() if key != "shot_ids"} for row in rows],
               "oracle_diagnostics": [{key: value for key, value in row.items() if key != "shot_ids"} for row in oracle],
               "history_figures": history_scope, "warnings": warnings}
    (output / "progress.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="Default: RESULTS/optimization/progress")
    parser.add_argument("--max-models", type=int, default=12, help="Maximum models per deficit panel; report includes all")
    parser.add_argument("--max-history", type=int, default=12, help="Maximum recently updated histories to plot")
    args = parser.parse_args()
    if min(args.max_models, args.max_history) < 1:
        parser.error("max-models and max-history must be positive")
    results = args.results.resolve()
    output = (args.output or results / "optimization/progress").resolve()
    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    rows, oracle, groups, _ = discover(results, output, warnings)
    score_progression(rows, groups, output)
    deficit_plot(rows, groups, output, args.max_models)
    history_scope = training_history(results, rows, output, warnings, args.max_history)
    report(rows, oracle, groups, results, output, warnings, history_scope)
    print(json.dumps({"output": str(output), "learned_scores": len(rows), "oracle_scores": len(oracle),
                      "validation_folds": len({row['fold_id'] for row in rows}), "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
