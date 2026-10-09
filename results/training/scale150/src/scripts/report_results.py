#!/usr/bin/env python3
"""Summarize completed runs from their recorded artifacts, without inferred scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/"data/processed/manifest.json").read_text())
    download=json.loads((ROOT/"data/raw/download_manifest.json").read_text())
    missing=[f["path"] for f in download["files"] if not (ROOT/"data/raw"/f["path"]).is_file() or (ROOT/"data/raw"/f["path"]).stat().st_size!=f["size"]]
    inventory={"repo":download["repo"],"revision":download["revision"],"expected_files":len(download["files"]),
               "completed_files":len(download["files"])-len(missing),"expected_bytes":sum(f["size"] for f in download["files"]),
               "missing_or_size_mismatch":missing,"complete":not missing}
    (ROOT/"data/download_summary.json").write_text(json.dumps(inventory,indent=2)+"\n")
    records=[]
    for name in ("deeponet","fourier_deeponet"):
        folder=ROOT/"results"/name
        record={"name":name}
        for key,path in (("run","run.json"),("score","score.json"),("metrics","validation_predictions.metrics.json"),("submission","validation.json")):
            record[key]=json.loads((folder/path).read_text())
        records.append(record)
    combined={"data":inventory,"models":records,"status":"local_results_and_submission_archives_ready",
              "official_submission_uploaded":False,"track":"Challenge 1: DIII-D",
              "validation_usage":"checkpoint selection and reported evaluation; not a separate final test"}
    (ROOT/"results/comparison.json").write_text(json.dumps(combined,indent=2)+"\n")
    text=["# First results: DeepONet and Fourier-DeepONet", "",
          "Both baselines are trained and evaluated locally, with complete Challenge 1 submission archives. **These are held-out validation scores, not Codabench leaderboard scores.** No submission has been uploaded.","",
          "| Model | Composite S ↑ | Flux R² ↑ | q95 R² ↑ | betaN R² ↑ | LCFS distance ↓ | Consistency ↑ | Training time |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        s,m,run=r["score"],r["metrics"],r["run"]
        label="DeepONet" if r["name"]=="deeponet" else "Fourier-DeepONet"
        text.append(f"| {label} | {s['S']:.4f} | {s['r2_psi']:.4f} | {m['r2_q95']:.4f} | {m['r2_betaN']:.4f} | {s['dlcfs']:.4f} | {s['consistency']:.4f} | {run['training_seconds']:.1f} s |")
    run=records[0]["run"]
    deep,fourier=records
    rmse_gain=100*(1-fourier["metrics"]["psi_rmse"]/deep["metrics"]["psi_rmse"])
    score_gain=fourier["score"]["S"]-deep["score"]["S"]
    text += ["",f"Same seed (2026), {run['train_shots']} training shots / {run['train_frames']:,} frames, and {run['validation_shots']} disjoint validation shots / {run['validation_frames']:,} frames. The validation set selects the best epoch and supplies these reported scores. Training used an {run['gpu']}; timings measure the training/validation loop, excluding data download and preprocessing.","",
             f"Fourier-DeepONet reduces flux RMSE by {rmse_gain:.1f}% and leads composite S by {score_gain:.4f}. DeepONet has the higher derived-scalar consistency score. The composite difference is small on this initial validation subset; the comparison does not use matched compute budgets.", "",
             "The composite uses the official version 3.2.0 scorer, including LCFS extraction and all seven derived scalars. Validation uses up to 32 uniformly selected original frames per shot; public-test submissions include every original EFIT frame.","",
             "## Artifacts", ""]
    for r in records:
        name=r["name"];folder=f"results/{name}"
        text += [f"- **{name}**: [submission ZIP]({folder}/submission.zip), [checkpoint]({folder}/best.pt), [score breakdown]({folder}/score.json), [submission validation]({folder}/validation.json). Best epoch: {r['run']['best_epoch']}; parameters: {r['run']['parameters']:,}."]
    text += ["", "![Target, predictions, and errors for a fixed held-out frame](results/field_comparison.png)","",
             "![Training and validation losses](results/training_curves.png)","", "## Data and checks", "",
             f"Downloaded {inventory['completed_files']:,}/{inventory['expected_files']:,} repository files ({inventory['expected_bytes']/1e9:.2f} GB), pinned to `{download['revision']}`. Full release: 7,041 DIII-D training shots, 874 DIII-D test shots, and 1,206 MAST test shots. See [download summary](data/download_summary.json).", "",
             "Only the stated DIII-D training subset was fitted. MAST files are downloaded for future work; these first submissions do not enter the zero-shot transfer track. Tests cover input-label separation, shot separation, timestamp correction, spectral coupling, missing labels, and output ordering. The real-data perfect-prediction scorer self-check returns S=1.0.", "",
             "Each ZIP was independently validated against all 874 raw test shots and contains exactly the expected 2,622 float32 arrays. Both flux maps and scalars are finite. Exact evaluated frame identities, failures, version metadata, and archive checksums are retained in the JSON artifacts.","",
             "## Upload", "",
             "Join [Codabench competition 17739](https://www.codabench.org/competitions/17739/) with your account, then upload either model's `submission.zip`. Official test targets are hidden; the leaderboard result can only be obtained after uploading. Development closes October 18, 2026; five submissions per day, 100 total. See [reproduction instructions](README.md) and [verified requirements](challenge_requirements.md).", ""]
    (ROOT/"RESULTS.md").write_text("\n".join(text))
    print("Wrote RESULTS.md, results/comparison.json, data/download_summary.json")
    if missing:raise SystemExit(f"Download incomplete: {len(missing)} files")


if __name__=="__main__":main()
