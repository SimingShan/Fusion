#!/usr/bin/env python3
"""Upload an existing validated prediction file and create a tiny Codabench ZIP.

Run `./.venv/bin/hf auth login` on the server first, using write access. The
separate repo-scoped read token is requested without echoing or command history.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from push_predictions import push_and_write_pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Your private dataset repo, USERNAME/fusion-eq-predictions")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/optimization/pod_full_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    run = args.run_dir.resolve()
    report = json.loads((run / "validation.json").read_text())
    prediction = run / "diii_d_public_test.npz"
    if report.get("status") != "passed" or Path(report["prediction_path"]).resolve() != prediction:
        raise ValueError("Run does not have a matching passed submission validation report")
    if prediction.stat().st_size != report["input_bytes"]:
        raise ValueError("Prediction size changed after validation")
    stage = ROOT / "results/hf_upload" / run.name
    stage.mkdir(parents=True, exist_ok=True)
    staged_prediction = stage / prediction.name
    if not staged_prediction.exists():
        os.link(prediction, staged_prediction)
    if not os.path.samefile(prediction, staged_prediction):
        raise ValueError("Existing staging prediction differs from the selected run")
    if set(stage.glob("*.npz")) != {staged_prediction}:
        raise ValueError("Upload staging must contain only the intended public-test prediction")
    output = stage / "submission_pointer.zip"
    if args.dry_run:
        return push_and_write_pointer(args.repo, None, stage, output, dry_run=True)
    from huggingface_hub import HfApi
    from huggingface_hub.errors import LocalTokenNotFoundError
    api = HfApi()
    try:
        who = api.whoami()
    except LocalTokenNotFoundError:
        parser.exit(1, "Log in on this server first: .venv/bin/hf auth login\n")
    print(f"Uploading account: {who['name']}")
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise ValueError("Choose a private predictions repository")
    print(f"Private repository ready: https://huggingface.co/datasets/{args.repo}")
    print("Create a fine-grained READ token at https://huggingface.co/settings/tokens")
    print(f"Grant only read access to the contents of {args.repo}.")
    read_token = getpass.getpass("Repository-scoped read token (hidden): ").strip()
    if not read_token:
        parser.exit(1, "No token entered; predictions have not been uploaded.\n")
    return push_and_write_pointer(args.repo, read_token, stage, output)


if __name__ == "__main__":
    raise SystemExit(main())
