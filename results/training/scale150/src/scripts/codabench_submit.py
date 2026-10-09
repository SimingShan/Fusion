#!/usr/bin/env python3
"""Upload a validated result directly to Codabench, with private resume state.

The password is never needed here. Authentication uses the token written by
codabench_login.py. Signed storage URLs remain in a mode-0600 file outside the
repository. Submission POSTs are not retried after an ambiguous response.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlparse

import requests

BASE = "https://www.codabench.org"


def write_json(path, value, private=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        os.fchmod(f.fileno(), 0o600 if private else 0o644)
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class UploadFile:
    def __init__(self, stream, size):
        self.stream, self.size = stream, size
        self.sent = 0
        self.start = self.last = time.monotonic()

    def __len__(self):
        return self.size

    def read(self, size=-1):
        block = self.stream.read(size)
        self.sent += len(block)
        now = time.monotonic()
        if now - self.last >= 20 or (not block and self.sent == self.size):
            elapsed = max(now - self.start, 0.001)
            print(f"Upload {self.sent / self.size:.1%}: {self.sent:,}/{self.size:,} bytes; "
                  f"{self.sent / elapsed / 1e6:.2f} MB/s", flush=True)
            self.last = now
        return block


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("results/optimization/pod_full_v1"))
    p.add_argument("--phase", type=int, default=29580)
    p.add_argument("--competition", type=int, default=17739)
    p.add_argument("--name", default="Submission " + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--description", default="Equilibrium predictions.")
    p.add_argument("--expected-sha256")
    p.add_argument("--status", action="store_true")
    p.add_argument("--watch-minutes", type=float, default=0, help="Poll status every 30 seconds for this many minutes")
    args = p.parse_args()
    if not re.fullmatch(r"Submission [0-9][0-9-]{2,39}", args.name):
        p.error("Public names must use a neutral numeric identifier: Submission YYYYMMDD-NNN")
    if args.description != "Equilibrium predictions.":
        p.error("Public descriptions are fixed to protect implementation confidentiality")
    run = args.run_dir.resolve()
    token_dir = Path.home() / ".config/fusion"
    token = (token_dir / "codabench_token").read_text().strip()
    auth = requests.Session()
    auth.headers.update({"Authorization": "Token " + token, "Accept": "application/json"})
    state_path = token_dir / "submissions" / f"{args.competition}_{run.name}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    last_printed = None

    def save():
        write_json(state_path, state, private=True)
        safe = {k: v for k, v in state.items() if k not in {"sassy_url"}}
        write_json(run / "codabench_submission.json", safe)

    def api(method, path, body=None):
        try:
            r = auth.request(method, BASE + path, json=body, timeout=(30, 90))
        except requests.RequestException:
            raise RuntimeError(f"Codabench {method} {path} connection failed; saved state retained.") from None
        if not r.ok:
            # Only selected error fields; redact URLs and credentials defensively.
            detail = ""
            try:
                errors = r.json()
                if isinstance(errors, dict):
                    errors = {k: v for k, v in errors.items() if k in {
                        "detail", "error", "data_file", "non_field_errors", "phase", "fact_sheet_answers"}}
                    detail = re.sub(r"https?://[^\s\"<>]+", "[URL omitted]", json.dumps(errors))
                    detail = detail.replace(token, "[credential omitted]")[:1000]
            except ValueError:
                pass
            raise RuntimeError(f"Codabench {method} {path} returned HTTP {r.status_code}. {detail}")
        return r.json() if r.content else {}

    def status():
        nonlocal last_printed
        sid = state.get("submission_id")
        if not sid:
            raise RuntimeError("No submission ID recorded yet.")
        result = api("GET", f"/api/submissions/{sid}/")
        state["official"] = {k: result.get(k) for k in [
            "id", "status", "status_details", "created_when", "scores", "phase",
            "on_leaderboard", "leaderboard", "children", "parent", "owner"]}
        state["last_checked_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        if state["official"] != last_printed:
            print(json.dumps(state["official"], indent=2), flush=True)
            last_printed = state["official"].copy()
        return result

    if args.status or state.get("submission_id"):
        result = status()
        deadline = time.monotonic() + args.watch_minutes * 60
        while result.get("status") not in {"Finished", "Failed", "Cancelled"} and time.monotonic() < deadline:
            time.sleep(min(30, max(0, deadline - time.monotonic())))
            result = status()
        return
    if state.get("submission_post_started"):
        raise RuntimeError("A submission POST was attempted without a recorded ID. Inspect account submissions before retrying.")
    report = json.loads((run / "validation.json").read_text())
    archive = run / "submission.zip"
    if report.get("status") != "passed" or Path(report["submission_zip"]).resolve() != archive:
        raise RuntimeError("Validation report does not match this submission ZIP.")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise RuntimeError("Submission SHA256 differs from the requested artifact.")
    if report.get("zip_sha256") and report["zip_sha256"] != digest:
        raise RuntimeError("Submission ZIP changed after structural validation.")
    if state.get("sha256", digest) != digest or state.get("phase", args.phase) != args.phase:
        raise RuntimeError("Resume state belongs to a different file or phase.")
    size = archive.stat().st_size
    competition = api("GET", f"/api/competitions/{args.competition}/")
    phase = next(x for x in competition["phases"] if x["id"] == args.phase)
    if competition.get("participant_status") != "approved" or phase["status"] != "Current":
        raise RuntimeError("Account approval or phase status does not permit submission.")
    for used, limit in [("used_submissions_per_day", "max_submissions_per_day"),
                        ("used_submissions_per_person", "max_submissions_per_person")]:
        if phase.get(limit) and phase.get(used, 0) >= phase[limit]:
            raise RuntimeError(f"Submission quota exhausted: {used}.")
    state.update({"archive": str(archive), "sha256": digest, "bytes": size,
                  "competition": args.competition, "phase": args.phase, "method_name": args.name})
    save()
    print(f"Validated archive: {size:,} bytes; SHA256 {digest}", flush=True)
    if not state.get("dataset_key"):
        if state.get("dataset_post_started"):
            raise RuntimeError("Dataset POST had an ambiguous response. Inspect account resources before retrying.")
        state["dataset_post_started"] = True
        save()
        result = api("POST", "/api/datasets/", {"type": "submission", "name": args.name,
                     "description": args.description,
                     "is_public": False, "request_sassy_file_name": "submission.zip",
                     "file_name": "submission.zip", "file_size": size, "competition": args.competition})
        state.update({"dataset_key": result["key"],
                      "sassy_url": result["sassy_url"]})
        save()
        print(f"Created private dataset {result['key']}", flush=True)
    if not state.get("upload_complete"):
        url = state["sassy_url"]
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise RuntimeError("Storage URL is not HTTPS.")
        headers = {"Content-Type": "application/zip", "Content-Length": str(size)}
        if (parsed.hostname or "").endswith(".blob.core.windows.net"):
            headers.update({"x-ms-blob-type": "BlockBlob", "x-ms-version": "2018-03-28"})
        with archive.open("rb") as stream:
            try:
                response = requests.put(url, data=UploadFile(stream, size), headers=headers,
                                        timeout=(30, 180), allow_redirects=False)
            except requests.RequestException as exc:
                # Report error types without leaking a signed URL embedded in exception text.
                kinds = []
                def inspect_error(error):
                    kinds.append(type(error).__name__)
                    for item in getattr(error, "args", ()):
                        if isinstance(item, BaseException):
                            inspect_error(item)
                inspect_error(exc)
                raise RuntimeError("Storage upload interrupted (" + "/".join(kinds) +
                                   "); no submission POST was sent. Private upload state retained.") from None
        if not response.ok:
            raise RuntimeError(f"Storage upload returned HTTP {response.status_code}; no submission POST was sent.")
        state["upload_complete"] = True
        save()
        print("Storage upload completed.", flush=True)
    if not state.get("dataset_complete"):
        api("PUT", f"/api/datasets/completed/{state['dataset_key']}/")
        state["dataset_complete"] = True
        state.pop("sassy_url", None)
        save()
        print("Codabench confirmed dataset completion.", flush=True)
    # The competition API stopped returning "queue" (2026-10-04); omit it so the phase default applies.
    body = {"data": state["dataset_key"], "phase": args.phase,
            "fact_sheet_answers": {"code_url": "", "method_name": args.name, "compute_light": False}}
    queue = (competition.get("queue") or {}).get("id")
    if queue is not None:
        body["queue"] = queue
    state["submission_post_started"] = True
    save()
    result = api("POST", "/api/submissions/", body)
    state["submission_id"] = result["id"]
    save()
    print(f"Created submission {result['id']}", flush=True)
    status()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"Submission stopped: {exc}", file=sys.stderr)
        sys.exit(1)
