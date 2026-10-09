#!/usr/bin/env python3
"""Download the public challenge at a pinned revision, with verified, atomic files."""
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "Sophelio/fusion-equilibrium-challenge"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/raw"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--first-train-shots", type=int, default=288)
    p.add_argument("--stage", choices=["initial", "test", "all"], default="all")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        api = HfApi()
        info = api.dataset_info(REPO, files_metadata=True)
        manifest = {"repo": REPO, "revision": info.sha, "files": [
            {"path": f.rfilename, "size": f.size,
             "sha256": getattr(f.lfs, "sha256", None) if f.lfs else None}
            for f in info.siblings]}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files = manifest["files"]
    train = sorted([f for f in files if f["path"].startswith("data/diii_d_train/")], key=lambda f: f["path"])
    test = sorted([f for f in files if f["path"].startswith("data/diii_d_public_test/")], key=lambda f: f["path"])
    initial = train[:args.first_train_shots]
    remainder = [f for f in files if f not in initial and f not in test]
    stages = [("initial", initial), ("test", test), ("remainder", remainder)]
    if args.stage == "initial": stages = stages[:1]
    if args.stage == "test": stages = stages[1:2]

    def fetch(f):
        dest = args.output / f["path"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            return f["path"]
        for attempt in range(5):
            try:
                path = hf_hub_download(REPO, f["path"], repo_type="dataset",
                    revision=manifest["revision"], local_dir=args.output)
                if Path(path).stat().st_size != f["size"]:
                    raise ValueError("Downloaded size mismatch: " + f["path"])
                if f["sha256"]:
                    h = hashlib.sha256()
                    with open(path, "rb") as stream:
                        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                            h.update(block)
                    if h.hexdigest() != f["sha256"]:
                        raise ValueError("Downloaded SHA256 mismatch: " + f["path"])
                return f["path"]
            except Exception:
                if attempt == 4: raise
                time.sleep(2 ** attempt)

    print(f"Pinned revision {manifest['revision']}; {len(files)} files, {sum(f['size'] for f in files)/1e9:.2f} GB", flush=True)
    for name, selected in stages:
        started = time.time()
        print(f"Starting {name}: {len(selected)} files", flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            pending = [pool.submit(fetch, f) for f in selected]
            for count, future in enumerate(concurrent.futures.as_completed(pending), 1):
                future.result()
                if count % 25 == 0 or count == len(selected):
                    print(f"{name}: {count}/{len(selected)} files, {time.time()-started:.0f}s", flush=True)
        (args.output / f"{name}.complete").write_text(manifest["revision"] + "\n")


if __name__ == "__main__":
    main()
