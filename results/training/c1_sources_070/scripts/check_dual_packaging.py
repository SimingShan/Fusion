"""Check machine identity/frame contracts and atomic dual-archive packaging."""
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from package_submission import validate_prediction, file_signature, sha256_file
from package_dual_submission import package_pair


def main():
    with tempfile.TemporaryDirectory(prefix="fusion_dual_check_") as td:
        root = Path(td)
        raw = root / "raw"
        raw.mkdir()
        items, predictions = [], {}
        for config, n in [("diii_d_public_test", 2), ("mast_public_test", 3)]:
            rel = f"data/{config}/shot.parquet"
            path = raw / rel
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{
                "efit_times": np.arange(n, dtype=float).tolist(),
                "efit_grid_R": np.linspace(.1, 2, 65).tolist(),
                "efit_grid_Z": np.linspace(-2, 2, 65).tolist(),
            }]), path)
            items.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
            pred = root / f"{config}.npz"
            np.savez_compressed(pred,
                                shot_0000_psirz=np.full((n, 65, 65), n, np.float32),
                                shot_0000_q95=np.full(n, 3, np.float32),
                                shot_0000_betaN=np.full(n, 1, np.float32))
            predictions[config] = pred
        (raw / "download_manifest.json").write_text(json.dumps({
            "repo": "Sophelio/fusion-equilibrium-challenge", "revision": "a" * 40, "files": items}))
        for config, pred in predictions.items():
            report = validate_prediction(pred, raw, expected_shots=1, config=config)
            assert report["frames"] == (2 if config.startswith("diii") else 3)
        try:
            validate_prediction(predictions["diii_d_public_test"], raw, expected_shots=1, config="mast_public_test")
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-machine frame mismatch was accepted")
        signatures = {k: file_signature(v) for k, v in predictions.items()}
        output = root / "submission.zip"
        package_pair(predictions, output, signatures)
        with zipfile.ZipFile(output) as archive:
            assert set(archive.namelist()) == {k + ".npz" for k in predictions}
            for config, pred in predictions.items():
                assert hashlib.sha256(archive.read(config + ".npz")).hexdigest() == sha256_file(pred)
        before = sha256_file(output)
        with predictions["mast_public_test"].open("ab") as stream:
            stream.write(b"changed")
        try:
            package_pair(predictions, output, signatures)
        except ValueError:
            pass
        else:
            raise AssertionError("Post-validation mutation was accepted")
        assert sha256_file(output) == before
    print("PASS: machine/frame contracts, swapped-machine rejection, byte-exact ZIP, mutation protection")


if __name__ == "__main__":
    main()
