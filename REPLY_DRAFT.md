Dear FusionEC-26 organizers,

Thank you for the update. The reproduction package for Codabench Final submission **1003861** is the public Git repository

**https://github.com/SimingShan/Fusion @ `1951284b9f21c9f641c4250d0d04f36a1e58ffeb`**

(MIT licence). `SUBMISSION_1003861.md` at the repository root maps each item of rule 6 to its location:

- **Source:** `fusion/`, `scripts/`, `train_members.py` and the referenced training/analysis scripts.
- **Environment:** `env/environment.yml`, with the full pip freeze in `env/pip_freeze_full.txt`.
- **Trained weights of this submission:** `weights_rebuild/`, with checksums in `SHA256SUMS`.
- **Training entrypoint:** `bash prepare.sh && bash train.sh`.
- **Inference entrypoint for the Final inputs:** `scripts/fetch_final_inputs.sh`, then `prepare_test.sh`, then `predict.sh` with `WEIGHTS=weights_rebuild`. The output zip has SHA-256 `a45366f935c560380a468dd8ea00d3d31578f1eb3ef9699c6a66455ba0a70456`.
- **Data provenance:** `PROVENANCE.md`. The entry is DIII-D only; no MAST data and no external data or pre-trained models were used.

**Compute declaration:**

- GPU: 17.4 GPU-hours on one NVIDIA A100 80GB PCIe, measured. That is 24 HLVF members 5.79 h, 40 HLAV members 10.61 h, direct regressors 0.27 h, closure 0.76 h. All of it is DIII-D; there is no GPU work for MAST, and inference needs no GPU.
- CPU, not budgeted: data preparation takes about 4–6 h on 8–16 cores plus about 1.5 h for the li stores. Inference on the Final inputs takes about 1.6 h on 64 cores.
- There is no synthetic data generation.

The methods report, which also covers data use and our use of AI coding assistance (Claude Code), is `METHODS.md` in the same commit. I'm happy to send it as a PDF if you prefer.

Best regards,
Siming Shan (Lu Group)
