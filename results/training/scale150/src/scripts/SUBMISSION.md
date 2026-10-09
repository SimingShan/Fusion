# Codabench submission monitoring

Verified against the live OpenAPI schema and official Codabench source on 2026-09-13. Competition **17739**, Development phase **29580**, task **35328**, leaderboard **19740**, queue **926**.

## Status and scores

Poll `GET https://www.codabench.org/api/submissions/{id}/` using the account's `Authorization: Token ...` header. Polling every 20–30 seconds is sufficient. Inspect:

- `status` and `status_details` for progress or failure explanations.
- `has_children`, `children`, and `parent` for task execution structure.
- `scores`, an array whose entries contain `column_key` and a decimal-string `score`.
- `on_leaderboard` to confirm placement after scoring.

The creation response's `id` is the submission to monitor. This challenge has one task, so normally the same ID runs and receives scores without children. Multiple tasks or multiple participant-group queues can create child submissions asynchronously. If `children` is nonempty, inspect every child's status and files. Child scores are also attached to the parent; a parent can become `Finished` once all children reach any terminal state, including failure, so parent status alone is insufficient in that case.

Possible states include `Submitting`, `Submitted`, `Preparing`, `Running`, `Scoring`, `Awaiting validation`, `Finished`, `Failed`, and `Cancelled`; not every submission traverses every state. Scores can arrive before the final status update.

## Logs and result files

`GET https://www.codabench.org/api/submissions/{id}/get_details/` returns:

| Field | Contents |
|---|---|
| `logs` | Entries with `name` and `data_file`, a signed download URL |
| `scoring_result` | Signed URL for the scoring-output ZIP, if available |
| `detailed_result` | Signed URL for detailed results, commonly HTML |
| `prediction_result` | Prediction-stage output archive, potentially large |
| `data_file` | Original submitted ZIP, approximately 3 GB here |

Useful log names include `scoring_stdout`, `scoring_stderr`, `scoring_ingestion_stdout`, and `scoring_ingestion_stderr`. Earlier-stage logs use `prediction_` prefixes. Empty or missing logs can occur before the worker writes them. Refresh `get_details` if signed URLs expire. Fetch storage URLs without forwarding the Codabench authorization header, and avoid recording signed URLs in public reports.

Download `scoring_result` and relevant logs to inspect results; downloading `data_file` or `prediction_result` is unnecessary for routine monitoring. The Development phase currently exposes output logs/results. If children exist, query each child's `get_details` for its execution logs.

`GET https://www.codabench.org/api/submissions/{id}/get_detail_result/` separately returns the detailed-result URL when enabled. `/api/upload_submission_scores/{id}/` is a **worker write endpoint**, not a participant results-query endpoint.

## Leaderboard verification

Use `GET https://www.codabench.org/api/leaderboards/19740/`. Its `submissions` array contains submission IDs, owners, and decimal-string scores without the display's four-decimal rounding. The primary metric is `d3d_S`; sort descending by that value when checking rank. Other keys are:

`g_ratio`, `mast_S`, `d3d_r2_psi`, `d3d_r2_qb`, `d3d_dlcfs`, `d3d_consistency`, `mast_r2_psi`, `mast_r2_qb`, `mast_dlcfs`, `mast_consistency`.

The phase's display endpoint is `GET https://www.codabench.org/api/phases/29580/get_leaderboard/?page_size=all`; use the leaderboard endpoint above for precise score comparisons.

The live submission rule is **`Force_Last`**: scoring automatically places the latest submission on the leaderboard. Ordinary users cannot manually change placement under this rule. `toggle_public` is a separate operation that exposes downloadable submission artifacts; it is not needed to appear on the leaderboard.

## Upload response reminder

`POST /api/datasets/` returns **`key` and `sassy_url`, not a numeric `id`**. Persist that response privately before doing anything else. The supported sequence is signed ZIP upload, `PUT /api/datasets/completed/{key}/`, then `POST /api/submissions/`. Dataset upload alone is not a scored submission. The dataset update endpoint does not regenerate a lost signed PUT URL.

## Primary sources

- [Live API documentation](https://www.codabench.org/api/docs/) and [OpenAPI JSON](https://www.codabench.org/api/api/schema/?format=json).
- [Submission serializers and file links](https://github.com/codalab/codabench/blob/acf93fb512adb607a2c7d0c4f8ae4c8523b662dd/src/apps/api/serializers/submissions.py).
- [Submission endpoints and leaderboard rules](https://github.com/codalab/codabench/blob/acf93fb512adb607a2c7d0c4f8ae4c8523b662dd/src/apps/api/views/submissions.py).
- [Task/queue child creation](https://github.com/codalab/codabench/blob/acf93fb512adb607a2c7d0c4f8ae4c8523b662dd/src/apps/competitions/tasks.py#L367-L474).
- [Parent status and score aggregation](https://github.com/codalab/codabench/blob/acf93fb512adb607a2c7d0c4f8ae4c8523b662dd/src/apps/competitions/models.py#L696-L730).
