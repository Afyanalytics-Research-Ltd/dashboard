# Book translation pipeline on Airflow

Operational tools of the **book-translation-pipeline** repo (Kenyan-language ASR / TTS / MT trained on Kaggle,
models kept on Google Drive), run by `dags/book_translation_dag.py`.

| Path | What |
|---|---|
| `tools/`, `notebooks/` | verbatim export from book-translation-pipeline (`tools/export_to_airflow.sh` there); revision in `SOURCE_REVISION` — **do not edit here**, change the source repo and re-export |
| `btp_airflow.py` | the Airflow wiring: credentials → env / temp file, running a tool, input checks |
| `../state/book_translation/` | scheduler state + log (git-ignored, survives restarts) |

## DAGs

| DAG | Schedule | Does |
|---|---|---|
| `btp_kaggle_scheduler_tick` | every 30 min | records each finished Kaggle run, copies its models to `My Drive/btp_workspace/models_kaggle` |
| `btp_kaggle_scheduler_admin` | manual | `status`, `resume-schedule` (clear a halt), `monitor-only` |
| `btp_kaggle_status` | manual | run status, notebook version, weekly GPU quota |
| `btp_kaggle_push` | manual | start a GPU run (`dry_run` on by default; refuses while a version is running/queued) |
| `btp_kaggle_logs` | manual | a version's logs, `run_summary.json`, stage results |
| `btp_models_to_drive` | manual | copy finished models + reports to Drive (idempotent, MD5-verified) |
| `btp_colab_logs` | manual | Colab run logs / reports from `My Drive/btp_workspace` |
| `btp_colab_notebook_push` | manual | replace the Colab notebook on Drive (`dry_run` on by default) |

New DAGs start paused. Unpause `btp_kaggle_scheduler_tick` on **one** server only (production) — two schedulers
would record and copy every run twice.

## Setup (Admin → Connections / Variables)

- Connection `kaggle_default` (Generic): Login = Kaggle username, Password = legacy API key,
  Extra `{"api_token": "KGAT_..."}`. Passed to the tools only as environment variables.
- Variable `btp_gdrive_token_json`: Google OAuth `authorized_user` JSON for the Drive holding `btp_workspace`.
  Written to a 0600 temp file for the task and deleted afterwards.
- Scheduler state: copy `~/.local/state/btp/scheduler.json` from the machine that ran the scheduler to
  `pipelines/state/book_translation/scheduler.json`. Without it `btp_kaggle_scheduler_tick` refuses to run
  (a fresh state would start launching GPU runs).

## Updating the tools

```bash
cd ~/audiobookmanager/book-translation-pipeline
tools/export_to_airflow.sh ~/dashboard     # only committed, notebook-in-sync sources are exported
```
then commit `pipelines/book_translation/` here and promote staging → main as usual.

Not exported: `ci.sh`, `release.sh` (development: test suite with torch, Colab release from a git checkout),
and `kaggle_run.py secrets` (needs the W&B key from the developer's `.env`).
