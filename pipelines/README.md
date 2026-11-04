# Data pipelines (Airflow)

The DAGs and pipeline scripts from the
[datapipelines](https://github.com/Afyanalytics-Research-Ltd/datapipelines) repo,
run by this repo's Airflow stack (`docker-compose.airflow.yaml`).

| Here | What | Inside the containers |
|---|---|---|
| `dags/` | DAG files, `pipelines_common.py`, `jsons/` | `/opt/airflow/dags` |
| `pipelines/` | the scripts the DAGs import, their tests, `requirements.txt` | `/opt/airflow/pipelines` (`PIPELINES_DIR`) |
| `config/airflow_local_settings.py` | Sentry + cluster policies (failure alerts, 1h run deadline) | `/opt/airflow/config` |
| `plugins/` | UI logo plugin | `/opt/airflow/plugins` |
| `Dockerfile.airflow` | stock Airflow + `pipelines/requirements.txt` + Chrome 114 | image `dashboard-airflow:3.3.0-pipelines` |

`pipelines/dags` is a symlink to `../dags` and
`pipelines/config/airflow_local_settings.py` to the real file in `config/`, so
every path the scripts and tests use (`ROOT/dags`, `dags/jsons/…`,
`config/rsa_key.p8`) is the same as in datapipelines.

## Not in git: secrets and migration state

`pipelines/.gitignore` keeps these out (datapipelines committed its state; this
repo doesn't):

- `pipelines/.env` – the pipelines' settings. **The Airflow containers load this
  file, not the Django `.env`.**
- `pipelines/config/rsa_key.p8`, `service_account.json`, `config/gdrive-sa.json`
- migration state: `.migration_state/`, `.migration_*.json`, watermark and progress
  files, `state/`, `input/`, `mapping/softdelete/`

Copy them from a datapipelines checkout on each machine:

```bash
scripts/sync_datapipelines.sh --with-state ~/datapipelines
```

Without `--with-state` the script refreshes only the code (`dags/`, `pipelines/`,
`plugins/`, `config/airflow_local_settings.py`). It copies the working tree,
never deletes anything here, and is safe to re-run.

## Running

`.env` (the repo root one, read by compose for `${…}` substitution) needs:

```bash
AIRFLOW_UID=1000          # `id -u` of the checkout's owner; airflow-init chowns pipelines/ to it
AIRFLOW_HOST_PORT=8080    # 8081 while the old datapipelines stack still holds 8080
```

```bash
docker compose -f docker-compose.airflow.yaml build
docker compose -f docker-compose.airflow.yaml up -d
docker compose -f docker-compose.airflow.yaml exec airflow-scheduler airflow dags list-import-errors
```

New DAGs start paused. One-time setup: `document_pipeline` runs its file tasks in a
pool that must exist, or they're never scheduled:

```bash
docker compose -f docker-compose.airflow.yaml exec airflow-scheduler \
  airflow pools set landingai_api_pool 8 "Concurrent LandingAI file tasks"
```

Tests (they use `tests/airflow_stub.py`, not a real Airflow):

```bash
cd pipelines && python -m pytest tests
```

## Cutover from the datapipelines Airflow: one stack at a time

The migration scripts prevent concurrent runs with `flock` on files in the state
directory. Two Airflow stacks, each with its **own copy** of the state, can't see
each other's locks. Both could migrate the same records and create duplicates in V3.
Only one stack may run these DAGs at a time:

1. Wait for running DAG runs in the old stack to finish, or pause them.
2. Stop it: `cd ~/datapipelines && docker compose down`.
3. Copy the final state: `scripts/sync_datapipelines.sh --with-state ~/datapipelines`.
4. Copy the Airflow Variables and Connections. Some DAGs read credentials from
   them, and the `v3__*` Variables are watermarks. The script needs both stacks
   up, so start just the old scheduler for it:
   `(cd ~/datapipelines && docker compose up -d airflow-scheduler)`, run
   `scripts/copy_airflow_metadata.sh`, then `docker compose down` there again.
5. Set `AIRFLOW_HOST_PORT=8080` (or drop it) and `docker compose -f docker-compose.airflow.yaml up -d`.
6. Unpause the DAGs you use. Run history in the old metadata DB is not copied.

### The dashboard's Pipelines pages (`/pipelines/`)

Django talks to Airflow through `AIRFLOW_BASE_URL`. When Django runs on the host
(`manage.py runserver`), it must be the **published** port — `http://localhost:8080` —
never a container name like `http://airflow-apiserver:8080`, which only resolves
inside Airflow's own Docker network (the symptom is a 502 "Failed to resolve
'airflow-apiserver'").

`http://localhost:8080` stays correct across the cutover: today it reaches the
datapipelines stack (API at the root), afterwards this stack (API under `/airflow`,
from `AIRFLOW__WEBSERVER__BASE_URL`). The client tries the other layout when
`/auth/token` 404s/405s, so no `.env` change is needed at cutover. Restart Django
after editing `.env`; if your shell exported `.env` (e.g. `set -a; . .env`),
re-source it first, because exported variables win over the file.
