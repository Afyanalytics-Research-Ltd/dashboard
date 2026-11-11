#!/usr/bin/env bash
# Copy Airflow Variables and Connections from the datapipelines Airflow to this
# repo's Airflow (docker-compose.airflow.yaml). Some DAGs read credentials from
# them (snowflake_default, mysql_conn, aws_default, per-facility connections,
# GOOGLE_SA_JSON, ...), and the v3__* Variables are watermarks, i.e. state:
# copy them at cutover together with `sync_datapipelines.sh --with-state`.
#
#   scripts/copy_airflow_metadata.sh [OLD_CONTAINER] [NEW_CONTAINER]
#
# Existing keys/connections in the new stack are overwritten. The exported
# files hold secrets; they only exist in temp dirs and are removed on exit.
set -euo pipefail
OLD="${1:-datapipelines-airflow-scheduler-1}"
NEW="${2:-dashboard-airflow-airflow-scheduler-1}"

tmp=$(mktemp -d)
cleanup() {
  rm -rf "$tmp"
  docker exec "$OLD" rm -f /tmp/_vars.json /tmp/_conns.json 2>/dev/null || true
  docker exec "$NEW" rm -f /tmp/_vars.json /tmp/_conns.json 2>/dev/null || true
}
trap cleanup EXIT
chmod 700 "$tmp"

docker exec "$OLD" airflow variables export /tmp/_vars.json >/dev/null
docker exec "$OLD" airflow connections export /tmp/_conns.json --file-format json >/dev/null
docker cp -q "$OLD:/tmp/_vars.json" "$tmp/vars.json"
docker cp -q "$OLD:/tmp/_conns.json" "$tmp/conns.json"
docker cp -q "$tmp/vars.json" "$NEW:/tmp/_vars.json"
docker cp -q "$tmp/conns.json" "$NEW:/tmp/_conns.json"
docker exec "$NEW" airflow variables import /tmp/_vars.json --action-on-existing-key overwrite
docker exec "$NEW" airflow connections import /tmp/_conns.json --overwrite
echo "Copied $(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$tmp/vars.json") variables," \
     "$(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$tmp/conns.json") connections: $OLD -> $NEW"
