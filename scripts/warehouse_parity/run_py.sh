#!/bin/bash
# Run one of these scripts inside the prod web container, at low priority,
# printing the host load before and after.
#
#   PROD_HOST=ubuntu@<prod-ip> PROD_SSH_KEY=~/.ssh/<key>.pem \
#     scripts/warehouse_parity/run_py.sh audit.py /tmp/databend_audit.json
#
# The script runs with the container's own Snowflake (SNOWFLAKE_ADMIN_*) and
# Databend settings - no credentials leave the server.
set -euo pipefail
: "${PROD_HOST:?set PROD_HOST=user@host}"
: "${PROD_SSH_KEY:?set PROD_SSH_KEY=path to the ssh key}"
CONTAINER=${CONTAINER:-dashboard-web-1}
here=$(cd "$(dirname "$0")" && pwd)
f=$1; shift; b=$(basename "$f")
scp -q -o BatchMode=yes -i "$PROD_SSH_KEY" "$here/$f" "$PROD_HOST:/tmp/$b"
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -i "$PROD_SSH_KEY" "$PROD_HOST" "
echo \"load before: \$(cut -d' ' -f1-3 /proc/loadavg) | mem avail \$(free -m | awk '/Mem/{print \$7}') MB\"
docker cp -q /tmp/$b $CONTAINER:/tmp/$b
docker exec -w / $CONTAINER nice -n 19 python /tmp/$b $* 2>&1 | grep -vi 'warning\|Snowflake Connector for Python\|Connecting to GLOBAL'
docker exec $CONTAINER rm -f /tmp/$b; rm -f /tmp/$b
echo \"load after: \$(cut -d' ' -f1-3 /proc/loadavg)\"
"
