#!/usr/bin/env python3
# pipelines/setup_bisync_bulk.py
"""
One-time setup for warehouse_bisync's bulk path (big Snowflake tables ->
S3 parquet -> Databend, server side). Idempotent - safe to re-run.

Run it inside the Airflow worker (it needs the pipelines' Snowflake
ACCOUNTADMIN key and an AWS key that can manage IAM):

    docker exec -w /opt/airflow/dags dashboard-airflow-airflow-worker-1 \
        python /opt/airflow/pipelines/setup_bisync_bulk.py

It creates:
  AWS        IAM role snowflake-afya-bisync - may only read/write/delete
             s3://<backup bucket>/_bisync_stage/*, and may only be assumed by
             Snowflake's own principal with this integration's external ID.
  Snowflake  STORAGE INTEGRATION AFYA_BISYNC_S3 (allowed location: that prefix only)
             STAGE HOSPITALS.ANALYTICS.BISYNC_STAGE
then proves it works with a 1-row unload + removal.

No AWS secret is ever stored in Snowflake or shows up in its query history.
Once the stage exists, warehouse_bisync turns the bulk path on by itself
(tables >= BISYNC_BULK_MIN_ROWS, default 200k); remove the stage to turn it off.

AWS key: BACKUP_AWS_ACCESS_KEY_ID/SECRET, else AWS_ACCESS_KEY_ID/SECRET.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, "/opt/airflow/dags")
from pipelines_common import use_pipelines_dir  # noqa: E402

use_pipelines_dir()
import boto3  # noqa: E402

import platform_backup as B  # noqa: E402
import warehouse_bisync_dag as M  # noqa: E402

logging.disable(logging.CRITICAL)

ACCOUNT, BUCKET = B.ACCOUNT_ID, B.BUCKET
PREFIX = "_bisync_stage/"
ROLE = "snowflake-afya-bisync"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/{ROLE}"
INTEGRATION = "AFYA_BISYNC_S3"
STAGE = os.getenv("BISYNC_SNOWFLAKE_STAGE", "HOSPITALS.ANALYTICS.BISYNC_STAGE")


def trust(principal: str, external_id: str | None = None) -> str:
    st = {"Effect": "Allow", "Principal": {"AWS": principal}, "Action": "sts:AssumeRole"}
    if external_id:
        st["Condition"] = {"StringEquals": {"sts:ExternalId": external_id}}
    return json.dumps({"Version": "2012-10-17", "Statement": [st]})


def main() -> None:
    key_id, secret = B.aws_keys()
    iam = boto3.client("iam", aws_access_key_id=key_id or None, aws_secret_access_key=secret or None)

    try:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=trust(f"arn:aws:iam::{ACCOUNT}:root"),
                        Description=f"Snowflake {INTEGRATION}: unload to s3://{BUCKET}/{PREFIX}",
                        Tags=[{"Key": "Purpose", "Value": "warehouse-bisync"}])
        print(f"IAM role {ROLE} created")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"IAM role {ROLE} exists")
    iam.put_role_policy(RoleName=ROLE, PolicyName="bisync-stage-rw", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["s3:PutObject", "s3:GetObject", "s3:GetObjectVersion", "s3:DeleteObject",
                        "s3:DeleteObjectVersion"],
             "Resource": f"arn:aws:s3:::{BUCKET}/{PREFIX}*"},
            {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
             "Resource": f"arn:aws:s3:::{BUCKET}",
             "Condition": {"StringLike": {"s3:prefix": [f"{PREFIX}*"]}}}]}))
    print(f"role policy: s3://{BUCKET}/{PREFIX}* only")

    conn = M._snowflake_conn("HOSPITALS")
    cur = conn.cursor()
    cur.execute(f"""CREATE STORAGE INTEGRATION IF NOT EXISTS {INTEGRATION} TYPE = EXTERNAL_STAGE
        STORAGE_PROVIDER = 'S3' ENABLED = TRUE STORAGE_AWS_ROLE_ARN = '{ROLE_ARN}'
        STORAGE_ALLOWED_LOCATIONS = ('s3://{BUCKET}/{PREFIX}')
        COMMENT = 'warehouse_bisync bulk path: Snowflake -> S3 parquet -> Databend'""")
    cur.execute(f"DESC STORAGE INTEGRATION {INTEGRATION}")
    props = {r[0]: r[2] for r in cur.fetchall()}
    iam.update_assume_role_policy(RoleName=ROLE, PolicyDocument=trust(props["STORAGE_AWS_IAM_USER_ARN"],
                                                                        props["STORAGE_AWS_EXTERNAL_ID"]))
    print(f"role trust locked to Snowflake's principal + {INTEGRATION}'s external id")
    cur.execute(f"""CREATE STAGE IF NOT EXISTS {STAGE} URL = 's3://{BUCKET}/{PREFIX}'
        STORAGE_INTEGRATION = {INTEGRATION} FILE_FORMAT = (TYPE = PARQUET)
        COMMENT = 'warehouse_bisync bulk unloads (files are purged after each load)'""")
    print(f"stage {STAGE} ready")

    for attempt in range(12):  # IAM trust changes take a little while to propagate
        try:
            cur.execute(f"COPY INTO @{STAGE}/_probe/ FROM (SELECT 1 AS X) "
                        "FILE_FORMAT = (TYPE = PARQUET) HEADER = TRUE OVERWRITE = TRUE")
            cur.execute(f"REMOVE @{STAGE}/_probe/")
            print("probe unload + remove OK - bulk path is live")
            return
        except Exception as e:  # noqa: BLE001
            print(f"probe attempt {attempt + 1}: {str(e)[:120]} - retrying in 10s")
            time.sleep(10)
    raise SystemExit("probe never succeeded - check the role trust policy / integration")


if __name__ == "__main__":
    main()
