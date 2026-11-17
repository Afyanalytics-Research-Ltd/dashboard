"""Where Connect Studio keeps uploaded files: S3 when a bucket is configured, else the server's media folder.

With S3 (``CONNECT_STUDIO_S3_BUCKET``), S3 is the only durable copy of every upload and of expensive derived data
(a restored SQL dump). The server's disk is a scratch space per job: a job downloads what it needs into its own
workspace and the workspace is deleted when the job ends (services.opened). Big uploads go to S3 in chunks as a
multipart upload, so they never touch the server's disk either.

Settings (environment):
  CONNECT_STUDIO_S3_BUCKET     the bucket (private; objects are written with SSE-S3 encryption). Unset = local
  CONNECT_STUDIO_S3_PREFIX     key prefix inside it (default "connect-studio")
  CONNECT_STUDIO_S3_REGION     region (default: AWS_REGION / AWS_DEFAULT_REGION / the SDK's own lookup)
  CONNECT_STUDIO_S3_ENDPOINT_URL   an S3-compatible endpoint instead of AWS (MinIO, for local testing)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_PROFILE, or the server's instance role — boto3's usual chain
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client = None             # tests set this to a fake


def bucket() -> str:
    return (os.getenv("CONNECT_STUDIO_S3_BUCKET") or "").strip()


def enabled() -> bool:
    return bool(bucket())


def prefix() -> str:
    return (os.getenv("CONNECT_STUDIO_S3_PREFIX") or "connect-studio").strip().strip("/")


def client():
    global _client
    with _lock:
        if _client is None:
            import boto3
            from botocore.config import Config

            region = os.getenv("CONNECT_STUDIO_S3_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
            _client = boto3.client("s3", region_name=region,
                                   endpoint_url=(os.getenv("CONNECT_STUDIO_S3_ENDPOINT_URL") or None),
                                   config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                                                 connect_timeout=15, read_timeout=300))
        return _client


class StorageError(Exception):
    """S3 refused or is unreachable; the message is user-facing."""


def _wrap(action: str):
    def deco(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except StorageError:
                raise
            except Exception as e:
                code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
                logger.exception("connect-studio S3 %s failed", action)
                raise StorageError(f"S3 {action} failed ({code or type(e).__name__}): {str(e)[:200]}") from None
        return inner
    return deco


def source_prefix(source_id: int) -> str:
    return f"{prefix()}/sources/{source_id}"


def upload_key(source_id: int, upload_id: str, name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).name)[:180] or "file"
    return f"{source_prefix(source_id)}/files/{upload_id}/{safe}"


def cache_prefix(source_id: int) -> str:
    return f"{source_prefix(source_id)}/cache"


# ─── multipart upload (chunks straight from the browser) ──────────────────

@_wrap("start upload")
def start_multipart(key: str) -> str:
    return client().create_multipart_upload(Bucket=bucket(), Key=key, ServerSideEncryption="AES256")["UploadId"]


@_wrap("upload part")
def put_part(key: str, upload_id: str, number: int, data: bytes, md5_b64: str) -> str:
    """Part numbers are 1-based in S3; ContentMD5 makes S3 reject a corrupted part."""
    return client().upload_part(Bucket=bucket(), Key=key, UploadId=upload_id, PartNumber=number, Body=data,
                                ContentMD5=md5_b64)["ETag"]


@_wrap("finish upload")
def complete_multipart(key: str, upload_id: str, etags: list[str]) -> None:
    client().complete_multipart_upload(Bucket=bucket(), Key=key, UploadId=upload_id, MultipartUpload={
        "Parts": [{"ETag": e, "PartNumber": i} for i, e in enumerate(etags, start=1)]})


def abort_multipart(key: str, upload_id: str) -> None:
    try:
        client().abort_multipart_upload(Bucket=bucket(), Key=key, UploadId=upload_id)
    except Exception as e:  # already gone is fine
        logger.info("connect-studio: abort multipart %s: %s", key, e)


# ─── objects ──────────────────────────────────────────────────────────────

@_wrap("upload")
def put_file(path: Path, key: str) -> int:
    client().upload_file(str(path), bucket(), key, ExtraArgs={"ServerSideEncryption": "AES256"})
    return path.stat().st_size


@_wrap("upload")
def put_fileobj(fileobj, key: str) -> None:
    client().upload_fileobj(fileobj, bucket(), key, ExtraArgs={"ServerSideEncryption": "AES256"})


@_wrap("download")
def get_file(key: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    client().download_file(bucket(), key, str(tmp))
    os.replace(tmp, path)
    return path


def exists(key: str) -> bool:
    try:
        client().head_object(Bucket=bucket(), Key=key)
        return True
    except Exception:
        return False


@_wrap("list")
def list_prefix(pfx: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        kw = {"Bucket": bucket(), "Prefix": pfx.rstrip("/") + "/"}
        if token:
            kw["ContinuationToken"] = token
        page = client().list_objects_v2(**kw)
        out += [(o["Key"], int(o.get("Size") or 0)) for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return out
        token = page.get("NextContinuationToken")


@_wrap("delete")
def delete_keys(keys: list[str]) -> None:
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        if batch:
            client().delete_objects(Bucket=bucket(), Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True})


def delete_prefix(pfx: str) -> int:
    """Delete everything under a prefix; returns the bytes freed."""
    objs = list_prefix(pfx)
    delete_keys([k for k, _ in objs])
    return sum(s for _, s in objs)


class S3Artifacts:
    """The engine's artifact cache (e.g. a restored SQL dump) kept in S3, so the next job on another workspace
    downloads it instead of rebuilding it. fetch / store move a named set of files."""

    def __init__(self, pfx: str):
        self.pfx = pfx.rstrip("/")

    def fetch(self, name: str, files: dict[str, Path]) -> bool:
        keys = {fn: f"{self.pfx}/{name}/{fn}" for fn in files}
        if not all(exists(k) for k in keys.values()):
            return False
        for fn, path in files.items():
            get_file(keys[fn], path)
        return True

    def store(self, name: str, files: dict[str, Path]) -> None:
        for fn, path in files.items():
            put_file(path, f"{self.pfx}/{name}/{fn}")
