#!/usr/bin/env python
"""Copy finished models (and reports) from the latest Kaggle run into Google Drive.

    python book-translation-pipeline/tools/sync_models_to_drive.py --token gdrive_token.json            # finished models + reports
    python book-translation-pipeline/tools/sync_models_to_drive.py --token gdrive_token.json --dry-run  # list what would be copied
    python book-translation-pipeline/tools/sync_models_to_drive.py --token gdrive_token.json --include-checkpoints

Destination: My Drive/btp_workspace/models_kaggle/  (kept apart from Colab's btp_workspace/models/, so the
two platforms never overwrite each other).

Safe to run repeatedly:
  * files already on Drive with the same size + MD5 are skipped (idempotent, cheap to re-run);
  * uploads are resumable (chunked) and verified against Drive's own MD5 afterwards;
  * Kaggle output is downloaded one group at a time into a temporary folder that is deleted right after
    its files are uploaded and verified, so at most one model sits on local disk;
  * a manifest (Kaggle version, files, sizes, MD5s) is written next to the models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from kaggle_run import SLUG, USER, best_version  # noqa: E402

FOLDER_MIME = "application/vnd.google-apps.folder"
# network failures worth retrying (the upload session survives them)
RETRYABLE = (ssl.SSLError, ConnectionError, TimeoutError, socket.timeout, OSError)

# What to copy: finished models + reports. Patterns are matched against paths inside the Kaggle output.
GROUPS = {
    "mt": r"btp_workspace/models/mt/final/.*",
    "asr": r"btp_workspace/models/asr/[^/]+/final/.*",
    "tts": r"btp_workspace/models/tts/[^/]+/final/.*",
    "reports": r"btp_workspace/(reports|status)/.*",
}
CHECKPOINTS = {"checkpoints": r"btp_workspace/models/.*(checkpoint.*|/init/.*)"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Drive:
    """Minimal Drive helper: folder paths, listing with md5, verified resumable uploads."""

    def __init__(self, service):
        self.s = service
        self._folders: dict[tuple[str, str], str] = {}

    def folder(self, parent: str, name: str) -> str:
        key = (parent, name)
        if key not in self._folders:
            q = (f"name='{name.replace(chr(39), chr(92) + chr(39))}' and '{parent}' in parents and "
                 f"mimeType='{FOLDER_MIME}' and trashed=false")
            r = self.s.files().list(q=q, fields="files(id)").execute()["files"]
            fid = r[0]["id"] if r else self.s.files().create(
                body={"name": name, "parents": [parent], "mimeType": FOLDER_MIME}, fields="id").execute()["id"]
            self._folders[key] = fid
        return self._folders[key]

    def path(self, parts: list[str], root: str = "root") -> str:
        cur = root
        for p in parts:
            cur = self.folder(cur, p)
        return cur

    def existing(self, parent: str, name: str) -> dict | None:
        q = f"name='{name.replace(chr(39), chr(92) + chr(39))}' and '{parent}' in parents and trashed=false"
        r = self.s.files().list(q=q, fields="files(id,size,md5Checksum)").execute()["files"]
        return r[0] if r else None

    def upload(self, local: Path, parent: str, max_retries: int = 8, sleep=time.sleep) -> dict:
        """Resumable upload in 8 MB chunks. A dropped connection (2026-11-20/23: ssl.SSLEOFError in the
        middle of the 2.4 GB MT model) retries the same chunk with backoff instead of aborting the copy:
        the upload session remembers how many bytes Drive has, so nothing is re-sent from the start."""
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(str(local), resumable=True, chunksize=8 << 20)
        old = self.existing(parent, local.name)
        if old:
            req = self.s.files().update(fileId=old["id"], media_body=media, fields="id,size,md5Checksum")
        else:
            req = self.s.files().create(body={"name": local.name, "parents": [parent]}, media_body=media,
                                        fields="id,size,md5Checksum")
        resp, failures = None, 0
        while resp is None:
            try:
                _, resp = req.next_chunk(num_retries=3)      # num_retries covers HTTP 5xx / 429
                failures = 0
            except RETRYABLE as e:
                failures += 1
                if failures > max_retries:
                    raise
                wait = min(2 ** failures, 120)
                print(f"  {local.name}: {type(e).__name__} at {req.resumable_progress / 1e6:.0f} MB -- "
                      f"retry {failures}/{max_retries} in {wait}s", flush=True)
                sleep(wait)
        return resp


def sync_dir(drive: Drive, local_root: Path, dest_root: str, dry_run: bool = False) -> list[dict]:
    """Upload every file under local_root to dest_root (mirroring folders); skip identical files; verify MD5."""
    out = []
    for f in sorted(p for p in local_root.rglob("*") if p.is_file()):
        rel = f.relative_to(local_root)
        size, digest = f.stat().st_size, md5(f)
        entry = {"path": rel.as_posix(), "size": size, "md5": digest}
        if dry_run:
            out.append({**entry, "action": "would copy"})
            continue
        parent = drive.path(list(rel.parts[:-1]), root=dest_root)
        old = drive.existing(parent, f.name)
        if old and int(old.get("size", -1)) == size and old.get("md5Checksum") == digest:
            out.append({**entry, "action": "unchanged"})
            continue
        resp = drive.upload(f, parent)
        if resp.get("md5Checksum") != digest:
            raise RuntimeError(f"MD5 mismatch after uploading {rel}: local {digest}, Drive {resp.get('md5Checksum')}")
        out.append({**entry, "action": "uploaded"})
    return out


def kaggle_output(pattern: str, dest: Path, version: int | None = None, attempts: int = 6,
                  sleep=time.sleep) -> None:
    """Download part of the Kaggle output; waits and retries when Kaggle rate-limits (HTTP 429) or the
    connection drops."""
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    ref = f"{USER}/{SLUG}" + (f"/{version}" if version else "")
    for attempt in range(1, attempts + 1):
        r = subprocess.run([exe, "kernels", "output", ref, "-p", str(dest), "--file-pattern", pattern],
                           capture_output=True, text=True)
        if not r.returncode:
            return
        err = r.stderr.strip() or r.stdout.strip()
        transient = any(s in err.lower() for s in ("429", "rate limit", "too many requests", "timed out",
                                                   "connection", "ssl", "502", "503", "504"))
        if not transient or attempt == attempts:
            raise RuntimeError(f"kaggle kernels output failed: {err}")
        wait = 60 * attempt
        print(f"  Kaggle download: {err.splitlines()[0][:90]} -- retry {attempt}/{attempts - 1} in {wait}s",
              flush=True)
        sleep(wait)


def main() -> int:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--dest", default="btp_workspace/models_kaggle")
    ap.add_argument("--include-checkpoints", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--version", type=int, default=None,
                    help="Kaggle notebook version to copy from (default: the most advanced recent version -- "
                         "the newest one may be a skipped run holding an older workspace)")
    args = ap.parse_args()

    version = args.version or best_version()
    print(f"copying from Kaggle version {version or 'latest'}")
    drive = Drive(build("drive", "v3", credentials=Credentials.from_authorized_user_file(args.token),
                        cache_discovery=False))
    dest_root = drive.path(args.dest.split("/"))
    groups = dict(GROUPS, **(CHECKPOINTS if args.include_checkpoints else {}))
    manifest = {"synced": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kaggle_version": version,
                "files": []}
    for name, pattern in groups.items():
        tmp = Path(tempfile.mkdtemp(prefix=f"btp_sync_{name}_"))
        try:
            kaggle_output(pattern, tmp, version)
            ws = tmp / "btp_workspace"
            if not ws.is_dir():
                print(f"{name}: nothing in the Kaggle output yet")
                continue
            res = sync_dir(drive, ws, dest_root, dry_run=args.dry_run)
            manifest["files"] += res
            counts = {a: sum(r["action"] == a for r in res) for a in ("uploaded", "unchanged", "would copy")}
            print(f"{name}: {len(res)} files, {sum(r['size'] for r in res) / 1e9:.2f} GB -> {counts}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)          # never keep model files on local disk
    if not args.dry_run:
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "sync_manifest.json"
            m.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            drive.upload(m, dest_root)
    n_up = sum(f["action"] == "uploaded" for f in manifest["files"])
    print(f"done: {n_up} uploaded, {len(manifest['files']) - n_up} unchanged/listed -> My Drive/{args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
