#!/usr/bin/env python
"""Read Colab run logs / reports from My Drive/btp_workspace without opening Colab.

    python tools/pull_logs.py --token ../gdrive_token.json [--ws btp_workspace/smoke] [--tail 60] [--reports]

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -c "cd ~/audiobookmanager/book-translation-pipeline && ~/.venvs/btp/bin/python tools/pull_logs.py --token ~/audiobookmanager/gdrive_token.json --ws btp_workspace/smoke --logs 1 --tail 25 2>&1 | grep -v file_cache"
    cd ~/audiobookmanager/book-translation-pipeline && ~/.venvs/btp/bin/python tools/pull_logs.py --token ~/audiobookmanager/gdrive_token.json --ws btp_workspace/smoke --logs 1 --tail 25 2>&1 | grep -v file_cache

    cd ~/audiobookmanager/book-translation-pipeline
    ~/.venvs/btp/bin/python tools/pull_logs.py --token ~/audiobookmanager/gdrive_token.json --ws btp_workspace/smoke --logs 2 --tail 40 --reports

"""
from __future__ import annotations

import argparse
import io
import sys


def main() -> int:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--ws", default="btp_workspace", help="workspace path under My Drive, e.g. btp_workspace/smoke")
    ap.add_argument("--tail", type=int, default=60)
    ap.add_argument("--logs", type=int, default=1, help="how many most recent logs to show")
    ap.add_argument("--reports", action="store_true")
    args = ap.parse_args()
    d = build("drive", "v3", credentials=Credentials.from_authorized_user_file(args.token), cache_discovery=False)

    def child(parent: str, name: str) -> str | None:
        r = d.files().list(q=f"name='{name}' and '{parent}' in parents and trashed=false", fields="files(id)").execute()
        return r["files"][0]["id"] if r["files"] else None

    def read(fid: str) -> str:
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, d.files().get_media(fileId=fid))
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue().decode("utf-8", "replace")

    cur = "root"
    for part in args.ws.split("/"):
        cur = child(cur, part)
        if cur is None:
            print(f"{args.ws} not found on Drive (has the notebook run yet?)")
            return 1
    for sub in ["logs"] + (["reports"] if args.reports else []):
        sid = child(cur, sub)
        if not sid:
            print(f"[{sub}] none yet")
            continue
        files = d.files().list(q=f"'{sid}' in parents and trashed=false", orderBy="modifiedTime desc",
                               fields="files(id,name,modifiedTime,size)", pageSize=50).execute()["files"]
        if sub == "logs":
            for f in files[: args.logs][::-1]:
                print(f"===== {f['name']} ({f['modifiedTime']})")
                print("\n".join(read(f["id"]).splitlines()[-args.tail:]))
        else:
            for f in files:
                print(f"===== {f['name']} ({f['modifiedTime']})")
                print(read(f["id"])[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
