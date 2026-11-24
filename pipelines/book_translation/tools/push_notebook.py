#!/usr/bin/env python
"""Upload notebooks/colab_pipeline.ipynb into the Colab notebook file on Google Drive.

    python tools/push_notebook.py --token ../gdrive_token.json [--file-id ID] [--env .env]

Replaces the content of the existing Drive file (same id / link / name), so the Colab
URL never changes. With --env, also uploads the .env (W&B key) to
My Drive/btp_workspace/.env, where the notebook looks for it.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FILE_ID = "1rcR2SDLa6zWifQ7Cfo70cj2ZAUkzDdDK"


def drive(token: str):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=Credentials.from_authorized_user_file(token), cache_discovery=False)


def ensure_folder(d, name: str, parent: str = "root") -> str:
    q = f"name='{name}' and '{parent}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    r = d.files().list(q=q, fields="files(id)").execute()["files"]
    if r:
        return r[0]["id"]
    return d.files().create(body={"name": name, "parents": [parent], "mimeType": "application/vnd.google-apps.folder"},
                            fields="id").execute()["id"]


def main() -> int:
    from googleapiclient.http import MediaIoBaseUpload
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--file-id", default=DEFAULT_FILE_ID)
    ap.add_argument("--notebook", default=str(REPO / "notebooks" / "colab_pipeline.ipynb"))
    ap.add_argument("--env", default=None, help="also upload this .env to My Drive/btp_workspace/.env")
    args = ap.parse_args()
    d = drive(args.token)
    data = Path(args.notebook).read_bytes()
    meta = d.files().update(fileId=args.file_id,
                            media_body=MediaIoBaseUpload(io.BytesIO(data), mimetype="application/x-ipynb+json"),
                            fields="id,name,modifiedTime,size").execute()
    print("notebook updated:", meta)
    if args.env:
        ws = ensure_folder(d, "btp_workspace")
        body = Path(args.env).read_bytes()
        existing = d.files().list(q=f"name='.env' and '{ws}' in parents and trashed=false",
                                  fields="files(id)").execute()["files"]
        media = MediaIoBaseUpload(io.BytesIO(body), mimetype="text/plain")
        if existing:
            d.files().update(fileId=existing[0]["id"], media_body=media).execute()
        else:
            d.files().create(body={"name": ".env", "parents": [ws]}, media_body=media, fields="id").execute()
        print("uploaded .env to My Drive/btp_workspace/.env")
    print(f"open: https://colab.research.google.com/drive/{args.file_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
