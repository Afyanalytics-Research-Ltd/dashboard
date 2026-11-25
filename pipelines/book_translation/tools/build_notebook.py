#!/usr/bin/env python
"""Build the Colab and Kaggle notebooks from the repository sources.

    notebooks/colab_pipeline.ipynb    interactive (Google Drive workspace, form cells)
    notebooks/kaggle_pipeline.ipynb   headless "Save & Run All" (JOB variable, Bible.is audio,
                                      workspace persisted as notebook output)

Both are self-contained: they write every source file of the repo with %%writefile cells,
so neither copy can drift from git. CI fails if a committed notebook is out of date.

    python tools/build_notebook.py            # write both notebooks
    python tools/build_notebook.py --check    # exit 1 if either notebook is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "colab_pipeline.ipynb"
OUT_KAGGLE = REPO / "notebooks" / "kaggle_pipeline.ipynb"
KAGGLE_ROOT = "/tmp/btp"
KAGGLE_SLUG = "adventproclaimer-multilingual-translator"
SOURCE_GLOBS = ["btp/*.py", "config/*.yaml", "third_party/__init__.py", "third_party/finetune_hf_vits/*.py",
                "third_party/finetune_hf_vits/LICENSE", "third_party/finetune_hf_vits/README.md",
                "requirements-colab.txt"]
COLAB_ROOT = "/content/btp"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, collapsed: bool = False) -> dict:
    meta = {"cellView": "form"} if collapsed else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def source_files() -> list[Path]:
    files: list[Path] = []
    for g in SOURCE_GLOBS:
        files += sorted(REPO.glob(g))
    return [f for f in files if f.is_file()]


def stage(cmd: str) -> str:
    return f"!cd {COLAB_ROOT} && python -m btp.cli {cmd}"


def source_cells(root: str) -> list[dict]:
    """mkdir + one %%writefile cell per repo source file + an import check."""
    cells = []
    dirs = sorted({str(Path(root) / f.relative_to(REPO).parent) for f in source_files()})
    cells.append(code("!mkdir -p " + " ".join(dirs)))
    for f in source_files():
        rel = f.relative_to(REPO).as_posix()
        body = f.read_text(encoding="utf-8")
        if not body.endswith("\n"):
            body += "\n"
        cells.append(code(f"%%writefile {root}/{rel}\n{body}"))
    cells.append(code(f"!touch {root}/btp/__init__.py && cd {root} && python -c \"import btp.cli; "
                      f"print('btp imports OK')\""))
    return cells


def build() -> dict:
    cells = [md("""
# Kenyan languages: Bible-seeded ASR, TTS and translation (book -> text -> audio)

This notebook is **generated** from the git repo `audiobookmanager/book-translation-pipeline`
(`tools/build_notebook.py`). Edit code in the repo, not here, then rebuild + push.

**How to run**
1. *Runtime -> Change runtime type -> T4 GPU.*
2. *Runtime -> Run all.* Section 1 mounts Drive (approve the pop-up) and installs packages.
3. Section 3 is a **smoke test** of every stage on one language with tiny limits (~30-45 min).
4. Section 4 is the **full run**. Every stage is resumable: if Colab disconnects, run it again
   and it continues where it stopped.

**Where things go**
* Raw audio (read-only): `My Drive/adventproclaimer_training_data/`
* Everything produced: `My Drive/btp_workspace/` (corpora, aligned clips, models, `logs/`, `reports/`)
* Metrics, audio samples and example translations: Weights & Biases project `kenyan-speech-mt`
"""),
             md("## 1. Setup"),
             code("""
#@title Mount Google Drive, check the GPU
from google.colab import drive
drive.mount('/content/drive')
import subprocess, torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE -- switch the runtime to T4 GPU')
""", collapsed=True),
             code(f"""
#@title Install packages (about 1-2 minutes)
!pip install -q uroman sacrebleu jiwer wandb safetensors python-docx pypdf
!mkdir -p {COLAB_ROOT}
""", collapsed=True),
             code("""
#@title Weights & Biases key (Colab secret WANDB_API_KEY, else My Drive/btp_workspace/.env)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   # less fragmentation on the 15 GB T4
key = None
try:
    from google.colab import userdata
    key = userdata.get('WANDB_API_KEY')
except Exception:
    pass
env_file = '/content/drive/MyDrive/btp_workspace/.env'
if not key and os.path.exists(env_file):
    for line in open(env_file):
        if line.startswith('WANDB_API_KEY='):
            key = line.split('=', 1)[1].strip()
if key:
    os.environ['WANDB_API_KEY'] = key
    os.environ.setdefault('WANDB_PROJECT', 'kenyan-speech-mt')
    print('W&B key loaded -> runs log online')
else:
    print('No W&B key found -> runs log offline to btp_workspace/wandb (sync later)')
""", collapsed=True),
             md("## 2. Source code\nEvery file of the repo, written to `/content/btp`. (Generated -- do not edit here.)")]
    cells += source_cells(COLAB_ROOT)

    cells += [md("""
## 3. Smoke test (one language, tiny limits)
Runs every stage end-to-end on `SMOKE_LANG` with a few chapters and a few training steps, so
problems surface in minutes instead of after hours. Logs: `My Drive/btp_workspace/logs/`.
"""),
              code("""
SMOKE_LANG = 'saq'  #@param ['saq', 'luo', 'swh', 'kik', 'som', 'orm', 'rel', 'dig', 'sxb', 'teo']
SMOKE_SET = {'saq': 'SAQBTLN1DA', 'luo': 'LUOBIBN1DA', 'swh': 'SWABIBN1DA', 'kik': 'KIKKGVN1DA', 'som': 'SOMTVIN2DA',
             'orm': 'ORCBTLN1DA', 'rel': 'RELBTLN1DA', 'dig': 'DIGTBLN1DA', 'sxb': 'SXBBTLN1DA',
             'teo': 'TEOBSUN2DA'}[SMOKE_LANG]
SMOKE_WS = '/content/drive/MyDrive/btp_workspace/smoke'   # keeps smoke outputs apart from real runs
import os; os.environ['BTP_WORKSPACE'] = SMOKE_WS
"""),
              code(stage("check --langs {SMOKE_LANG}")),
              code(stage("corpus --langs {SMOKE_LANG}")),
              code(stage("import-openbible --langs {SMOKE_LANG} --max-chapters 1")),
              code(stage("align --langs {SMOKE_LANG} --sets {SMOKE_SET} --books MRK COL PHP --max-chapters-per-book 2")),
              code(stage("status --langs {SMOKE_LANG}")),
              code(stage("asr --langs {SMOKE_LANG} --max-steps 30 --set asr.eval_steps=15 --set asr.save_steps=15 "
                         "--set asr.eval_samples=20 --set align.min_clip_sec=1.0")),
              code(stage("mt --langs {SMOKE_LANG} --max-steps 30 --set mt.eval_steps=15 --set mt.save_steps=30 "
                         "--set mt.eval_samples_per_lang=20 --set mt.warmup_steps=5")),
              code(stage("tts --langs {SMOKE_LANG} --max-steps 30 --set tts.eval_steps=15 --set tts.save_steps=30 "
                         "--set tts.per_device_batch=8")),
              code(stage("evaluate --langs {SMOKE_LANG}")),
              code("""
#@title Book round trip on a short original paragraph (translate -> synthesize)
import os
os.makedirs(SMOKE_WS + '/books', exist_ok=True)
PARAGRAPHS = [
    'The rain came early this year. Farmers planted maize and beans before the first storm.',
    "In the evening, the children gathered around the fire and listened to their grandmother's stories.",
]
open(SMOKE_WS + '/books/sample.txt', 'w').write((chr(10) * 2).join(PARAGRAPHS))   # 2 paragraphs, 3 sentences
""", collapsed=True),
              code(stage("translate-book --langs {SMOKE_LANG} --book {SMOKE_WS}/books/sample.txt")),
              code(stage("synthesize-book --langs {SMOKE_LANG} --review {SMOKE_WS}/books/{SMOKE_LANG}/sample.review.tsv")),
              code("""
#@title Listen to the synthesized sample
from IPython.display import Audio, display
import glob
for f in sorted(glob.glob(SMOKE_WS + f'/books/{SMOKE_LANG}/sample/*.wav')):
    print(f); display(Audio(f))
""", collapsed=True),
              code("""
#@title Smoke-test summary
import json, glob, os
for f in sorted(glob.glob(SMOKE_WS + '/reports/stage_*.json')):
    d = json.load(open(f))
    print(os.path.basename(f), '->', 'ERROR: ' + d['error'] if 'error' in d else 'ok')
""", collapsed=True),
              md("""
## 4. Full run
Run the cells in order. Each is resumable and can be re-run after a disconnect. Pick
languages with `LANGS` ('a' = the 17 target languages, 'b:text' = new languages that have
Bible text, or a comma list like 'swh,kik').
"""),
              code("""
LANGS = 'a'  #@param {type:"string"}
import os; os.environ.pop('BTP_WORKSPACE', None)   # back to the real workspace
"""),
              code(stage("check --langs {LANGS}")),
              code(stage("corpus --langs {LANGS}")),
              code(stage("import-openbible --langs {LANGS}")),
              code(stage("align --langs {LANGS}")),
              code(stage("pseudo --langs {LANGS}")),
              code(stage("status --langs {LANGS}")),
              code(stage("mt --langs {LANGS}")),
              code(stage("asr --langs {LANGS}")),
              code(stage("tts --langs {LANGS}")),
              code(stage("evaluate --langs {LANGS}")),
              md("""
## 4b. Pipeline B -- languages beyond the 17
* **With Bible text** (Daasanach `dsh`, Ilwana `mlk`, Meru `mer`): run section 4 with `LANGS = 'b:text'`.
  Their ASR/TTS start from a related language's MMS model and MT gets a new language token.
* **Audio only** (Ekegusii, Maasai, Turkana, the Luhya varieties, ...): no matching text exists.
  1. `pseudo` writes rough drafts with the related language's ASR.
  2. `queue` exports clips + drafts to `btp_workspace/transcription/<lang>/batch_XX/`.
  3. A fluent speaker fills in the `corrected` column of `queue.tsv` (put `x` in `skip` for bad clips).
  4. `import-transcripts` turns corrections into training data; then run `asr` and `tts` for the language.
  Two to three hours of corrected speech is enough for a usable first ASR model.
"""),
              code("""
B_LANG = 'guz'  #@param {type:"string"}
B_CLIPS = 300  #@param {type:"integer"}
"""),
              code(stage("corpus --langs {B_LANG}")),
              code(stage("pseudo --langs {B_LANG} --max-chapters 60")),
              code(stage("queue --langs {B_LANG} --n {B_CLIPS}")),
              code(stage("import-transcripts --langs {B_LANG}")),
              md("""
## 5. Translate a book into audio
1. Upload an English book (.txt, .docx, .pdf or .epub) to `My Drive/btp_workspace/books/`.
2. Run *translate-book*: it writes `books/<lang>/<book>.review.tsv`.
3. A fluent speaker fills in the `reviewed_translation` column (Google Sheets can open the TSV).
4. Run *synthesize-book*: narrated WAV files + an intelligibility report appear in `books/<lang>/<book>/`.
"""),
              code("""
BOOK = '/content/drive/MyDrive/btp_workspace/books/my_book.txt'  #@param {type:"string"}
BOOK_LANG = 'swh'  #@param {type:"string"}
import os; BOOK_STEM = os.path.splitext(os.path.basename(BOOK))[0]
"""),
              code(stage("translate-book --langs {BOOK_LANG} --book '{BOOK}'")),
              code(stage("synthesize-book --langs {BOOK_LANG} --review /content/drive/MyDrive/btp_workspace/books/{BOOK_LANG}/{BOOK_STEM}.review.tsv")),
              md("## 6. Monitoring"),
              code(f"""
#@title Latest log lines + workspace status
!ls -t /content/drive/MyDrive/btp_workspace/logs | head -5
!tail -n 30 "$(ls -t /content/drive/MyDrive/btp_workspace/logs/*.log | head -1)"
!cd {COLAB_ROOT} && python -m btp.cli status --langs a
""", collapsed=True),
              code("""
#@title Upload offline W&B runs (only needed if a run logged offline)
!wandb sync --sync-all /content/drive/MyDrive/btp_workspace/wandb/wandb 2>/dev/null | tail -5
""", collapsed=True)]

    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def build_kaggle() -> dict:
    """Headless Kaggle variant: every cell runs on "Save & Run All"; JOB picks what runs."""
    r = KAGGLE_ROOT
    cells = [md("""
# adventproclaimer_multilingual_translator

Kaggle copy of the Colab pipeline (generated from the git repo `book-translation-pipeline` by
`tools/build_notebook.py`; edit the repo, not this notebook). It runs headless: **Save & Run All**
(or `tools/kaggle_run.py` from the repo) executes the `JOB` chosen below.

* Audio is streamed from Bible.is (Kaggle cannot mount Google Drive), nothing is stored but clips.
* The workspace lives in `/kaggle/working/btp_workspace` and is saved as this notebook's output;
  the next run attaches that output and continues where the previous one stopped.
* Every stage stops cleanly ~1 h before Kaggle's 12 h limit (a run that times out saves nothing).
* Settings -> Accelerator: **GPU**, Internet: **on**. W&B key: Kaggle secret `WANDB_API_KEY`
  (Add-ons -> Secrets) or the private dataset `btp-secrets`.
"""),
             code("""
# ---- run parameters (tools/kaggle_run.py rewrites these lines) ----
JOB = 'smoke'        # smoke | full | b | book
LANGS = 'a'          # 'a' = the 17 target languages, 'b:text', or 'swh,kik'
SMOKE_LANG = 'saq'
ALIGN_CAP_HOURS = 8  # aligned hours per language (keeps the saved workspace under Kaggle's 20 GB)
DEADLINE_HOURS = 11.0  # stages stop cleanly before this; Kaggle kills runs at 12 h and then saves nothing
WEEKLY_GPU_HOURS = 30.0  # Kaggle's weekly GPU quota: runs shrink their deadline / skip to stay inside it
QUOTA_MARGIN_HOURS = 1.5  # head-room for GPU use this notebook cannot see (other notebooks, smoke tests)
STAGE_HOURS = {'mt': 5.0, 'asr': 2.5}  # per-run time slice, so translation, ASR and voices all advance each run
OUTPUT_LIMIT_GB = 19.3  # Kaggle saves NOTHING if the output exceeds 20 GB: trim to stay below this
B_LANG = 'guz'
BOOK = ''            # path of an English book under /kaggle/input for JOB='book'
BOOK_LANG = 'swh'
"""),
             code("""
import glob, json, os, shutil, subprocess, sys, time
START = time.time()
WS = '/kaggle/working/btp_workspace'
os.environ.update({
    'BTP_WORKSPACE': WS, 'BTP_AUDIO_SOURCE': 'bibleis', 'BTP_RAW_AUDIO': '/tmp/no_drive',
    'BTP_LOCAL_CACHE': '/tmp/btp_cache', 'BTP_DEADLINE': str(START + float(DEADLINE_HOURS) * 3600),
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'WANDB_PROJECT': 'kenyan-speech-mt',
    'BTP_KEEP_PREV': '0',      # no <checkpoint>.prev copies: tmp+rename already prevents torn files (20 GB limit)
})
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE -- enable a GPU accelerator')

# W&B key: Kaggle secret first, then the private dataset btp-secrets
key = None
try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret('WANDB_API_KEY')
except Exception:
    pass
for f in glob.glob('/kaggle/input/**/.env', recursive=True) + glob.glob('/kaggle/input/**/btp.env', recursive=True):
    for line in open(f):
        if not key and line.startswith('WANDB_API_KEY='):
            key = line.split('=', 1)[1].strip()
if key:
    os.environ['WANDB_API_KEY'] = key
print('W&B:', 'online' if key else 'offline (no key found)')

# ---- Self-sufficient start (safe for runs started by Kaggle's own scheduler) ----
SKIP = None
if not torch.cuda.is_available():
    SKIP = 'no GPU assigned (weekly quota used up?) -- nothing run, no GPU hours used'


def ws_progress(ws):
    # (#finished trainings, #done units, sum of checkpoint steps) of a saved workspace
    trained = done = steps = 0
    for f in glob.glob(ws + '/status/*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):          # e.g. kaggle_ledger.json is a list, not a status record
            continue
        if os.path.basename(f).startswith('train_'):
            trained += bool(d.get('finished'))
        elif isinstance(d.get('done'), list):
            done += len(d['done'])
    for f in glob.glob(ws + '/models/*/*/checkpoint.pt') + glob.glob(ws + '/models/mt/checkpoint/trainer_state.pt'):
        try:
            steps += int(torch.load(f, map_location='cpu', weights_only=False)['step'])
        except Exception:
            pass
    return (trained, done, steps)


# Candidate workspaces: the attached previous output, plus (if a Kaggle token secret is available) a fresh
# download of this notebook's latest output -- so a scheduled run never restarts from stale progress.
cands = [p for p in glob.glob('/kaggle/input/*/btp_workspace') + glob.glob('/kaggle/input/*/*/btp_workspace')
         if os.path.isdir(p)]
KREF = 'lutherlunyamwi/adventproclaimer-multilingual-translator'


def summary_score(s):
    # same ordering as ws_progress, from a run's run_summary.json (no need to download its workspace)
    p = (s or {}).get('progress') or {}
    steps = sum(v for v in (p.get('checkpoint_steps') or {}).values() if isinstance(v, int))
    return (len(p.get('trained') or []), sum((p.get('status_done') or {}).values()), steps)


def pick_best(summaries):
    scored = [(summary_score(s), v) for v, s in summaries.items() if s]
    return max(scored)[1] if scored else None


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def current_version(api):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as k:
        r = ApiGetKernelRequest()
        r.user_name, r.kernel_slug = KREF.split('/')
        return int(k.kernels.kernels_api_client.get_kernel(r).metadata.current_version_number)


tok = API = None
BEST_VERSION = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
except Exception:
    pass
if tok:
    os.environ['KAGGLE_API_TOKEN'] = tok
    subprocess.run('pip install -q kaggle six >/dev/null 2>&1', shell=True)
    try:
        API = kaggle_api()
    except (Exception, SystemExit) as e:
        print('Kaggle API unavailable:', e)
else:
    print('WARNING: no KAGGLE_API_TOKEN secret -- restoring only the attached input (may be an older version)')
if API is not None and not SKIP:
    # The newest version is not necessarily the most advanced: Kaggle's daily schedule started version 13
    # while version 12 was training; 13 got no GPU, skipped and saved the OLD workspace. So compare the
    # recent versions' summaries and restore the best one.
    try:
        cur = current_version(API)
    except Exception as e:
        cur = None
        print('current version unknown:', e)
    sums = {}
    for v in (range(cur - 1, max(cur - 9, 0), -1) if cur else []):
        d = f'/tmp/summary_v{v}'
        subprocess.run(f"kaggle kernels output {KREF}/{v} -p {d} --file-pattern 'run_summary.json' -q >/dev/null 2>&1",
                       shell=True)
        try:
            sums[v] = json.load(open(d + '/run_summary.json'))
        except Exception:
            sums[v] = None
        shutil.rmtree(d, ignore_errors=True)
    BEST_VERSION = pick_best(sums)
    print('recent versions (trained, done, steps):', {v: summary_score(s) for v, s in sums.items() if s},
          '-> restoring version', BEST_VERSION)
    ref = f'{KREF}/{BEST_VERSION}' if BEST_VERSION else KREF
    rc = subprocess.run(f"kaggle kernels output {ref} -p /tmp/latest_output --file-pattern 'btp_workspace/.*' -q",
                        shell=True).returncode
    if rc == 0 and os.path.isdir('/tmp/latest_output/btp_workspace'):
        cands.append('/tmp/latest_output/btp_workspace')
RESTORED = None
if cands and not os.path.exists(WS):
    best = max(cands, key=ws_progress)
    print('workspace candidates:', {c: ws_progress(c) for c in cands}, '-> restoring', best)
    shutil.copytree(best, WS)
    RESTORED = best
os.makedirs(WS + '/status', exist_ok=True)
shutil.rmtree('/tmp/latest_output', ignore_errors=True)

# Weekly GPU budget, from a ledger kept inside the workspace (rolling 7 days, conservative)
LEDGER = WS + '/status/kaggle_ledger.json'
ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
used_7d = sum(e['hours'] for e in ledger if e['start'] >= START - 7 * 86400)
left = WEEKLY_GPU_HOURS - QUOTA_MARGIN_HOURS - used_7d
QUOTA = None
if API is not None:
    # Kaggle's own count (every notebook and editor session, real weekly reset) beats this notebook's ledger
    try:
        g = API.quota_view().gpu_quota
        QUOTA = {'used_h': round(g.time_used.total_seconds() / 3600, 2),
                 'allowed_h': round(g.total_time_allowed.total_seconds() / 3600, 2)}
        used_7d = QUOTA['used_h']
        left = QUOTA['allowed_h'] - 0.5 - used_7d
    except Exception as e:
        print('GPU quota unavailable -- using the ledger:', e)
if not SKIP:
    DEADLINE_HOURS = min(float(DEADLINE_HOURS), left - 0.4)   # 0.4 h: start-up + summary
    if DEADLINE_HOURS < 1.0:
        SKIP = f'weekly GPU budget: {used_7d:.1f} h used in the last 7 days, {left:.1f} h left'
    os.environ['BTP_DEADLINE'] = str(START + max(DEADLINE_HOURS, 0) * 3600)
print(f'GPU budget: {used_7d:.1f} h used (7 days) | deadline this run: {DEADLINE_HOURS:.2f} h | skip: {SKIP}')
"""),
             code(f"!pip install -q uroman sacrebleu jiwer wandb safetensors python-docx pypdf\n!mkdir -p {r}"),
             md("## Source code (generated)")]
    cells += source_cells(r)
    cells += [md("## Run the job"),
              code(f"""
FAILED = []
STAGE_SECONDS = {{}}
def btp(args, env=None):
    \"\"\"Run one pipeline stage, streaming its output; failures are recorded, not fatal.\"\"\"
    print(f"\\n===== btp {{args}}  ({{(time.time() - START) / 3600:.2f}} h elapsed)", flush=True)
    t0 = time.time()
    e = dict(os.environ, **(env or {{}}))
    p = subprocess.Popen(f"cd {r} && python -m btp.cli {{args}}", shell=True, env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    rc = p.wait()
    name = args.split()[0]
    STAGE_SECONDS[name] = round(STAGE_SECONDS.get(name, 0) + time.time() - t0, 1)
    if rc:
        FAILED.append((name, rc))
    return rc

def time_left_h():
    return (float(os.environ['BTP_DEADLINE']) - time.time()) / 3600
"""),
              code("""
if JOB == 'smoke' and not SKIP:
    SMOKE = {'BTP_WORKSPACE': '/tmp/btp_smoke'}           # not saved: smoke outputs are throwaway
    sets = {'saq': 'SAQBTLN1DA', 'luo': 'LUOBIBN1DA', 'swh': 'SWABIBN1DA', 'kik': 'KIKKGVN1DA', 'dig': 'DIGTBLN1DA'}
    btp(f"check --langs {SMOKE_LANG}", SMOKE)
    btp(f"corpus --langs {SMOKE_LANG}", SMOKE)
    btp(f"import-openbible --langs {SMOKE_LANG} --max-chapters 1", SMOKE)   # 1 file (~0.8 GB) if available
    btp(f"align --langs {SMOKE_LANG} --sets {sets[SMOKE_LANG]} --books MRK COL PHP --max-chapters-per-book 2", SMOKE)
    btp(f"asr --langs {SMOKE_LANG} --max-steps 30 --set asr.eval_steps=15 --set asr.save_steps=15 --set asr.eval_samples=20", SMOKE)
    btp(f"mt --langs {SMOKE_LANG} --max-steps 30 --set mt.eval_steps=15 --set mt.save_steps=30 --set mt.eval_samples_per_lang=20 --set mt.warmup_steps=5", SMOKE)
    btp(f"tts --langs {SMOKE_LANG} --max-steps 30 --set tts.eval_steps=15 --set tts.save_steps=30 --set tts.per_device_batch=8", SMOKE)
    btp(f"evaluate --langs {SMOKE_LANG}", SMOKE)
    os.makedirs('/tmp/btp_smoke/books', exist_ok=True)
    PARAGRAPHS = ['The rain came early this year. Farmers planted maize and beans before the first storm.',
                  "In the evening, the children gathered around the fire and listened to their grandmother's stories."]
    open('/tmp/btp_smoke/books/sample.txt', 'w').write((chr(10) * 2).join(PARAGRAPHS))
    btp(f"translate-book --langs {SMOKE_LANG} --book /tmp/btp_smoke/books/sample.txt", SMOKE)
    btp(f"synthesize-book --langs {SMOKE_LANG} --review /tmp/btp_smoke/books/{SMOKE_LANG}/sample.review.tsv", SMOKE)
    for sub in ('reports', 'logs'):                       # keep the evidence, not the models
        shutil.copytree(f'/tmp/btp_smoke/{sub}', f'/kaggle/working/smoke/{sub}', dirs_exist_ok=True)
    for f in glob.glob(f'/tmp/btp_smoke/books/{SMOKE_LANG}/sample/*.wav'):
        shutil.copy(f, '/kaggle/working/smoke/')
"""),
              code("""
TRAINED_BEFORE = {os.path.basename(f)[len('train_'):-5] for f in glob.glob(WS + '/status/train_*.json')
                  if json.load(open(f)).get('finished')}
if SKIP:
    print('SKIPPED:', SKIP)
elif JOB == 'full':
    # storage pruning keeps only final/ of finished models (20 GB output limit)
    cap = (f"--set align.max_hours_per_lang={ALIGN_CAP_HOURS} --set tts.prune_checkpoint_when_done=true "
           f"--set mt.prune_checkpoint_when_done=true")
    for stage in ('check', 'corpus', 'import-openbible', 'align', 'pseudo', 'mt', 'asr', 'tts', 'evaluate'):
        if time_left_h() < 0.5:
            print(f'deadline reached -- {stage} and later stages run in the next session'); break
        env = None
        if stage in STAGE_HOURS:      # this stage's own (earlier) deadline; it checkpoints and resumes next run
            env = {'BTP_DEADLINE': str(min(float(os.environ['BTP_DEADLINE']), time.time() + STAGE_HOURS[stage] * 3600))}
        btp(f"{stage} --langs {LANGS} {cap}", env)
    btp(f"status --langs {LANGS}")
# (stages whose training already finished return immediately: btp.utils.training_done)
elif JOB == 'b':
    for stage in ('corpus', 'pseudo --max-chapters 60', 'queue --n 300', 'import-transcripts'):
        btp(f"{stage} --langs {B_LANG}")
elif JOB == 'book' and BOOK:
    btp(f"translate-book --langs {BOOK_LANG} --book '{BOOK}'")
    stem = os.path.splitext(os.path.basename(BOOK))[0]
    btp(f"synthesize-book --langs {BOOK_LANG} --review {WS}/books/{BOOK_LANG}/{stem}.review.tsv")
"""),
              code("""
# Summary + keep the saved output under Kaggle's 20 GB limit
shutil.rmtree('/tmp/btp_cache', ignore_errors=True)
for f in sorted(glob.glob('/kaggle/working/**/reports/stage_*.json', recursive=True)):
    d = json.load(open(f))
    bad = 'error' in d or any(isinstance(v, dict) and 'error' in v for v in (d.get('result') or {}).values())
    print(('ERROR ' if bad else 'ok    ') + f)
def tree_gb(top):
    return sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(top) for fn in fns
               if not os.path.islink(os.path.join(dp, fn))) / 1e9

def drop(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)

# W&B files are already uploaded when W&B was online -- no need to keep them in the output
if os.environ.get('WANDB_API_KEY'):
    drop(WS + '/wandb')
used = tree_gb('/kaggle/working')
TRIMMED = []
if used > OUTPUT_LIMIT_GB:
    # Over the limit Kaggle saves nothing at all (all of this run's progress lost). Trim, least valuable first.
    for pattern in (WS + '/wandb', WS + '/**/*.prev', WS + '/**/*.tmp', WS + '/models/tts/*/checkpoint.pt'):
        for p in glob.glob(pattern, recursive=True):
            if used <= OUTPUT_LIMIT_GB:
                break
            gb = tree_gb(p) if os.path.isdir(p) else os.path.getsize(p) / 1e9
            drop(p)
            used -= gb
            TRIMMED.append(os.path.relpath(p, WS))
    print('OUTPUT OVER THE LIMIT -- trimmed:', TRIMMED)
BREAKDOWN = {}
for top in sorted(glob.glob(WS + '/*/*')):
    if os.path.isdir(top):
        g = tree_gb(top)
        if g >= 0.05:
            BREAKDOWN[os.path.relpath(top, WS)] = round(g, 2)
print('output by folder (GB):', json.dumps(dict(sorted(BREAKDOWN.items(), key=lambda kv: -kv[1]))))
print(f'saved output: {used:.1f} GB of 20 GB | failed stages: {FAILED or "none"} | elapsed {(time.time() - START) / 3600:.2f} h')
# Progress snapshot: lets the scheduler verify that a resumed run continued (and when training is complete)
sys.path.insert(0, '/tmp/btp')
progress = {'status_done': {}, 'checkpoint_steps': {}, 'trained': []}
for f in glob.glob(WS + '/status/*.json'):
    name = os.path.basename(f)[:-5]
    d = json.load(open(f))
    if not isinstance(d, dict):            # e.g. kaggle_ledger.json is a list, not a status record
        continue
    if name.startswith('train_'):
        if d.get('finished'):
            progress['trained'].append(name[len('train_'):])
    elif isinstance(d.get('done'), list):
        progress['status_done'][name] = len(d['done'])
for f in glob.glob(WS + '/models/*/*/checkpoint.pt') + glob.glob(WS + '/models/mt/checkpoint/trainer_state.pt'):
    try:
        progress['checkpoint_steps'][os.path.relpath(f, WS + '/models')] = int(torch.load(f, map_location='cpu', weights_only=False)['step'])
    except Exception as e:
        progress['checkpoint_steps'][os.path.relpath(f, WS + '/models')] = f'unreadable: {e}'
all_done = False
if JOB == 'full':
    from btp.languages import resolve_langs
    codes = resolve_langs(LANGS)
    t = set(progress['trained'])
    all_done = any(x.startswith('mt_') for x in t) and all(f'asr_{c}' in t and f'tts_{c}' in t for c in codes)
if JOB == 'full' and not SKIP:
    ledger.append({'start': START, 'hours': round((time.time() - START) / 3600 + 0.1, 3)})
    json.dump(ledger, open(LEDGER, 'w'))
json.dump({'finished': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'skip': SKIP,
           'gpu_hours_7d_before': round(used_7d, 2),
           'job': JOB, 'langs': LANGS, 'deadline_hours': DEADLINE_HOURS, 'restored_from': RESTORED,
           'restored_version': BEST_VERSION, 'gpu_quota': QUOTA,
           'failed': FAILED, 'stage_seconds': STAGE_SECONDS, 'progress': progress, 'all_done': all_done,
           'output_gb': round(used, 2), 'output_breakdown_gb': BREAKDOWN, 'trimmed': TRIMMED,
           'hours': round((time.time() - START) / 3600, 2)},
          open('/kaggle/working/run_summary.json', 'w'), indent=1)
print('progress:', json.dumps(progress), '| all training done:', all_done)
""")]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    targets = [(OUT, build()), (OUT_KAGGLE, build_kaggle())]
    if args.check:
        stale = [p.name for p, nb in targets if not p.is_file() or p.read_text(encoding="utf-8") != render(nb)]
        if stale:
            print(f"stale notebook(s): {', '.join(stale)} -- run: python tools/build_notebook.py")
            return 1
        print("notebooks up to date")
        return 0
    for path, nb in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(nb), encoding="utf-8")
        print(f"wrote {path} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
