# PharmaPlus — Restructure Log

**Date:** 2026-04-10
**Purpose:** Separate demo (CSV-driven) and production (DB-driven) concerns into distinct folders for clean GitHub collaboration and future CI/CD.

---

## What Changed

### Before
```
pharma/
├── app.py          # demo UI — ran engine on sim CSVs
├── main.py         # production pipeline — read MySQL, wrote CSVs
├── streamlit_app.py# older draft (abandoned, hardcoded absolute paths)
├── src/            # shared engine code
└── data/           # all CSVs mixed together
```

### After
```
pharma/
├── demo/
│   ├── app.py      # demo UI (updated imports + explicit data paths)
│   └── data/       # sim data only — committed to git
│       ├── expiry_stock_sim.csv
│       ├── analysis_targets_20.csv
│       ├── branch_market_dna.csv
│       └── competitor_prices.csv
│
├── pipeline/
│   ├── main.py     # production pipeline (updated imports + explicit data paths)
│   └── data/       # runtime outputs — gitignored, .gitkeep holds the folder
│       └── .gitkeep
│
├── src/            # shared core — both demo and pipeline depend on this
│   ├── engine.py
│   ├── loader.py
│   ├── seasonal.py
│   ├── database.py
│   ├── competitor_pricing.py
│   ├── serp_pricing.py
│   ├── goodlife_scraper.py
│   └── simulate_data.py
│
├── mapping/        # geo enrichment scripts (produces branch_market_dna.csv)
├── pharmapluslogo.ico
├── pharmapluslogo.png
├── requirements.txt
├── .gitignore
└── RESTRUCTURE.md
```

---

## Code Changes

### `demo/app.py`
- Added `from pathlib import Path` and `sys.path.insert(0, str(ROOT))` at top
- `ROOT = Path(__file__).parent.parent` — resolves to project root
- `DATA = Path(__file__).parent / "data"` — resolves to `demo/data/`
- All CSV path constants (`ENGINE_CSV`, `BUNDLE_CSV`, `GEO_CSV`, `SERP_CSV`) now use `DATA /`
- `page_icon` and logo `_img_b64()` call now use `ROOT /` (logos stay at project root)
- `load_demo_inputs()` call now passes explicit `expiry_path`, `geo_path`, `serp_path` from `DATA`
- `os.makedirs("data")` replaced with `DATA.mkdir(parents=True, exist_ok=True)`

### `pipeline/main.py`
- Added `from pathlib import Path` and `sys.path.insert(0, str(ROOT))` at top
- `ROOT = Path(__file__).parent.parent` — resolves to project root
- `DATA = Path(__file__).parent / "data"` — resolves to `pipeline/data/`
- All `write_csv("data/...")` calls now use `str(DATA / "...")`

---

## How to Run

**Demo (no database needed):**
```bash
# From project root
streamlit run demo/app.py
```

**Production pipeline:**
```bash
# Requires .env with DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
python pipeline/main.py
```

**Generate fresh sim data:**
```bash
python src/simulate_data.py
# Copies output to demo/data/ manually or update simulate_data.py output path
```

**Generate geo DNA (run once):**
```bash
cd mapping && python verify.py
# Output: data/branch_market_dna.csv — copy to demo/data/
```

---

## How to Merge Back (if needed)

If the separation needs to be undone and the project returned to a flat structure:

1. Copy `demo/app.py` → `app.py` (root)
2. Copy `pipeline/main.py` → `main.py` (root)
3. In both files: remove the `ROOT/DATA/sys.path` block at the top
4. Restore path constants to `"data/..."` strings
5. Restore `load_demo_inputs()` call to `load_demo_inputs(ref_date=date.today())`
6. Delete `demo/` and `pipeline/` folders
7. Move `demo/data/*` back to `data/`

---

## What Was NOT Changed

- `src/` — untouched. Both entry points share it via `sys.path`.
- `mapping/` — untouched. Run independently to regenerate geo data.
- `streamlit_app.py` — not migrated (abandoned draft, has hardcoded absolute paths). Do not commit.
- `data/` at root — kept as-is for now. Legacy files remain. Clean up separately if needed.

---

## GitHub Notes

- `.gitignore` added: excludes `.env`, `venv/`, `__pycache__/`, pipeline runtime CSVs, SERP cache
- `pipeline/data/.gitkeep` — empty file to preserve the folder in git
- Demo data (`demo/data/*.csv`) **is committed** — it's simulated, contains no real patient/product data
- Never commit `.env` — use `.env.example` to document required keys
