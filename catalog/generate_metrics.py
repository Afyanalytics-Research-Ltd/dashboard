"""
generate_metrics.py
───────────────────
Auto-generates catalog/metrics.yaml from a live Cube.dev /meta endpoint.

For each cube it calls GPT-4o-mini to produce:
  - a human-readable metric name
  - a rich description (used by the classifier to match questions)
  - a sensible default cube_query (measures, dimensions, timeDimensions)

Usage:
    python generate_metrics.py \
        --cube-url http://localhost:4000 \
        --cube-token martin \
        --openai-key sk-... \
        --output catalog/metrics.yaml

Or set env vars CUBE_API_URL, CUBE_API_TOKEN, OPENAI_API_KEY and run:
    python generate_metrics.py
"""

import argparse
import json
import os
import sys
import textwrap
import time

import httpx
import yaml
from openai import OpenAI

# ── defaults (override via CLI args or env vars) ──────────────────────────────

DEFAULT_CUBE_URL   = os.getenv("CUBE_API_URL",   "http://localhost:4000").strip()
DEFAULT_CUBE_TOKEN = os.getenv("CUBE_API_TOKEN",  "martin").strip()
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY",  "").strip()
DEFAULT_OUTPUT     = os.path.join(os.path.dirname(__file__), "metrics.yaml")
DEFAULT_MODEL      = "gpt-4o-mini"

# Cubes to skip entirely (e.g. internal staging tables)
SKIP_CUBES: set[str] = set()

# ── fetch /meta ───────────────────────────────────────────────────────────────

def fetch_meta(cube_url: str, token: str) -> list[dict]:
    resp = httpx.get(
        f"{cube_url}/cubejs-api/v1/meta",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["cubes"]


# ── LLM metric generation ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a data analyst helping build an analytics agent for a hospital system.

You will be given a Cube.dev cube definition (name, measures, dimensions).
Your job is to produce ONE metric entry for a metrics.yaml catalog file.

Return ONLY valid JSON — no markdown, no explanation — with this structure:
{
  "id": "<snake_case id, unique, descriptive>",
  "name": "<Short human-readable title>",
  "description": "<2-4 sentence description. Describe what this metric measures, \
when to use it, example questions it answers, and any important filter hints. \
This text is used by an LLM classifier to match natural-language questions.>",
  "measures": ["<cube.measure>", ...],
  "dimensions": ["<cube.dimension>", ...],
  "time_dimension": "<cube.dimension_name if a time dimension exists, else null>",
  "time_granularity": "month"
}

Rules:
- measures: include ALL numeric measures from the cube.
- dimensions: include the most useful string/boolean dimensions (facility,
  category, type, status fields). Skip primary keys and raw IDs.
- time_dimension: use the most useful time dimension if one exists (prefer
  *_month over *_at for granularity). Null if none.
- id must be unique and not conflict with other metric ids.
- Keep the description dense with keywords a user might say when asking
  about this data — this is what drives question matching.
"""

def generate_metric(client: OpenAI, cube: dict, model: str) -> dict | None:
    name = cube["name"]
    measures = cube.get("measures", [])
    dimensions = cube.get("dimensions", [])

    if not measures:
        return None  # nothing to query

    cube_summary = {
        "cube_name": name,
        "measures": [{"name": m["name"], "type": m["type"]} for m in measures],
        "dimensions": [{"name": d["name"], "type": d["type"]} for d in dimensions],
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(cube_summary)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"  [WARN] GPT failed for {name}: {exc}", file=sys.stderr)
        return None


# ── build cube_query dict from LLM output ────────────────────────────────────

def build_cube_query(llm: dict) -> dict:
    query: dict = {}

    measures = llm.get("measures") or []
    if measures:
        query["measures"] = measures

    dimensions = llm.get("dimensions") or []
    if dimensions:
        query["dimensions"] = dimensions
    else:
        query["dimensions"] = []

    td = llm.get("time_dimension")
    if td:
        query["timeDimensions"] = [
            {
                "dimension": td,
                "granularity": llm.get("time_granularity", "month"),
            }
        ]
    else:
        query["timeDimensions"] = []

    query["filters"] = []
    query["limit"] = 500

    return query


# ── yaml serialisation ────────────────────────────────────────────────────────

def metrics_to_yaml(metrics: list[dict]) -> str:
    header = textwrap.dedent("""\
        # Metric Catalog — auto-generated by generate_metrics.py
        #
        # Re-generate any time your Cube schema changes:
        #   python generate_metrics.py
        #
        # All cube/measure/dimension names are verified against /cubejs-api/v1/meta.

    """)

    # Use yaml.dump but with block style for readability
    doc = yaml.dump(
        {"metrics": metrics},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=88,
    )
    return header + doc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate metrics.yaml from Cube /meta")
    parser.add_argument("--cube-url",   default=DEFAULT_CUBE_URL)
    parser.add_argument("--cube-token", default=DEFAULT_CUBE_TOKEN)
    parser.add_argument("--openai-key", default=DEFAULT_OPENAI_KEY)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Cube names to skip",
    )
    args = parser.parse_args()

    skip = SKIP_CUBES | set(args.skip)

    if not args.openai_key:
        sys.exit("ERROR: --openai-key or OPENAI_API_KEY is required")

    print(f"Fetching Cube meta from {args.cube_url} ...")
    cubes = fetch_meta(args.cube_url, args.cube_token)
    print(f"Found {len(cubes)} cubes.")

    client = OpenAI(api_key=args.openai_key)
    metrics = []
    seen_ids: set[str] = set()

    for cube in cubes:
        cube_name = cube["name"]
        if cube_name in skip:
            print(f"  skip  {cube_name}")
            continue

        print(f"  gen   {cube_name} ...", end=" ", flush=True)
        llm = generate_metric(client, cube, args.model)

        if llm is None:
            print("skipped (no measures or LLM error)")
            continue

        # Deduplicate ids
        metric_id = llm.get("id", cube_name)
        if metric_id in seen_ids:
            metric_id = f"{metric_id}_{cube_name}"
        seen_ids.add(metric_id)

        metric = {
            "id":          metric_id,
            "name":        llm.get("name", cube_name),
            "description": llm.get("description", ""),
            "cube_query":  build_cube_query(llm),
        }
        metrics.append(metric)
        print(f"→ {metric_id}")

        time.sleep(0.3)  # stay inside rate limits

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    yaml_text = metrics_to_yaml(metrics)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\nWrote {len(metrics)} metrics → {args.output}")


if __name__ == "__main__":
    main()