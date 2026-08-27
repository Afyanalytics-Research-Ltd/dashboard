"""
build_embeddings.py
────────────────────
Builds catalog/embeddings.npz — the semantic-retrieval index used by
agents/retrieval.py to search measures, dimensions, catalog metrics, and
business-glossary terms by meaning (not just exact keyword/id match).

Embeds FOUR kinds of entries:
  - metric    — one per catalog/metrics.yaml entry (name + description)
  - measure   — one per Cube measure, from the live /meta endpoint
  - dimension — one per Cube dimension, from the live /meta endpoint
  - glossary  — one per catalog/glossary.yaml term (term + description)

Deliberately reads live /meta directly rather than only catalog/metrics.yaml:
metrics.yaml is a curated subset (~26 entries) of the full live schema
(~62 cubes) — embedding only the curated subset would mean the retriever
can never surface anything metrics.yaml doesn't already cover, which
defeats the point of adding retrieval in the first place.

Usage:
    python catalog/build_embeddings.py \
        --cube-url http://localhost:4000 \
        --cube-token martin \
        --openai-key sk-... \
        --output catalog/embeddings.npz

Or set env vars CUBE_API_URL, CUBE_API_TOKEN, OPENAI_API_KEY and run:
    python catalog/build_embeddings.py

Re-run this whenever any of the following change (three independent
triggers, no single script ties them together on purpose — they change at
different cadences and are owned by different people):
  - model/cubes/*.yml       (data engineers add/change a cube)
  - catalog/glossary.yaml   (analysts add/edit a business term)
  - catalog/metrics.yaml    (someone re-runs generate_metrics.py)
"""

import argparse
import json
import os
import sys
import time

import httpx
import numpy as np
import yaml
from openai import OpenAI

DEFAULT_CUBE_URL   = os.getenv("CUBE_API_URL",   "http://localhost:4000").strip()
DEFAULT_CUBE_TOKEN = os.getenv("CUBE_API_TOKEN",  "martin").strip()
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY",  "").strip()
DEFAULT_OUTPUT     = os.path.join(os.path.dirname(__file__), "embeddings.npz")
DEFAULT_METRICS    = os.path.join(os.path.dirname(__file__), "metrics.yaml")
DEFAULT_GLOSSARY   = os.path.join(os.path.dirname(__file__), "glossary.yaml")
EMBEDDING_MODEL    = "text-embedding-3-small"
BATCH_SIZE         = 100


# ── fetch /meta ───────────────────────────────────────────────────────────────

def fetch_meta(cube_url: str, token: str) -> list[dict]:
    resp = httpx.get(
        f"{cube_url}/cubejs-api/v1/meta",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["cubes"]


def _qualified_name(cube_name: str, field_name: str) -> str:
    """Cube's /meta normally returns already-qualified 'cube.field' names —
    this only prefixes when a meta response ever gives a bare field name."""
    return field_name if "." in field_name else f"{cube_name}.{field_name}"


# ── collect entries to embed ──────────────────────────────────────────────────

def collect_entries(cubes: list[dict], metrics: list[dict], glossary_terms: list[dict]) -> list[dict]:
    """
    Returns a list of {"id", "source", "text", "metadata"} dicts — one per
    thing to embed. "text" is what actually gets sent to the embeddings API;
    "metadata" is what agents/retrieval.py returns to the caller.
    """
    entries: list[dict] = []

    for cube in cubes:
        cube_name = cube["name"]
        for measure in cube.get("measures", []):
            qname = _qualified_name(cube_name, measure["name"])
            label = measure.get("title") or measure.get("shortTitle") or measure["name"]
            desc = (measure.get("description") or "").strip()
            # Metric/glossary entries below lead with human-readable text
            # (name + description) and score noticeably better for the same
            # underlying concept — measures/dimensions used to lead with the
            # technical cube/field identifier instead ("rpt_x field_y (sum):
            # Title"), which dilutes the semantic content that actually
            # matches how people phrase questions. Leading with the label/
            # description here (technical id moved to the end, still present
            # for exact-name lookups) brings measures in line with how
            # metric/glossary entries are already embedded.
            text = label
            if desc:
                text += f". {desc}"
            text += f" ({cube_name}.{measure['name']}, {measure.get('type', '')})"
            entries.append({
                "id": f"measure::{qname}",
                "source": "measure",
                "text": text,
                "metadata": {
                    "cube": cube_name, "field": qname, "kind": "measure",
                    "cube_measure_type": measure.get("type", ""),
                    "label": label, "description": desc,
                    "metric_id": None, "glossary_term": None,
                },
            })
        for dimension in cube.get("dimensions", []):
            qname = _qualified_name(cube_name, dimension["name"])
            label = dimension.get("title") or dimension.get("shortTitle") or dimension["name"]
            desc = (dimension.get("description") or "").strip()
            text = label
            if desc:
                text += f". {desc}"
            text += f" ({cube_name}.{dimension['name']}, {dimension.get('type', '')})"
            entries.append({
                "id": f"dimension::{qname}",
                "source": "dimension",
                "text": text,
                "metadata": {
                    "cube": cube_name, "field": qname, "kind": "dimension",
                    "cube_measure_type": dimension.get("type", ""),
                    "label": label, "description": desc,
                    "metric_id": None, "glossary_term": None,
                },
            })

    for metric in metrics:
        entries.append({
            "id": f"metric::{metric['id']}",
            "source": "metric",
            "text": f"{metric['name']}. {metric['description']}",
            "metadata": {
                "cube": None, "field": None, "kind": "metric",
                "cube_measure_type": "", "label": metric["name"],
                "description": metric["description"],
                "metric_id": metric["id"], "glossary_term": None,
            },
        })

    for entry in glossary_terms:
        term = entry["term"]
        entries.append({
            "id": f"glossary::{term}",
            "source": "glossary",
            "text": f"{term}. {entry.get('description', '')}",
            "metadata": {
                "cube": None, "field": entry.get("maps_to"), "kind": "glossary",
                "cube_measure_type": "", "label": term,
                "description": entry.get("description", ""),
                "metric_id": None, "glossary_term": term,
                "formula": entry.get("formula"), "variables": entry.get("variables"),
            },
        })

    return entries


# ── embed + write ──────────────────────────────────────────────────────────────

def embed_all(client: OpenAI, entries: list[dict]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i + BATCH_SIZE]
        print(f"  embedding {i + 1}-{i + len(batch)} of {len(entries)} ...", flush=True)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[e["text"] for e in batch])
        vectors.extend(d.embedding for d in resp.data)
        time.sleep(0.1)  # stay comfortably inside rate limits
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms  # L2-normalize once at build time


def main():
    parser = argparse.ArgumentParser(description="Build catalog/embeddings.npz for semantic retrieval")
    parser.add_argument("--cube-url",   default=DEFAULT_CUBE_URL)
    parser.add_argument("--cube-token", default=DEFAULT_CUBE_TOKEN)
    parser.add_argument("--openai-key", default=DEFAULT_OPENAI_KEY)
    parser.add_argument("--metrics",    default=DEFAULT_METRICS)
    parser.add_argument("--glossary",   default=DEFAULT_GLOSSARY)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.openai_key:
        sys.exit("ERROR: --openai-key or OPENAI_API_KEY is required")

    print(f"Fetching Cube meta from {args.cube_url} ...")
    cubes = fetch_meta(args.cube_url, args.cube_token)
    print(f"Found {len(cubes)} cubes.")

    with open(args.metrics, "r") as f:
        metrics = (yaml.safe_load(f) or {}).get("metrics", [])
    with open(args.glossary, "r") as f:
        glossary_terms = (yaml.safe_load(f) or {}).get("terms", [])

    entries = collect_entries(cubes, metrics, glossary_terms)
    n_measures   = sum(1 for e in entries if e["source"] == "measure")
    n_dimensions = sum(1 for e in entries if e["source"] == "dimension")
    n_metrics    = sum(1 for e in entries if e["source"] == "metric")
    n_glossary   = sum(1 for e in entries if e["source"] == "glossary")

    client = OpenAI(api_key=args.openai_key)
    vectors = embed_all(client, entries)

    ids = np.array([e["id"] for e in entries], dtype=object)
    sources = np.array([e["source"] for e in entries], dtype=object)
    metadata_json = json.dumps([e["metadata"] for e in entries])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        vectors=vectors,
        ids=ids,
        sources=sources,
        metadata=np.array(metadata_json),
    )

    size_kb = os.path.getsize(args.output) / 1024
    print(
        f"\n{n_metrics} metrics, {n_measures} measures, {n_dimensions} dimensions, "
        f"{n_glossary} glossary terms → {args.output} ({size_kb:.1f} KB)"
    )


if __name__ == "__main__":
    main()
