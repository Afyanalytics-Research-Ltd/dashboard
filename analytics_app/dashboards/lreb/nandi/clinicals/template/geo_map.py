"""
geo_map.py: interactive Kenya -> county -> sub-county -> ward maps.

The maps run inside a Streamlit components iframe so that drilling in and out is
instant and does not rerun the page. Data is prepared in Python (views/views_geo.py)
and passed as a JSON payload; drawing happens in geo_engine.js.

Map: Leaflet (bundled in template/leaflet.js and leaflet.css, BSD-2) over a street basemap, so
it zooms and pans like any web map. The basemap tiles are fetched by the browser from OpenStreetMap (no key needed);
set NANDI_TILE_URL to use another provider.
County outlines: template/kenya_counties.json (svg) and kenya_counties_ll.json (lon/lat).
Sub-county and ward shapes: payload["bounds"] (BOUNDARY_GEOJSON from the geo lookup
table) when present, otherwise a schematic subdivision of the county outline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit.components.v1 as components

_DIR = Path(__file__).parent
_KP = _JS = _KLL = _LCSS = _LJS = _STAMP = None

IFRAME_CSS = """
:root{--primary:#0F6E56;--blue:#0C447C;--critical:#A32D2D;--amber:#D97706;--neutral:#888780;--bg:#fff;--surface2:#F0F1F5;--border:#E5E7EB;--text:#111827;--muted:#6B7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,'Segoe UI',system-ui,sans-serif;font-size:14px;color:var(--text);background:transparent;line-height:1.5;font-variant-numeric:tabular-nums}
svg text{font-family:Inter,'Segoe UI',system-ui,sans-serif}
.card{background:#fff;border:1px solid var(--border);border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600}
.card-note{font-size:11px;font-style:italic;color:var(--muted);margin:6px 0 0}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media(max-width:820px){.g2{grid-template-columns:1fr}}
.ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:8px;font-size:11px;color:var(--muted)}
.ctl select{font:inherit;font-size:12px;padding:4px 8px;border:1px solid var(--border);border-radius:6px;background:#fff;color:var(--text);margin-left:4px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:6px;overflow:hidden}
.seg button{font:inherit;font-size:11px;padding:4px 10px;border:0;background:#fff;color:var(--muted);cursor:pointer}
.seg button.on{background:var(--primary);color:#fff;font-weight:600}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.pills button{font:inherit;font-size:11px;padding:4px 10px;border-radius:99px;border:1px solid var(--border);background:#fff;color:var(--text);cursor:pointer}
.pills button.on{background:var(--blue);color:#fff;border-color:var(--blue);font-weight:600}
.legend{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:11px;color:var(--muted);margin:4px 0 6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.crumb{font-size:12px;color:var(--muted);margin-bottom:8px}.crumb b{color:var(--text)}
.back{color:var(--blue);font-weight:600;cursor:pointer}.back:hover{text-decoration:underline}
.mini-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:900px){.mini-grid{grid-template-columns:repeat(2,1fr)}}
.mini{border:1px solid var(--border);border-radius:6px;padding:6px}.mini .t{font-size:11px;font-weight:600;margin-bottom:2px}
.explain{background:#F8FAFC;border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:12px;margin-top:10px;line-height:1.55}
table{border-collapse:collapse;width:100%;font-size:12px}
th{padding:6px 8px;font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);text-align:left}
td{padding:7px 8px;border-bottom:1px solid var(--border)}
.lmap{width:100%;border-radius:6px;overflow:hidden;position:relative;background:#E8EDF2;z-index:0}
.leaflet-container{font:inherit}
.leaflet-control-attribution{font-size:10px!important;line-height:1.35}
.mini .leaflet-control-zoom{display:none}
.lb{position:absolute;transform:translate(-50%,-50%);font-size:11px;font-weight:600;color:#111827;white-space:nowrap;text-align:center;pointer-events:none;line-height:1.15;text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 2px #fff}
.lb small{display:block;font-weight:500;font-size:10px;color:#374151}
.lb.dk{color:#fff;text-shadow:0 0 3px rgba(0,0,0,.6)}.lb.dk small{color:#fff}
.maphint{position:absolute;left:8px;bottom:22px;z-index:500;background:rgba(255,255,255,.92);font-size:10px;padding:2px 7px;border-radius:4px;color:#6B7280;pointer-events:none}
.leaflet-tooltip.lt{font-size:11px;line-height:1.4;padding:6px 8px;border:0;border-radius:5px;background:#111827;color:#fff;box-shadow:0 2px 6px rgba(0,0,0,.25)}
.leaflet-tooltip-top.lt:before{border-top-color:#111827}
.lgd-row{display:flex;flex-direction:column;gap:8px;margin-top:10px}.lgd-sws{display:flex;flex-wrap:wrap;gap:6px 22px}.lgd-sws .lgd-sw{margin:0}.lgd{min-width:240px}.lgd-t{font-size:11px;font-weight:600;color:var(--text);margin-bottom:5px}.lgd-bar{height:10px;border-radius:5px;width:260px;border:1px solid var(--border)}.lgd-tk{display:flex;justify-content:space-between;width:260px;font-size:10px;color:var(--muted);margin-top:2px}.lgd-s{font-size:10px;font-style:italic;color:var(--muted);margin-top:3px}.lgd-sw{display:flex;align-items:center;font-size:11px;color:var(--muted);margin-bottom:4px}.lgd-sw i{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:7px}
.tip{position:fixed;pointer-events:none;background:#111827;color:#fff;font-size:11px;padding:6px 8px;border-radius:5px;display:none;z-index:9;line-height:1.4;max-width:220px}
"""


def _assets() -> tuple[str, str, str, str, str]:
    """Read the bundled files, re-reading when any of them changes on disk."""
    global _KP, _JS, _KLL, _LCSS, _LJS, _STAMP
    names = ("kenya_counties.json", "geo_engine.js", "kenya_counties_ll.json", "leaflet.css", "leaflet.js")
    stamp = tuple((_DIR / n).stat().st_mtime for n in names)
    if _KP is None or stamp != _STAMP:
        rd = lambda n: (_DIR / n).read_text(encoding="utf-8")  # noqa: E731
        _KP, _JS, _KLL, _LCSS, _LJS = (rd(n) for n in names)
        _STAMP = stamp
    return _KP, _JS, _KLL, _LCSS, _LJS


def geo_html(kind: str, payload: dict) -> str:
    kp, js, kll, lcss, ljs = _assets()
    # Basemap tiles. Default is OpenStreetMap's standard tiles, which need no key. To use another
    # provider (MapTiler, Stadia, Esri...) set NANDI_TILE_URL (and optionally NANDI_TILE_ATTR) in .env.
    if os.getenv("NANDI_TILE_URL"):
        payload = {**payload, "tile": dict(url=os.environ["NANDI_TILE_URL"], attr=os.getenv("NANDI_TILE_ATTR", "Map tiles"))}
    data = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>" + lcss + IFRAME_CSS + "</style></head>"
        "<body><div id='root'></div><script>" + ljs + "</script><script>"
        f"window.KP={kp};window.KLL={kll};window.PAYLOAD={data};window.KIND={json.dumps(kind)};"
        "</script><script>" + js + "</script></body></html>"
    )


def render_geo(kind: str, payload: dict, height: int = 760, scrolling: bool = True) -> None:
    """kind: catchment | rings | disease | shift | female | ncd"""
    components.html(geo_html(kind, payload), height=height, scrolling=scrolling)
