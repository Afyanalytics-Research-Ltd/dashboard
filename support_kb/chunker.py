"""Turn the support sources into retrieval chunks.

Every chunk carries a short context header (source, screen/section path, page address)
that is embedded together with the body, so a chunk is findable even when its own text
never repeats the screen's name ("contextual" chunking).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

MAX_CHARS = 2200   # ~550 tokens: small enough to rank precisely, big enough to hold a procedure
OVERLAP_LINES = 2

SOURCE_LABELS = {
    "kb": "HMIS setup and user guide",
    "screens": "HMIS screen reference",
    "api": "HMIS backend API reference",
}


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    breadcrumb: str
    url_path: str
    body: str
    meta: dict = field(default_factory=dict)

    @property
    def content(self) -> str:
        head = [f"[{SOURCE_LABELS.get(self.source, self.source)}] {self.breadcrumb or self.title}"]
        if self.url_path:
            head.append(f"Page address: {self.url_path}")
        return "\n".join(head) + "\n\n" + self.body.strip()

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.content.encode()).hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80] or "x"


def _split(lines: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Pack lines into pieces of at most max_chars, carrying a little overlap."""
    flat: list[str] = []
    for ln in lines:  # a single over-long line (a huge list) is cut at '; ' / '. ' boundaries
        while len(ln) > max_chars:
            cut = max(ln.rfind("; ", 0, max_chars), ln.rfind(". ", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            head, ln = ln[:cut], ln[cut:].lstrip()
            flat.append(head)
            label = head.split(":", 1)[0] if ":" in head[:60] else ""
            if label:
                ln = f"{label} (cont.): {ln}"
        flat.append(ln)
    lines = flat
    pieces, cur, size = [], [], 0
    for ln in lines:
        if cur and size + len(ln) + 1 > max_chars:
            pieces.append("\n".join(cur))
            cur = cur[-OVERLAP_LINES:] if OVERLAP_LINES else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        size += len(ln) + 1
    if cur and any(x.strip() for x in cur):
        pieces.append("\n".join(cur))
    return pieces


# ── Markdown (setup/user guide, API reference) ───────────────────────────────

def chunk_markdown(text: str, source: str, max_level: int = 3) -> list[Chunk]:
    """Split on headings down to `max_level`; deeper headings stay inside the body."""
    chunks: list[Chunk] = []
    path: list[str] = []
    body: list[str] = []
    doc_title = ""

    def flush():
        if not path or not any(b.strip() for b in body):
            return
        crumb = " > ".join(path)
        url = ""
        m = re.search(r"Page address:\s*`([^`]+)`", "\n".join(body))
        if m:
            url = m.group(1)
        # FAQ-style sections ("**Question?**" + answer): one chunk per question, so each
        # answer is scored on its own instead of being buried mid-chunk.
        qs = [i for i, ln in enumerate(body) if re.match(r"^\*\*[^*]{5,}[?.\"”]\*\*\s*$", ln.strip())]
        if len(qs) >= 3:
            pre = body[: qs[0]]
            if any(x.strip() for x in pre):
                for i, piece in enumerate(_split(pre)):
                    chunks.append(Chunk(f"{source}:{_slug(crumb)}:intro{i}", source, path[-1], crumb, url, piece))
            for a, b in zip(qs, qs[1:] + [len(body)]):
                q = body[a].strip().strip("*")
                for i, piece in enumerate(_split(body[a:b])):
                    chunks.append(Chunk(f"{source}:{_slug(crumb)}:{_slug(q)}:{i}", source, q, f"{crumb} > {q}",
                                        url, piece))
            return
        for i, piece in enumerate(_split(body)):
            cid = f"{source}:{_slug(crumb)}:{i}"
            chunks.append(Chunk(cid, source, path[-1], crumb, url, piece))

    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if m and len(m.group(1)) <= max_level:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                doc_title = title
                continue
            flush()
            body = []
            path = path[: level - 2] + [title]
            continue
        body.append(line)
    flush()
    if doc_title:
        for c in chunks:
            c.meta["doc"] = doc_title
    return chunks


# ── HTML screen reference ────────────────────────────────────────────────────

class _ScreenParser(HTMLParser):
    """Collect <article> blocks: h3 title, intro <p>s, and (h4, [li...]) groups."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.module = ""
        self.articles: list[dict] = []
        self.cur: dict | None = None
        self.tag = None
        self.buf = ""

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.cur = {"id": dict(attrs).get("id", ""), "module": self.module, "title": "",
                        "intro": [], "groups": []}
        if tag in ("h2", "h3", "h4", "p", "li"):
            self.tag, self.buf = tag, ""

    def handle_data(self, data):
        if self.tag:
            self.buf += data

    def handle_endtag(self, tag):
        if tag != self.tag:
            if tag == "article" and self.cur:
                self.articles.append(self.cur)
                self.cur = None
            return
        text = re.sub(r"\s+", " ", unescape(self.buf)).strip()
        self.tag = None
        if tag == "h2":
            self.module = text
        elif self.cur is None:
            return
        elif tag == "h3":
            self.cur["title"] = text
        elif tag == "p":
            self.cur["intro"].append(text)
        elif tag == "h4":
            self.cur["groups"].append([re.sub(r"\s+—\s+.*$", "", text), []])
        elif tag == "li" and self.cur["groups"]:
            self.cur["groups"][-1][1].append(text)


def chunk_screens_html(text: str, source: str = "screens") -> list[Chunk]:
    p = _ScreenParser()
    p.feed(text)
    chunks: list[Chunk] = []
    for a in p.articles:
        if not a["title"]:
            continue
        intro = " ".join(a["intro"])
        url = ""
        m = re.search(r"Page address:\s*([^\s.]+(?:\.[^\s]+)?)\.?\s", intro + " ")
        if m:
            url = m.group(1).rstrip(".")
        crumb = f"{a['module']} > {a['title']}"
        lines = [intro] if intro else []
        for heading, items in a["groups"]:
            if items:
                lines.append(f"{heading}: " + "; ".join(items))
        for i, piece in enumerate(_split(lines)):
            chunks.append(Chunk(f"{source}:{a['id'] or _slug(crumb)}:{i}", source, a["title"], crumb, url, piece,
                                {"module": a["module"]}))
    return chunks


# ── All sources ──────────────────────────────────────────────────────────────

SOURCES = [
    ("kb", "FRONTEND_HMIS_KNOWLEDGE_BASE.md"),
    ("screens", "HMIS_SCREEN_CONTENT.html"),
    ("api", "API_ENDPOINTS.md"),
]


def load_all(src_dir: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for source, fname in SOURCES:
        f = src_dir / fname
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if fname.endswith(".html"):
            out += chunk_screens_html(text, source)
        else:
            out += chunk_markdown(text, source)
    # ids must be unique even if two headings slug the same
    seen: dict[str, int] = {}
    for c in out:
        n = seen.get(c.id, 0)
        seen[c.id] = n + 1
        if n:
            c.id = f"{c.id}~{n}"
    return out
