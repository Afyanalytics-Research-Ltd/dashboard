"""Turn a Postman collection (v2.0 / v2.1) or an OpenAPI 3 / Swagger 2 document
into a REST API connector proposal: base URL, auth, pagination and endpoints.

    parse(text_or_bytes, filename="") -> {
        "format": "Postman v2.1" | "OpenAPI 3.0.1" | "Swagger 2.0",
        "title": str,
        "config": {base_url, auth_type, token?, username?, password?, api_key_name?,
                   login_path?, login_body?, token_path?, headers?, pagination?,
                   page_param?, size_param?, page_size?, records_path?},
        "endpoints": [{id, name, method, path, body, records, selected, reason, folder, summary}],
        "variables": [{name, value, used_in}],     # Postman {{vars}} still unresolved
        "notes": [str],
    }

Framework-free; the UI shows the endpoints as a picker and writes the chosen ones
into the connector's Streams field (see `stream_line`).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qsl, urlparse

PAGE_KEYS = ("page", "page_number", "pagenumber", "pageno", "p")
SIZE_KEYS = ("per_page", "perpage", "page_size", "pagesize", "size", "limit", "count", "pageSize", "perPage")
OFFSET_KEYS = ("offset", "skip", "start")
CURSOR_KEYS = ("cursor", "after", "next_cursor", "starting_after", "page_token", "pageToken", "continuation")
RECORD_KEYS = ("data", "items", "results", "records", "rows", "entries", "list", "values", "content", "hits")
READ_HINTS = re.compile(r"\b(list|read|search|query|fetch|get|all|index|export|find|browse)\b", re.I)
LOGIN_HINTS = re.compile(r"(login|log-in|signin|sign-in|auth/token|oauth/token|/token$|authenticate|session)", re.I)
WRITE_HINTS = re.compile(r"\b(create|insert|update|delete|remove|add|upload|post|edit|patch|save|cancel|approve)\b", re.I)
VAR = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


class ImportError_(ValueError):
    pass


# --------------------------------------------------------------------------- entry

def parse(content: str | bytes, filename: str = "") -> dict:
    doc = _load(content, filename)
    if not isinstance(doc, dict):
        raise ImportError_("Expected a JSON/YAML object (a Postman collection or an OpenAPI document).")
    if "openapi" in doc or "swagger" in doc:
        out = _openapi(doc)
    elif "item" in doc or "info" in doc and "_postman_id" in (doc.get("info") or {}):
        out = _postman(doc)
    elif isinstance(doc.get("collection"), dict):            # Postman API export wraps it
        out = _postman(doc["collection"])
    else:
        raise ImportError_("Not a Postman collection (no “item”) or OpenAPI document (no “openapi”/“swagger”).")
    _dedupe_names(out["endpoints"])
    for e in out["endpoints"]:                 # credentials stay in the (encrypted) login body only
        if isinstance(e.get("body"), dict):
            e["body"] = _mask(e["body"])
    out["config"] = {k: v for k, v in out["config"].items() if v not in (None, "")}
    return out


SECRET_KEY = re.compile(r"pass|secret|token|api_?key|credential|otp|pin$", re.I)


def _mask(value):
    if isinstance(value, dict):
        return {k: ("••••••" if SECRET_KEY.search(str(k)) and isinstance(v, str) and v and not VAR.fullmatch(v) else _mask(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    return value


def stream_line(ep: dict) -> str:
    """One Streams-field line: name = [METHOD ]path[ {body}][ -> records]."""
    method = (ep.get("method") or "GET").upper()
    parts = [f"{ep['name']} =", ep["path"] if method == "GET" else f"{method} {ep['path']}"]
    if ep.get("body") not in (None, "", {}):
        parts.append(json.dumps(ep["body"], separators=(",", ":")))
    line = " ".join(parts)
    if ep.get("records"):
        line += f" -> {ep['records']}"
    return line


def _load(content, filename):
    text = content.decode("utf-8-sig", "replace") if isinstance(content, bytes) else str(content)
    text = text.strip()
    if not text:
        raise ImportError_("The file is empty.")
    try:
        return json.loads(text)
    except ValueError as e:
        if text[:1] in "{[" and not filename.lower().endswith((".yaml", ".yml")):
            raise ImportError_(f"Not valid JSON: {e}") from None
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001
        raise ImportError_(f"Neither JSON nor YAML: {e}") from None


# --------------------------------------------------------------------------- helpers

def _slug(text: str, fallback="stream") -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text or "")).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    if s and s[0].isdigit():
        s = f"s_{s}"
    return s[:60] or fallback


def _dedupe_names(endpoints):
    seen = Counter()
    for ep in endpoints:
        base = ep["name"]
        seen[base] += 1
        if seen[base] > 1:
            ep["name"] = f"{base}_{seen[base]}"


def _find_records(sample: Any, depth=0) -> str | None:
    """Dot path of the record list inside a response body ('' = the body itself)."""
    if isinstance(sample, list):
        return ""
    if not isinstance(sample, dict) or depth > 3:
        return None
    keys = sorted(sample, key=lambda k: (k.lower() not in RECORD_KEYS, k))
    for k in keys:
        v = sample[k]
        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
            return k
    for k in keys:
        if isinstance(sample[k], dict):
            sub = _find_records(sample[k], depth + 1)
            if sub is not None:
                return f"{k}.{sub}" if sub else k
    return None


def _pagination(keys: list[str], where: str) -> dict:
    low = {k.lower(): k for k in keys}
    size = next((low[k.lower()] for k in SIZE_KEYS if k.lower() in low), None)
    page = next((low[k] for k in PAGE_KEYS if k in low), None)
    if page:
        return {"pagination": "page", "page_param": page, "size_param": size, "_where": where}
    off = next((low[k] for k in OFFSET_KEYS if k in low), None)
    if off:
        return {"pagination": "offset", "page_param": off, "size_param": size, "_where": where}
    cur = next((low[k.lower()] for k in CURSOR_KEYS if k.lower() in low), None)
    if cur:
        return {"pagination": "cursor", "page_param": cur, "_where": where}
    return {}


def _vote(pags: list[dict], notes: list[str]) -> dict:
    pags = [p for p in pags if p]
    if not pags:
        return {}
    key = lambda p: (p["pagination"], p.get("page_param"), p.get("size_param"))
    best, n = Counter(key(p) for p in pags).most_common(1)[0]
    chosen = next(p for p in pags if key(p) == best)
    out = {"pagination": best[0], "page_param": best[1], "size_param": best[2]}
    if best[0] in ("page", "offset"):
        out["page_size"] = 100
    if best[0] == "cursor":
        notes.append("Cursor pagination detected — set “Cursor / next path in body” to where the API returns the next cursor.")
    notes.append(f"Pagination: {best[0]} via “{best[1]}”" + (f" / “{best[2]}”" if best[2] else "")
                 + (" in the request body" if chosen.get("_where") == "body" else "") + f" ({n} endpoint(s)).")
    return out


# --------------------------------------------------------------------------- Postman

def _postman(doc: dict) -> dict:
    info = doc.get("info") or {}
    schema = str(info.get("schema") or "")
    fmt = "Postman v2.1" if "v2.1" in schema else "Postman v2.0" if "v2.0" in schema else "Postman collection"
    variables = {v.get("key"): v.get("value") for v in doc.get("variable") or [] if v.get("key")}
    notes: list[str] = []
    used: dict[str, set] = {}
    requests: list[tuple[list[str], dict, dict | None]] = []    # (folders, item, inherited auth)

    def walk(items, folders, auth):
        for it in items or []:
            a = it.get("auth") or auth
            if isinstance(it.get("item"), list):
                walk(it["item"], folders + [it.get("name") or ""], a)
            elif isinstance(it.get("request"), (dict, str)):
                requests.append((folders, it, a))

    walk(doc.get("item"), [], doc.get("auth"))
    if not requests:
        raise ImportError_("The collection has no requests.")

    def sub(text: str, where: str) -> str:
        def rep(m):
            name = m.group(1)
            val = variables.get(name)
            if val not in (None, ""):
                return str(val)
            used.setdefault(name, set()).add(where)
            return m.group(0)
        return VAR.sub(rep, text)

    parsed = []
    for folders, it, auth in requests:
        req = it["request"] if isinstance(it["request"], dict) else {"url": it["request"], "method": "GET"}
        url = req.get("url")
        raw = url if isinstance(url, str) else (url or {}).get("raw") or ""
        if not raw and isinstance(url, dict):
            host = ".".join(url.get("host") or []) if isinstance(url.get("host"), list) else url.get("host") or ""
            path = "/".join(url.get("path") or []) if isinstance(url.get("path"), list) else url.get("path") or ""
            raw = f"{url.get('protocol', 'https')}://{host}/{path}"
        query = [(q.get("key"), q.get("value")) for q in (url.get("query") or []) if isinstance(url, dict) and not q.get("disabled")] \
            if isinstance(url, dict) else parse_qsl(urlparse(raw).query)
        body = None
        b = req.get("body") or {}
        if b.get("mode") == "raw" and (b.get("raw") or "").strip():
            text = re.sub(r'(?<!")\{\{([^{}]+)\}\}(?!")', r'"{{\1}}"', b["raw"])   # bare {{var}} → valid JSON
            try:
                body = json.loads(text)
            except ValueError:
                body = None
        elif b.get("mode") in ("urlencoded", "formdata"):
            body = {p.get("key"): p.get("value") for p in b.get(b["mode"]) or [] if p.get("key") and not p.get("disabled")}
        sample = None
        for ex in it.get("response") or []:
            try:
                sample = json.loads(ex.get("body") or "")
                break
            except (ValueError, TypeError):
                continue
        parsed.append({"folders": folders, "name": it.get("name") or "", "method": (req.get("method") or "GET").upper(),
                       "raw": raw.split("?")[0], "query": [(k, v) for k, v in query if k], "body": body, "auth": auth,
                       "sample": sample, "headers": req.get("header") or [], "description": _text(req.get("description"))})

    # base URL: a leading {{var}} shared by most requests, else the common scheme://host/prefix
    lead = Counter(m.group(1) for p in parsed if (m := re.match(r"^\{\{\s*([^{}]+?)\s*\}\}", p["raw"])))
    base_var = lead.most_common(1)[0][0] if lead and lead.most_common(1)[0][1] >= len(parsed) / 2 else None
    if base_var:
        base_url = str(variables.get(base_var) or "")
        if not base_url:
            notes.append(f"The base URL comes from the variable {{{{{base_var}}}}}, which has no value in the collection — fill it in.")
        strip = lambda raw: re.sub(r"^\{\{\s*" + re.escape(base_var) + r"\s*\}\}", "", raw) or "/"
    else:
        resolved = [sub(p["raw"], "url") for p in parsed]
        base_url = _common_base(resolved)
        strip = lambda raw: sub(raw, "url")[len(base_url):] if sub(raw, "url").startswith(base_url) else sub(raw, "url")

    config: dict[str, Any] = {"base_url": base_url.rstrip("/")}

    # login request → login auth; else the collection/request auth
    login = next((p for p in parsed if p["method"] == "POST" and LOGIN_HINTS.search(p["raw"] + " " + p["name"])
                  and isinstance(p["body"], dict) and any("pass" in k.lower() or "secret" in k.lower() for k in p["body"])), None)
    auths = [p["auth"] for p in parsed if p["auth"] and p is not login]
    auth = auths[0] if auths else doc.get("auth")
    config.update(_postman_auth(auth, variables, notes))
    if login and (config.get("auth_type") in (None, "none", "bearer") and not config.get("token")):
        lbody = {k: (sub(str(v), "login") if isinstance(v, str) else v) for k, v in login["body"].items()}
        config.update({"auth_type": "login", "login_path": strip(login["raw"]), "login_body": json.dumps(lbody, indent=1),
                       "token_path": _token_path(login["sample"])})
        config.pop("token", None)
        notes[:] = [n for n in notes if not n.startswith("Bearer auth")]
        notes.append(f"Login: “{login['name']}” is used to get a token; it is sent as a Bearer header."
                     + (" Replace the {{variables}} in the login body." if VAR.search(json.dumps(lbody)) else ""))

    # headers set on most requests (minus auth / content negotiation)
    hdr = Counter()
    for p in parsed:
        for hh in p["headers"]:
            k = (hh.get("key") or "").strip()
            if k and not hh.get("disabled") and k.lower() not in ("authorization", "content-type", "accept"):
                hdr[(k, sub(str(hh.get("value") or ""), "headers"))] += 1
    common = [f"{k}: {v}" for (k, v), c in hdr.items() if c >= max(1, len(parsed) // 2)]
    if common:
        config["headers"] = "\n".join(common)

    endpoints, pags = [], []
    for i, p in enumerate(parsed):
        is_login = p is login
        path = strip(p["raw"])
        has_param = bool(re.search(r"(/:[\w]+)|(\{\{[^}]+\}\})|(\{[\w]+\})", path))
        body = p["body"] if isinstance(p["body"], dict) else None
        qkeys = [k for k, _ in p["query"]]
        pag = _pagination(qkeys, "query") or (_pagination(list(body), "body") if body else {})
        static_q = [(k, v) for k, v in p["query"] if k not in (pag.get("page_param"), pag.get("size_param"))]
        if static_q:
            path += "?" + "&".join(f"{k}={sub(str(v or ''), 'query')}" for k, v in static_q)
        if body:
            body = _sub_deep({k: v for k, v in body.items() if k not in (pag.get("page_param"), pag.get("size_param"))},
                             lambda s: sub(s, "body"))
        label = f"{p['name']} {path}"
        if is_login:
            selected, reason = False, "login request (used for auth)"
        elif p["method"] == "GET":
            selected, reason = (False, "needs a path parameter") if has_param else (True, "")
        elif p["method"] == "POST" and body is not None and READ_HINTS.search(label + " " + json.dumps(body)) \
                and not WRITE_HINTS.search(p["name"]):
            selected, reason = (not has_param), ("reads data with a POST body" if not has_param else "needs a path parameter")
        else:
            selected, reason = False, (f"{p['method']} — looks like it writes data" if WRITE_HINTS.search(label) or p["method"] != "POST"
                                       else "doesn't look like a list of records")
        if selected:
            pags.append(pag)
        name = _slug(p["name"]) if p["name"] else _slug(path.split("?")[0].strip("/").split("/")[-1])
        records = _find_records(p["sample"]) if p["sample"] is not None else None
        endpoints.append({"id": f"e{i}", "name": name, "method": p["method"], "path": path, "body": body or None,
                          "records": records, "fields": _sample_fields(p["sample"], records),
                          "selected": selected, "reason": reason, "folder": " / ".join(f for f in p["folders"] if f),
                          "summary": p["description"][:200]})

    config.update({k: v for k, v in _vote(pags, notes).items()})
    recs = Counter(e["records"] for e in endpoints if e["selected"] and e["records"])
    if recs:
        config["records_path"] = recs.most_common(1)[0][0]
    if not any(e["records"] for e in endpoints):
        notes.append("No saved example responses — the record list is detected automatically on the first call.")
    vars_out = [{"name": k, "value": variables.get(k), "used_in": sorted(v)} for k, v in sorted(used.items())]
    if vars_out:
        notes.append(f"{len(vars_out)} variable(s) have no value in the collection: "
                     + ", ".join(f"{{{{{v['name']}}}}}" for v in vars_out[:8]) + ".")
    return {"format": fmt, "title": info.get("name") or "Postman collection", "config": config,
            "endpoints": endpoints, "variables": vars_out, "notes": notes}


def _sub_deep(value, fn):
    """Resolve {{vars}} inside a JSON body; a value that was only a variable keeps its JSON type when it resolves."""
    if isinstance(value, dict):
        return {k: _sub_deep(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_sub_deep(v, fn) for v in value]
    if isinstance(value, str) and VAR.search(value):
        out = fn(value)
        if out != value and VAR.fullmatch(value.strip()):
            try:
                return json.loads(out)
            except ValueError:
                return out
        return out
    return value


def _text(d) -> str:
    if isinstance(d, dict):
        return str(d.get("content") or "")
    return str(d or "")


def _kv(entries) -> dict:
    if isinstance(entries, dict):
        return entries
    return {e.get("key"): e.get("value") for e in entries or [] if isinstance(e, dict)}


def _postman_auth(auth, variables, notes) -> dict:
    if not auth or auth.get("type") in (None, "noauth", "inherit"):
        return {"auth_type": "none"}
    t = auth["type"]
    vals = {k: v for k, v in _kv(auth.get(t)).items()}

    def real(v):
        v = str(v or "")
        m = VAR.fullmatch(v.strip())
        if m:
            return str(variables.get(m.group(1)) or "")
        return v

    if t == "bearer":
        tok = real(vals.get("token"))
        if not tok:
            notes.append("Bearer auth — paste the token (it is a variable without a value in the collection).")
        return {"auth_type": "bearer", "token": tok}
    if t == "basic":
        return {"auth_type": "basic", "username": real(vals.get("username")), "password": real(vals.get("password"))}
    if t == "apikey":
        where = (vals.get("in") or "header").lower()
        return {"auth_type": "api_key_query" if where == "query" else "api_key_header",
                "api_key_name": vals.get("key") or "X-API-Key", "token": real(vals.get("value"))}
    if t == "oauth2":
        notes.append("OAuth 2 — paste an access token; the authorization flow itself is not run here.")
        return {"auth_type": "bearer", "token": real(vals.get("accessToken"))}
    notes.append(f"Auth type “{t}” isn't supported directly — choose an authentication method.")
    return {"auth_type": "none"}


def _token_path(sample) -> str:
    if isinstance(sample, dict):
        for path in ("access_token", "token", "data.token", "data.access_token", "accessToken", "jwt", "id_token"):
            cur = sample
            for part in path.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            if isinstance(cur, str):
                return path
    return ""


def _common_base(urls: list[str]) -> str:
    urls = [u for u in urls if u.startswith(("http://", "https://"))]
    if not urls:
        return ""
    origins = Counter(f"{urlparse(u).scheme}://{urlparse(u).netloc}" for u in urls)
    origin = origins.most_common(1)[0][0]
    paths = [urlparse(u).path.strip("/").split("/") for u in urls if u.startswith(origin)]
    prefix = []
    for parts in zip(*paths):
        if len(set(parts)) == 1 and not re.search(r"[:{]", parts[0]):
            prefix.append(parts[0])
        else:
            break
    if len(paths) > 1 and prefix and all(len(p) == len(prefix) for p in paths):
        prefix = prefix[:-1]                     # keep at least the last segment as the path
    return origin + ("/" + "/".join(prefix) if prefix else "")


# --------------------------------------------------------------------------- OpenAPI / Swagger

def _openapi(doc: dict) -> dict:
    v3 = "openapi" in doc
    fmt = f"OpenAPI {doc.get('openapi')}" if v3 else f"Swagger {doc.get('swagger')}"
    info = doc.get("info") or {}
    notes: list[str] = []

    def ref(obj, depth=0):
        while isinstance(obj, dict) and "$ref" in obj and depth < 20:
            target = doc
            for part in obj["$ref"].lstrip("#/").split("/"):
                target = (target or {}).get(part.replace("~1", "/").replace("~0", "~")) if isinstance(target, dict) else None
            obj, depth = target or {}, depth + 1
        return obj or {}

    if v3:
        servers = doc.get("servers") or []
        base = ""
        if servers:
            s = servers[0]
            base = s.get("url") or ""
            for k, var in (s.get("variables") or {}).items():
                base = base.replace("{" + k + "}", str(var.get("default") or ""))
        if not base.startswith("http"):
            notes.append("The document has no absolute server URL — fill in the base URL." if not base else
                         f"Server URL is relative (“{base}”) — prefix it with the API host.")
    else:
        host = doc.get("host") or ""
        scheme = (doc.get("schemes") or ["https"])[0]
        base = f"{scheme}://{host}{doc.get('basePath') or ''}" if host else (doc.get("basePath") or "")
        if not host:
            notes.append("The document has no host — fill in the base URL.")
    config: dict[str, Any] = {"base_url": base.rstrip("/")}

    # security
    schemes = (doc.get("components") or {}).get("securitySchemes") if v3 else doc.get("securityDefinitions")
    schemes = {k: ref(v) for k, v in (schemes or {}).items()}
    used = [next(iter(req), None) for req in doc.get("security") or [] if req] or list(schemes)
    sch = schemes.get(used[0]) if used and used[0] in schemes else None
    if sch:
        t, scheme_name = sch.get("type"), (sch.get("scheme") or "").lower()
        if t == "http" and scheme_name == "bearer" or t == "oauth2" or t == "openIdConnect":
            config["auth_type"] = "bearer"
            if t != "http":
                notes.append(f"{t} — paste an access token; the authorization flow itself is not run here.")
        elif t == "http" and scheme_name == "basic" or t == "basic":
            config["auth_type"] = "basic"
        elif t == "apiKey":
            config["auth_type"] = "api_key_query" if sch.get("in") == "query" else "api_key_header"
            config["api_key_name"] = sch.get("name") or "X-API-Key"
            if sch.get("in") == "cookie":
                notes.append("API key in a cookie — add it as a “Cookie:” extra header instead.")
                config["auth_type"] = "none"
    else:
        config["auth_type"] = "none"

    endpoints, pags, login = [], [], None
    i = 0
    for path, item in (doc.get("paths") or {}).items():
        item = ref(item)
        shared = item.get("parameters") or []
        for method in ("get", "post", "put", "patch", "delete"):
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            i += 1
            params = [ref(p) for p in shared + (op.get("parameters") or [])]
            path_params = [p for p in params if p.get("in") == "path"]
            qnames = [p.get("name") for p in params if p.get("in") == "query" and p.get("name")]
            body_schema = None
            if v3:
                content = (ref(op.get("requestBody")).get("content") or {})
                body_schema = ref((content.get("application/json") or next(iter(content.values()), {}) or {}).get("schema"))
            else:
                bp = next((p for p in params if p.get("in") == "body"), None)
                body_schema = ref(bp.get("schema")) if bp else None
            body_props = list((body_schema or {}).get("properties") or {})
            pag = _pagination(qnames, "query") or (_pagination(body_props, "body") if body_props else {})
            success = _success_schema(op, v3, ref)
            records = _schema_records(success, ref)
            text = " ".join(str(x) for x in (op.get("operationId"), op.get("summary"), path) if x)
            m = method.upper()
            if m == "POST" and LOGIN_HINTS.search(text) and any("pass" in k.lower() for k in body_props):
                login = (path, body_props, op)
            if m == "GET":
                selected, reason = (False, "needs a path parameter") if path_params else (True, "")
            elif m == "POST" and READ_HINTS.search(text) and not WRITE_HINTS.search(str(op.get("operationId") or op.get("summary") or "")):
                selected, reason = (not path_params), "reads data with a POST body" if not path_params else "needs a path parameter"
            else:
                selected, reason = False, f"{m} — looks like it writes data"
            body = None
            if m != "GET" and body_schema:
                ex = body_schema.get("example")
                body = ex if isinstance(ex, dict) else {k: _example(ref(v), ref) for k, v in (body_schema.get("properties") or {}).items()
                                                        if k in (body_schema.get("required") or []) or len(body_props) <= 6}
                body = {k: v for k, v in (body or {}).items() if k not in (pag.get("page_param"), pag.get("size_param"))} or None
            name = _slug(op.get("operationId") or "_".join(s for s in path.strip("/").split("/") if not s.startswith("{"))[-60:] or path)
            if selected:
                pags.append(pag)
            endpoints.append({"id": f"e{i}", "name": name, "method": m, "path": path, "body": body, "records": records,
                              "fields": _schema_fields(success, records, ref), "selected": selected, "reason": reason,
                              "folder": ", ".join(op.get("tags") or []), "summary": str(op.get("summary") or op.get("description") or "")[:200]})

    if login and config.get("auth_type") in ("none", "bearer"):
        lpath, props, op = login
        config.update({"auth_type": "login", "login_path": lpath,
                       "login_body": json.dumps({k: "" for k in props if k.lower() in ("username", "email", "user", "login", "password", "client_id", "client_secret")} or {k: "" for k in props}, indent=1),
                       "token_path": _token_path(_example(_schema_records_root(_success_schema(op, v3, ref), ref), ref))})
        notes.append(f"Login: {lpath} is used to get a token — fill in the login body.")
    config.update(_vote(pags, notes))
    recs = Counter(e["records"] for e in endpoints if e["selected"] and e["records"] is not None)
    if recs:
        config["records_path"] = recs.most_common(1)[0][0]
    if not endpoints:
        raise ImportError_("The document has no operations under “paths”.")
    return {"format": fmt, "title": info.get("title") or "OpenAPI", "config": config, "endpoints": endpoints,
            "variables": [], "notes": notes}


def _success_schema(op, v3, ref):
    for code, resp in (op.get("responses") or {}).items():
        if str(code).startswith("2"):
            resp = ref(resp)
            if v3:
                content = resp.get("content") or {}
                c = content.get("application/json") or next(iter(content.values()), None) or {}
                return ref(c.get("schema"))
            return ref(resp.get("schema"))
    return {}


def _schema_records_root(schema, ref):
    return ref(schema)


def _sample_fields(sample, records) -> list[str]:
    """Field names of the first record in an example response."""
    if sample is None or records is None:
        return []
    cur = sample
    for part in [p for p in records.split(".") if p]:
        cur = cur.get(part) if isinstance(cur, dict) else None
    if isinstance(cur, list) and cur and isinstance(cur[0], dict):
        return list(cur[0])[:40]
    return []


def _schema_fields(schema, records, ref) -> list[str]:
    """Property names of one record, following the records path through the response schema."""
    if records is None:
        return []
    cur = ref(schema)
    for part in [p for p in records.split(".") if p]:
        for key in ("allOf", "oneOf", "anyOf"):
            if cur.get(key) and not cur.get("properties"):
                cur = ref(next((s for s in cur[key] if ref(s).get("properties")), cur[key][0]))
        cur = ref((cur.get("properties") or {}).get(part))
    item = ref(cur.get("items")) if cur.get("type") == "array" or "items" in cur else cur
    props = dict(item.get("properties") or {})
    for s in item.get("allOf") or []:
        props.update(ref(s).get("properties") or {})
    return list(props)[:40]


# --------------------------------------------------------------------------- AI

AI_SYSTEM = """You configure a REST API source for a data-integration tool (like Airbyte's connector builder).
The source's endpoints become TABLES that will later be mapped onto a target data model, so choose the endpoints
that return lists of business entities (patients, visits, invoices, products …) — one table per entity.

You get the parsed API: base URL, detected auth/pagination, and every endpoint (id, method, path, body, record path,
sample record fields, summary, the parser's guess).

Rules:
- Return ONLY the endpoints that should become tables; everything you leave out is skipped.
- A table endpoint READS a collection of business records. Leave out: login/logout/password/token, health/ping/
  version, metadata (list models, describe, schema, classes, stats, counts), single-record lookups by id/uuid,
  writes (insert/create/update/delete), test or error-demo requests, generic proxies with placeholder paths.
- One table per entity: the same path+model read with extra ids/filters/search/order/trashed options is a
  duplicate — keep the plainest full read.
- table: a short snake_case plural entity name (patients, lab_results), unique.
- body: optional — the request body with filter/ids/search/order/limit keys REMOVED so it reads every record.
  Only remove keys; never add or change values.
- records: dot path to the array of records in the response ("" = the body is the array, null = unknown/auto-detect).
  Keep the parser's value unless you have evidence it is wrong.
- pagination: one of none|page|offset|cursor|link with the parameter names, only if the evidence supports it.
  page/offset need page_param (+ size_param); cursor needs page_param (the request key) AND cursor_path (where the
  response carries the next cursor). Use null when unsure.
- Never invent endpoints; use only the ids given.

Reply with JSON only:
{"endpoints":[{"id":"e3","table":"patients","records":"data","body":{"model":"patient","action":"read"},"why":"lists patients"}],
 "pagination":{"pagination":"page","page_param":"page","size_param":"per_page","page_size":100} | null,
 "records_path":"data" | null,
 "summary":"one sentence: what this API holds and which tables were chosen",
 "notes":["anything the user must still do, e.g. fill in a variable"]}"""

PAGINATIONS = ("none", "page", "offset", "cursor", "link")


def ai_refine(proposal: dict, *, usage=None, chat=None) -> dict:
    """Let the LLM choose the endpoints worth mapping and name them as tables.

    Everything it returns is checked against the parsed proposal; anything invalid is ignored.
    Returns a new proposal with `ai` = {"used": bool, "summary", "error"?}."""
    from . import llm

    chat = chat or llm.chat_json
    eps = proposal["endpoints"]
    compact = [{"id": e["id"], "method": e["method"], "path": e["path"][:200],
                "body": json.dumps(e["body"])[:300] if e.get("body") else None, "records": e.get("records"),
                "fields": (e.get("fields") or [])[:25], "summary": (e.get("summary") or "")[:160],
                "folder": e.get("folder") or None, "guess": "include" if e["selected"] else f"skip ({e['reason']})"}
               for e in eps[:180]]
    cfg = proposal["config"]
    user = json.dumps({"api": proposal.get("title"), "format": proposal.get("format"), "base_url": cfg.get("base_url"),
                       "auth": cfg.get("auth_type"),
                       "pagination": {k: cfg.get(k) for k in ("pagination", "page_param", "size_param") if cfg.get(k)},
                       "records_path": cfg.get("records_path"), "endpoints": compact}, separators=(",", ":"))
    out = {**proposal, "endpoints": [dict(e) for e in eps], "config": dict(cfg), "notes": list(proposal.get("notes") or [])}
    try:
        res = chat(AI_SYSTEM, user, usage=usage, max_tokens=6000)
    except Exception as e:  # noqa: BLE001 — no key, rate limit, bad JSON: keep the parser's proposal
        out["ai"] = {"used": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        return out

    by_id = {e["id"]: e for e in out["endpoints"]}
    chosen, tables = [], set()
    for d in res.get("endpoints") or []:
        e = by_id.get(str(d.get("id"))) if isinstance(d, dict) else None
        if not e or e in chosen or d.get("include") is False:
            continue
        chosen.append(e)
        table = _slug(d.get("table") or e["name"])
        while table in tables:
            table += "_2"
        tables.add(table)
        e["name"] = table
        e["ai"] = {"include": True, "why": str(d.get("why") or "")[:200]}
        rec = d.get("records")
        if isinstance(rec, str) and re.fullmatch(r"[\w.]*", rec):
            e["records"] = rec
        nb = d.get("body")
        if isinstance(nb, dict) and isinstance(e.get("body"), dict) and set(nb) <= set(e["body"]) \
                and all(nb[k] == e["body"][k] for k in nb):
            e["body"] = {k: e["body"][k] for k in e["body"] if k in nb}      # keys removed only, order kept
    if not chosen:
        out["ai"] = {"used": False, "error": "The AI chose no endpoints — kept the parser's selection."}
        return out
    for e in out["endpoints"]:
        was = e["selected"]
        e["selected"] = e in chosen
        if was and not e["selected"]:
            e["reason"] = "AI: not a separate table (metadata, duplicate or lookup)"
    pag = res.get("pagination")
    name_ok = lambda v: isinstance(v, str) and re.fullmatch(r"[\w.\-]+", v)
    if isinstance(pag, dict) and pag.get("pagination") in PAGINATIONS:
        mode = pag["pagination"]
        # a mode is only usable with its parameter names; otherwise keep the parser's choice
        usable = mode in ("none", "link") or name_ok(pag.get("page_param")) and (mode != "cursor" or name_ok(pag.get("cursor_path")))
        if usable:
            for k in ("pagination", "page_param", "size_param", "cursor_path", "page_size"):
                out["config"].pop(k, None)
            out["config"]["pagination"] = mode
            for k in ("page_param", "size_param", "cursor_path"):
                if name_ok(pag.get(k)):
                    out["config"][k] = pag[k]
            if isinstance(pag.get("page_size"), int) and 1 <= pag["page_size"] <= 10000:
                out["config"]["page_size"] = pag["page_size"]
            elif mode in ("page", "offset"):
                out["config"]["page_size"] = 100
        else:
            out["notes"].append(f"The AI suggested {mode} pagination without naming its parameters — set it under Pagination.")
    rp = res.get("records_path")
    if isinstance(rp, str) and re.fullmatch(r"[\w.]*", rp):
        out["config"]["records_path"] = rp
    _dedupe_names([e for e in out["endpoints"] if e["selected"]])
    out["notes"] += [str(n)[:300] for n in (res.get("notes") or [])[:8] if n]
    out["ai"] = {"used": True, "summary": str(res.get("summary") or "")[:400], "included": len(chosen)}
    return out


def _schema_records(schema, ref, depth=0) -> str | None:
    schema = ref(schema)
    if not schema or depth > 3:
        return None
    for key in ("allOf", "oneOf", "anyOf"):
        for s in schema.get(key) or []:
            r = _schema_records(s, ref, depth + 1)
            if r is not None:
                return r
    if schema.get("type") == "array" or "items" in schema and "properties" not in schema:
        return ""
    props = {k: ref(v) for k, v in (schema.get("properties") or {}).items()}
    for k in sorted(props, key=lambda k: (k.lower() not in RECORD_KEYS, k)):
        if props[k].get("type") == "array":
            return k
    for k in sorted(props, key=lambda k: (k.lower() not in RECORD_KEYS, k)):
        if props[k].get("type") == "object" or props[k].get("properties"):
            sub = _schema_records(props[k], ref, depth + 1)
            if sub is not None:
                return f"{k}.{sub}" if sub else k
    return None


def _example(schema, ref, depth=0):
    schema = ref(schema)
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if schema.get("enum"):
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or schema.get("properties"):
        if depth > 3:
            return {}
        return {k: _example(v, ref, depth + 1) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        return []
    return {"integer": 0, "number": 0, "boolean": False}.get(t, "")
