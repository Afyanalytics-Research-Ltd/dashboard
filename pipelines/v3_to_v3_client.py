"""
v3_to_v3_client.py — a gateway client bound to ONE V3 instance + tenant, so a
V3 → V3 copy can hold two at once (the v2_to_v3 module keeps a single global
login). Settings per side come from env, keyed like the other V3 DAGs:

  AFYA_<KEY>_USERNAME / _PASSWORD          the account (its org IS the tenant)
  AFYA_<KEY>_URL_TEMPLATE                   e.g. https://{service}.afyaanalytics.ai/api/
  AFYA_<KEY>_<SERVICE>_URL                  per-service override
  AFYA_<KEY>_ORGANIZATION_ID / _FACILITY_ID expected tenant (login must agree)
  AFYA_<KEY>_MIGRATION_KEY                  X-Migration-Key for theatre / dialysis
                                            (falls back to MODEL_GATEWAY_MIGRATION_KEY)
  AFYA_<KEY>_PLATFORM=1                     a platform account (e.g. "analyst") acting FOR the
                                            configured ORGANIZATION_ID / FACILITY_ID: reads send
                                            source_tenant_id, writes destination_tenant_id and
                                            destination_facility_id of that org — never the
                                            login's own org. Both ids are required.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

SERVICES = ("core", "reception", "evaluation", "inventory", "finance", "theatre", "inpatient", "dialysis")
DEFAULT_URL_TEMPLATE = "https://{service}migrate.afyaanalytics.ai/api/"
MIGRATION_KEY_SERVICES = {"theatre", "dialysis"}
TOKEN_TTL = 50 * 60
MAX_PER_PAGE = 500                       # the gateway's validation cap
# connections kept open per host: a model's POST workers and page readers
# share them, so this bounds the useful concurrency of one client
HTTP_POOL = int(os.getenv("V3V3_HTTP_POOL") or 96)
# a platform account's grant: 60 calls a minute per instance over every org (429 "Try again in N s")
PLATFORM_RATE = int(os.getenv("V3V3_PLATFORM_RATE") or 55)
RATE_LIMIT_WAITS = 40
_RATE_RETRY = re.compile(r"Try again in (\d+) ?s")


_SCHEMA_ERROR_RE = re.compile(r'"sqlstate"\s*:\s*"42S(02|22)"|Base table or view not found|Unknown column '
                              r"|Table '[^']+' doesn't exist")


def schema_error(text: str) -> bool:
    """V3's database is missing the model's table (42S02) or a column (42S22) — a permanent fault on V3's
    side (a model the gateway lists but whose service DB was never migrated), not a transient 500."""
    return bool(_SCHEMA_ERROR_RE.search(text or ""))


class GatewayError(RuntimeError):
    pass


class FilterUnsupported(GatewayError):
    """The instance ignores read filters (updated_since / after_id): fall back to tail / full reads."""


class V3Client:
    def __init__(self, key: str, *, urls: dict[str, str] | None = None, username: str | None = None,
                 password: str | None = None, organization_id: int | None = None,
                 facility_id: int | None = None, migration_key: str | None = None,
                 session: requests.Session | None = None, pool_size: int = HTTP_POOL):
        self.key = key
        up = key.upper()
        env = lambda name: (os.getenv(f"AFYA_{up}_{name}") or "").strip()
        tmpl = env("URL_TEMPLATE") or DEFAULT_URL_TEMPLATE
        self.urls = dict(urls or {s: env(f"{s.upper()}_URL") or tmpl.format(service=s) for s in SERVICES})
        self.username = username or env("USERNAME")
        self.password = password or env("PASSWORD")
        org, fac = env("ORGANIZATION_ID"), env("FACILITY_ID")
        self.organization_id = organization_id if organization_id is not None else (int(org) if org.isdigit() else None)
        self.facility_id = facility_id if facility_id is not None else (int(fac) if fac.isdigit() else None)
        self.migration_key = migration_key or env("MIGRATION_KEY") or (os.getenv("MODEL_GATEWAY_MIGRATION_KEY") or "").strip()
        self.platform = env("PLATFORM").lower() in ("1", "true", "yes")
        if self.platform and (self.organization_id is None or self.facility_id is None):
            raise GatewayError(f"{key}: a platform account needs AFYA_{up}_ORGANIZATION_ID and _FACILITY_ID "
                               "(the org it acts for) — never its own")
        self.user_id: int | None = None
        self.send_facility_header = False          # X-Facility-Id on every call (eagle_to_v3: one client per facility)
        self._session = session or requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=len(SERVICES), pool_maxsize=pool_size,
                                                max_retries=0, pool_block=True)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._token: tuple[str, float] | None = None
        self._lock = threading.Lock()

    def __repr__(self):
        return f"V3Client({self.key}: org {self.organization_id} / facility {self.facility_id} @ {self.urls['core']})"

    # ── auth ──────────────────────────────────────────────────────────────
    _platform_tokens: dict[tuple, tuple] = {}     # (core url, username) → (token, user id, at): one login per process
    _platform_lock = threading.Lock()

    def login(self) -> str:
        if not self.username or not self.password:
            raise GatewayError(f"{self.key}: set AFYA_{self.key.upper()}_USERNAME / _PASSWORD")
        if self.platform:
            return self._platform_login()
        url = f"{self.urls['core'].rstrip('/')}/v1/login"
        body = {"username": self.username, "password": self.password}
        r = self._session.post(url, json=body, timeout=30)
        if r.status_code != 200:
            raise GatewayError(f"{self.key}: login failed {r.status_code} {r.text[:200]}")
        res = r.json()
        if res.get("step") == 2:                      # several facilities: name ours
            # older cores send allowed_facility_ids; current ones a "facilities" list ({id, name, region…})
            allowed = res.get("allowed_facility_ids") or [f.get("id") for f in (res.get("facilities") or []) if f.get("id")]
            if self.facility_id is not None and self.facility_id not in allowed:
                raise GatewayError(f"{self.key}: account can't log into facility {self.facility_id} ({allowed})")
            body["facility_id"] = self.facility_id if self.facility_id is not None else allowed[0]
            r = self._session.post(url, json=body, timeout=30)
            if r.status_code != 200:
                raise GatewayError(f"{self.key}: login (step 2) failed {r.status_code} {r.text[:200]}")
            res = r.json()
        token = res.get("access_token")
        if not token:
            raise GatewayError(f"{self.key}: no access_token in login response")
        user = res.get("user") or res.get("data") or {}
        org = (res.get("tenant") or {}).get("id") or user.get("organization_id")
        fac = ((res.get("facility") or {}).get("id") or (res.get("allowed_facility_ids") or [None])[0]
               or user.get("facility_id"))
        # the tenant comes from the account: a wrong account would silently
        # read or write another organisation — refuse instead
        if self.organization_id is not None and org != self.organization_id:
            raise GatewayError(f"{self.key}: account belongs to organization {org}, expected {self.organization_id}")
        self.organization_id = org
        if self.facility_id is None:
            self.facility_id = fac
        self.user_id = user.get("id")
        log.info("%s: login ok — org %s facility %s", self.key, org, self.facility_id)
        return token

    def _platform_login(self) -> str:
        """A platform account's token, shared by every client of the same instance and
        account in this process (one client per org). Its own org / facility are NOT
        adopted: this client acts for the configured ones."""
        core = self.urls["core"].rstrip("/")
        k = (core, self.username)
        with V3Client._platform_lock:
            hit = V3Client._platform_tokens.get(k)
            if hit and time.time() - hit[2] < TOKEN_TTL - 60:
                self.user_id = hit[1]
                return hit[0]
            r = self._session.post(f"{core}/v1/login", json={"username": self.username, "password": self.password},
                                   timeout=60)
            if r.status_code != 200:
                raise GatewayError(f"{self.key}: platform login failed {r.status_code} {r.text[:200]}")
            res = r.json()
            token = res.get("access_token")
            if not token:
                raise GatewayError(f"{self.key}: no access_token in platform login response")
            self.user_id = (res.get("user") or {}).get("id")
            V3Client._platform_tokens[k] = (token, self.user_id, time.time())
            log.info("%s: platform login ok (%s) — acting for org %s facility %s",
                     self.key, self.username, self.organization_id, self.facility_id)
            return token

    def token(self) -> str:
        with self._lock:
            if not self._token or time.time() - self._token[1] > TOKEN_TTL:
                self._token = (self.login(), time.time())
            return self._token[0]

    def _headers(self, service: str) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json",
             "Authorization": f"Bearer {self.token()}"}
        if service in MIGRATION_KEY_SERVICES and self.migration_key:
            h["X-Migration-Key"] = self.migration_key
        # a superadmin's login has every facility in scope and no active one, so V3 files new rows under no
        # facility (or the column default 1); the header makes this client's facility the active one
        if self.send_facility_header and self.facility_id is not None:
            h["X-Facility-Id"] = str(self.facility_id)
        return h

    # ── gateway ───────────────────────────────────────────────────────────
    def post(self, service: str, body: dict, *, timeout: int = 120, retries: int = 5,
             retry_500: bool = False) -> requests.Response:
        """One gateway call. Connection errors, 429 and 502–504 (and 500 with
        retry_500 — reads, which change nothing) are retried with jittered
        backoff, so threads that fail together don't all retry together."""
        url = f"{self.urls[service].rstrip('/')}/v1/gateway"
        data = json.dumps(body, default=str).encode()
        transient = {429, 502, 503, 504} | ({500} if retry_500 else set())
        wait = 2
        rate_waits = 0
        attempt = 0
        while attempt < retries:
            attempt += 1
            self._throttle()
            try:
                r = self._session.post(url, headers=self._headers(service), data=data, timeout=timeout)
            except requests.RequestException as e:
                if attempt == retries:
                    raise
                log.warning("%s %s: %s — retry in ~%ss", self.key, service, e, wait)
                time.sleep(wait * random.uniform(0.5, 1.5))
                wait = min(wait * 2, 60)
                continue
            if r.status_code == 401 and attempt < retries:
                with self._lock:
                    self._token = None
                if self.platform:
                    with V3Client._platform_lock:
                        V3Client._platform_tokens.pop((self.urls["core"].rstrip("/"), self.username), None)
                continue
            m = _RATE_RETRY.search(r.text) if r.status_code == 429 else None
            if m and rate_waits < RATE_LIMIT_WAITS:
                # the account's per-minute grant (60 calls over every org): wait as long as V3 says,
                # without using up an attempt — a burst must slow down, not fail
                rate_waits += 1
                attempt -= 1
                time.sleep(int(m.group(1)) + random.uniform(1, 3))
                continue
            if r.status_code in transient and attempt < retries:
                if r.status_code == 500 and schema_error(r.text):
                    return r                          # a missing table / column on V3: retrying can't fix it
                time.sleep(wait * random.uniform(0.5, 1.5))
                wait = min(wait * 2, 60)
                continue
            return r
        return r

    # one bucket per instance + account: the grant is per account over every org (60 a minute)
    _buckets: dict[tuple, deque] = {}
    _bucket_lock = threading.Lock()

    def _throttle(self) -> None:
        """Platform accounts: at most PLATFORM_RATE calls a minute per instance, shared by every
        client and thread of this process, so a run stays under V3's grant instead of
        bouncing off 429s."""
        if not self.platform or PLATFORM_RATE <= 0:
            return
        k = (self.urls["core"].rstrip("/"), self.username)
        while True:
            with V3Client._bucket_lock:
                q = V3Client._buckets.setdefault(k, deque())
                now = time.time()
                while q and now - q[0] >= 60:
                    q.popleft()
                if len(q) < PLATFORM_RATE:
                    q.append(now)
                    return
                pause = 60 - (now - q[0]) + 0.05
            time.sleep(pause)

    def _call(self, service: str, body: dict, what: str, **kw) -> requests.Response:
        """post(), with a host that can't be reached (DNS, TLS, refused) as a GatewayError like any other —
        callers skip a service that isn't deployed on this instance instead of crashing."""
        try:
            return self.post(service, body, **kw)
        except requests.RequestException as e:
            raise GatewayError(f"{self.key} {what}: {type(e).__name__}: {e}") from e

    def models(self, service: str) -> list[dict]:
        r = self._call(service, {"action": "list"}, f"{service} list", timeout=60)
        if not r.ok:
            raise GatewayError(f"{self.key} {service} list: {r.status_code} {r.text[:200]}")
        return [e for e in (r.json().get("data") or []) if e.get("alias")]

    def describe(self, service: str, alias: str) -> dict:
        r = self._call(service, {"action": "describe", "model": alias}, f"{service}/{alias} describe", timeout=60)
        if not r.ok:
            raise GatewayError(f"{self.key} {service}/{alias} describe: {r.status_code} {r.text[:200]}")
        return r.json().get("data") or {}

    def read_window(self, service: str, alias: str, since: str, before: str | None = None,
                    per_page: int = MAX_PER_PAGE, after_id: int = 0):
        """Rows updated in [since, before) — the gateway's updated_since / updated_before,
        paged by id with after_id / meta.next_after_id (generator). Raises
        FilterUnsupported when this instance ignores the filter (no next_after_id in meta)."""
        self.token()
        after = after_id
        while True:
            body = {"action": "read", "model": alias, "source_tenant_id": self.organization_id,
                    "per_page": min(per_page, MAX_PER_PAGE), "page": 1, "updated_since": since, "after_id": after}
            if before:
                body["updated_before"] = before
            try:
                r = self.post(service, body, timeout=120, retry_500=True, retries=8)
            except requests.RequestException as e:
                raise GatewayError(f"{self.key} {service}/{alias} read since {since}: {type(e).__name__}: {e}") from e
            if not r.ok:
                raise GatewayError(f"{self.key} {service}/{alias} read since {since}: {r.status_code} {r.text[:200]}")
            j = r.json()
            meta = j.get("meta") or {}
            if "next_after_id" not in meta:
                raise FilterUnsupported(f"{self.key} {service}/{alias}: updated_since / after_id not supported here")
            rows = j.get("data") or []
            if isinstance(rows, dict):
                rows = rows.get("data") or list(rows.values())
            yield from rows
            nxt = meta.get("next_after_id")
            if not nxt or not rows:
                return
            after = nxt

    def org_has(self, service: str, alias: str, row_id: int) -> bool:
        """Is row_id one of this tenant's rows? One call: the tenant's first row with id > row_id-1."""
        self.token()
        body = {"action": "read", "model": alias, "source_tenant_id": self.organization_id,
                "per_page": 1, "page": 1, "after_id": int(row_id) - 1}
        r = self.post(service, body, timeout=60, retry_500=True, retries=8)
        if not r.ok:
            raise GatewayError(f"{self.key} {service}/{alias} id {row_id}: {r.status_code} {r.text[:200]}")
        j = r.json()
        if "next_after_id" not in (j.get("meta") or {}):
            raise FilterUnsupported(f"{self.key} {service}/{alias}: after_id not supported here")
        rows = j.get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("data") or []
        return bool(rows) and rows[0].get("id") == int(row_id)

    def read_page(self, service: str, alias: str, page: int, per_page: int = MAX_PER_PAGE) -> tuple[list[dict], dict]:
        self.token()                                  # the tenant id comes from the login
        body = {"action": "read", "model": alias, "source_tenant_id": self.organization_id,
                "per_page": min(per_page, MAX_PER_PAGE), "page": page}
        try:
            r = self.post(service, body, timeout=120, retry_500=True, retries=8)
        except requests.RequestException as e:      # the server kept dropping us: a gateway error like any other
            raise GatewayError(f"{self.key} {service}/{alias} read p{page}: {type(e).__name__}: {e}") from e
        if not r.ok:
            raise GatewayError(f"{self.key} {service}/{alias} read p{page}: {r.status_code} {r.text[:200]}")
        j = r.json()
        rows = j.get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("data") or list(rows.values())
        return rows, (j.get("meta") or {})

    def count(self, service: str, alias: str) -> int | None:
        _, meta = self.read_page(service, alias, 1, per_page=1)
        return meta.get("total")

    def read_all(self, service: str, alias: str, per_page: int = MAX_PER_PAGE, workers: int = 1):
        """Every row this tenant has, in page order (generator).

        workers > 1 reads pages ahead in parallel: the gateway caps a page at
        500 rows, so one reader tops out at a few hundred rows/s. Rows still
        come out in page order, and at most 2×workers pages are in memory. A
        row seen twice (offset paging over a table that is being written to)
        is yielded once."""
        per_page = min(per_page, MAX_PER_PAGE)
        seen: set = set()

        def fresh(batch):
            for r in batch:
                rid = r.get("id")
                if rid is not None:
                    if rid in seen:
                        continue
                    seen.add(rid)
                yield r

        def full_page(p: int, last_page: int | None) -> tuple[list[dict], dict]:
            """A page that isn't the last must be full: an overloaded server sometimes
            answers 200 with fewer rows (or none). Re-read it; if it stays short, stop
            loudly — a silently skipped page is a silently missing table."""
            rows_, meta_ = self.read_page(service, alias, p, per_page)
            tries = 0
            while last_page and p < last_page and len(rows_) < per_page and tries < 5:
                tries += 1
                log.warning("%s %s/%s page %d/%d came back with %d rows — reading it again", self.key, service,
                            alias, p, last_page, len(rows_))
                time.sleep(2 * tries)
                rows_, meta_ = self.read_page(service, alias, p, per_page)
            if last_page and p < last_page and len(rows_) < per_page:
                raise GatewayError(f"{self.key} {service}/{alias}: page {p} of {last_page} keeps coming back with "
                                   f"{len(rows_)} of {per_page} rows — the server is dropping rows; not continuing "
                                   f"with a partial read")
            return rows_, meta_

        rows, meta = self.read_page(service, alias, 1, per_page)
        last = meta.get("last_page")
        if last and last > 1 and len(rows) < per_page:
            rows, meta = full_page(1, last)
        yield from fresh(rows)
        if not rows or len(rows) < per_page or (last and last <= 1):
            return
        page = 2
        if workers > 1 and last:
            with ThreadPoolExecutor(workers, thread_name_prefix=f"read-{alias}") as pool:
                window: deque = deque()
                try:
                    while page <= last or window:
                        while page <= last and len(window) < workers * 2:
                            window.append(pool.submit(full_page, page, last))
                            page += 1
                        rows, _ = window.popleft().result()
                        yield from fresh(rows)
                finally:
                    for f in window:
                        f.cancel()
            if len(rows) < per_page:
                return
            # rows added after page 1 was read land past the old last page
        while True:
            rows, meta = full_page(page, last) if last and page < last else self.read_page(service, alias, page,
                                                                                            per_page)
            yield from fresh(rows)
            last = meta.get("last_page") or last
            if not rows or len(rows) < per_page or (last and page >= last):
                return
            page += 1

    def insert(self, service: str, alias: str, data: dict, *, match_on: str | None = "uuid",
               facility_id: int | None = None) -> requests.Response:
        self.token()
        body = {"action": "insert", "model": alias, "destination_tenant_id": self.organization_id, "data": data}
        if self.platform or facility_id is not None:
            # a platform login's active facility is its own: the target facility is always named
            body["destination_facility_id"] = facility_id if facility_id is not None else self.facility_id
        if match_on and data.get(match_on):
            body["match_on"] = match_on
        return self.post(service, body, timeout=120)
