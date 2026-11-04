# Afya Analytics Data Pipelines: Airflow REST API Integration Guide

This guide shows how to connect your system to our Airflow server. Once connected you can list pipelines (DAGs), trigger runs, check their status, and read task logs.

| | |
|---|---|
| **Base URL** | `https://datapipelines.afyaanalytics.com` |
| **API prefix** | `/api/v2` |
| **Airflow version** | 3.1.7 (stable REST API v2) |
| **Auth** | JWT bearer token, obtained with username + password |
| **Format** | JSON request and response bodies |
| **Interactive docs** | https://datapipelines.afyaanalytics.com/docs (Swagger UI) |
| **OpenAPI spec** | https://datapipelines.afyaanalytics.com/openapi.json |

> We send credentials separately (not in this document). Keep them in environment variables or a secrets manager, never in source code.

---

## 1. Authentication

Airflow 3 does **not** accept HTTP Basic auth on the API (`curl -u user:pass` returns `401`). Instead you:

1. `POST` your username and password to `/auth/token` to get an access token.
2. Send that token in the `Authorization: Bearer <token>` header on every API call.

### 1.1 Get a token

```bash
curl -X POST "https://datapipelines.afyaanalytics.com/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "<USERNAME>", "password": "<PASSWORD>"}'
```

Response (`201 Created`):

```json
{ "access_token": "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9...." }
```

### 1.2 Use the token

```bash
curl "https://datapipelines.afyaanalytics.com/api/v2/dags" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

### 1.3 Token lifetime

- Tokens are valid for **24 hours** (the `exp` claim in the JWT).
- There is no refresh endpoint. When a call returns `401` or `403`, request a new token and retry once.
- Cache the token and reuse it. Don't log in again before every request.

### 1.4 Auth errors

| Situation | Status | Body |
|---|---|---|
| Wrong username/password on `/auth/token` | `401` | `{"detail":"Invalid credentials"}` |
| No `Authorization` header | `401` | `{"detail":"Not authenticated"}` |
| Invalid or expired token | `403` | `{"detail":"Invalid JWT token"}` |

---

## 2. Key concepts

| Term | Meaning |
|---|---|
| **DAG** | A pipeline, identified by `dag_id` (e.g. `flatten_jsons_schemas`). |
| **DAG run** | One execution of a DAG, identified by `dag_run_id` (e.g. `manual__2026-10-08T05:30:11.137146+00:00`). |
| **Task instance** | One task inside a DAG run, identified by `task_id` (+ `map_index` for mapped tasks, `-1` when not mapped). |
| **conf** | A JSON object you pass when triggering a run. The DAG reads its parameters from it. |
| **States** | Run: `queued`, `running`, `success`, `failed`. Task: also `scheduled`, `up_for_retry`, `upstream_failed`, `skipped`, etc. |

> **URL-encode run IDs.** Run IDs contain `:` and `+`, so encode them in paths:
> `scheduled__2026-10-21T00:00:00+00:00` → `scheduled__2026-10-21T00%3A00%3A00%2B00%3A00`.
> If you send a raw `+`, the server reads it as a space and returns `404`.

---

## 3. Common endpoints

All paths below are relative to `https://datapipelines.afyaanalytics.com`.

### 3.1 Health check (no auth needed)

```
GET /api/v2/monitor/health
```

```json
{
  "metadatabase": {"status": "healthy"},
  "scheduler":    {"status": "healthy", "latest_scheduler_heartbeat": "2026-10-21T12:13:54Z"},
  "triggerer":    {"status": "healthy", "latest_triggerer_heartbeat": "2026-10-21T12:13:58Z"},
  "dag_processor":{"status": "healthy", "latest_dag_processor_heartbeat": "2026-10-21T12:13:56Z"}
}
```

### 3.2 List DAGs

```
GET /api/v2/dags?limit=50&offset=0
```

Useful query parameters: `limit`, `offset`, `dag_id_pattern` (SQL LIKE, e.g. `%snowflake%`), `tags`, `paused` (`true`/`false`), `last_dag_run_state`, `order_by` (e.g. `dag_id`, `-last_run_start_date`).

```json
{
  "dags": [
    {
      "dag_id": "api_incremental_to_snowflake",
      "dag_display_name": "api_incremental_to_snowflake",
      "is_paused": true,
      "timetable_summary": "0 * * * *",
      "next_dagrun_run_after": null,
      "tags": [],
      "owners": ["airflow"],
      "has_import_errors": false
    }
  ],
  "total_entries": 44
}
```

### 3.3 Get one DAG

```
GET /api/v2/dags/{dag_id}
GET /api/v2/dags/{dag_id}/details     # more fields, incl. params
GET /api/v2/dags/{dag_id}/tasks       # task list
```

### 3.4 Pause / unpause a DAG

```
PATCH /api/v2/dags/{dag_id}
Content-Type: application/json

{ "is_paused": false }
```

### 3.5 Trigger a DAG run

```
POST /api/v2/dags/{dag_id}/dagRuns
Content-Type: application/json

{
  "logical_date": null,
  "conf": { "facilities": ["kisumu_v3"], "mode": "plan" },
  "note": "Triggered from <your system name>"
}
```

- `logical_date` is **required** in the body. Send `null` to mean "now".
- `dag_run_id` is optional. Set your own unique ID (e.g. `ext__<your-request-id>`) to make retries idempotent and runs easy to find later. A duplicate ID returns `409 Conflict`.
- `conf` holds the DAG's parameters. Ask us for the keys each DAG expects.
- The DAG must be unpaused for the run to start. A run on a paused DAG stays `queued`.

Response (`200`) is the created DAG run. Keep its `dag_run_id`:

```json
{
  "dag_run_id": "manual__2026-10-21T12:30:00.123456+00:00",
  "dag_id": "snowflake_to_v3_migration",
  "state": "queued",
  "run_type": "manual",
  "conf": { "facilities": ["kisumu_v3"], "mode": "plan" },
  "queued_at": "2026-10-21T12:30:00.150000Z",
  "start_date": null,
  "end_date": null
}
```

### 3.6 Check a run's status

```
GET /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}
```

Poll this every 15–60 s until `state` is `success` or `failed`.

### 3.7 List runs

```
GET /api/v2/dags/{dag_id}/dagRuns?limit=20&order_by=-run_after
GET /api/v2/dags/~/dagRuns?state=failed&limit=20     # "~" = all DAGs
```

Filters: `state`, `run_type` (`manual`, `scheduled`), `start_date_gte`, `start_date_lte`, `end_date_gte`, `run_id_pattern`, `order_by`.

### 3.8 Task instances of a run

```
GET /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances
```

```json
{
  "task_instances": [
    {"task_id": "plan",    "state": "success", "try_number": 1, "map_index": -1, "duration": 12.4},
    {"task_id": "migrate", "state": "running", "try_number": 1, "map_index": -1, "duration": null}
  ],
  "total_entries": 2
}
```

### 3.9 Task logs

```
GET /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}
Accept: application/json
```

- For mapped tasks, add `?map_index=<n>`. Without it, the server returns `404 TaskInstance not found`.
- With `Accept: application/json` the response is `{"content": [ {timestamp, level, event, ...}, ... ], "continuation_token": ...}`.
- With `Accept: text/plain` you get plain text.

### 3.10 Re-run / stop a run

```
POST  /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/clear
      { "dry_run": false, "only_failed": true }         # re-run failed tasks

PATCH /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}
      { "state": "failed" }                             # mark run failed (stops it)
```

### 3.11 XCom (values returned by tasks)

```
GET /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries
GET /api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries/return_value
```

---

## 4. Pagination, errors, conventions

- **Pagination:** list endpoints take `limit` (default 50, server maximum 100) and `offset`, and return `total_entries`. Keep increasing `offset` until you have read `total_entries` items.
- **Sorting:** `order_by=<field>`; prefix the field with `-` for descending.
- **Dates:** ISO-8601 in UTC, e.g. `2026-10-21T00:00:00Z`.
- **Errors** always come back as `{"detail": ...}`:

| Status | Meaning |
|---|---|
| `400` / `422` | Bad request body or parameter (`detail` says which field) |
| `401` | Not authenticated / bad credentials |
| `403` | Invalid or expired token, or no permission |
| `404` | DAG / run / task not found (check URL-encoding of the run ID) |
| `409` | A DAG run with that `dag_run_id` already exists |
| `5xx` | Server problem. Retry with backoff and contact us if it persists. |

---

## 5. Integration examples

### 5.1 Python (`requests`)

```python
import os
import time
from urllib.parse import quote

import requests

BASE = "https://datapipelines.afyaanalytics.com"


class AirflowClient:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self._token = None

    def _login(self):
        r = requests.post(f"{BASE}/auth/token",
                          json={"username": self.username, "password": self.password},
                          timeout=30)
        r.raise_for_status()
        self._token = r.json()["access_token"]
        self.session.headers["Authorization"] = f"Bearer {self._token}"

    def request(self, method: str, path: str, **kwargs):
        if self._token is None:
            self._login()
        r = self.session.request(method, f"{BASE}/api/v2{path}", timeout=60, **kwargs)
        if r.status_code in (401, 403):          # token expired -> log in again once
            self._login()
            r = self.session.request(method, f"{BASE}/api/v2{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json() if r.content else None

    def list_dags(self):
        dags, offset = [], 0
        while True:
            page = self.request("GET", "/dags", params={"limit": 100, "offset": offset})
            dags += page["dags"]
            offset += len(page["dags"])
            if offset >= page["total_entries"] or not page["dags"]:
                return dags

    def trigger(self, dag_id: str, conf: dict, run_id: str | None = None):
        body = {"logical_date": None, "conf": conf}
        if run_id:
            body["dag_run_id"] = run_id
        return self.request("POST", f"/dags/{dag_id}/dagRuns", json=body)

    def get_run(self, dag_id: str, run_id: str):
        return self.request("GET", f"/dags/{dag_id}/dagRuns/{quote(run_id, safe='')}")

    def wait(self, dag_id: str, run_id: str, poll_seconds: int = 30):
        while True:
            run = self.get_run(dag_id, run_id)
            if run["state"] in ("success", "failed"):
                return run
            time.sleep(poll_seconds)


if __name__ == "__main__":
    client = AirflowClient(os.environ["AIRFLOW_USERNAME"], os.environ["AIRFLOW_PASSWORD"])
    for d in client.list_dags():
        print(d["dag_id"], "paused" if d["is_paused"] else "active")

    # run = client.trigger("my_dag_id", {"key": "value"})
    # print(client.wait("my_dag_id", run["dag_run_id"])["state"])
```

### 5.2 PHP (built-in cURL, no dependencies)

Works on PHP 8.0+ with the `curl` and `json` extensions (enabled by default on most hosts).

```php
<?php

class AirflowClient
{
    private string $base = 'https://datapipelines.afyaanalytics.com';
    private ?string $token = null;

    public function __construct(private string $username, private string $password) {}

    private function login(): void
    {
        [$status, $body] = $this->http('POST', '/auth/token', [
            'username' => $this->username,
            'password' => $this->password,
        ], false);
        if ($status !== 201 && $status !== 200) {
            throw new RuntimeException("Airflow login failed ($status): " . json_encode($body));
        }
        $this->token = $body['access_token'];   // valid 24h, reuse it
    }

    /** Low-level HTTP call. Returns [statusCode, decodedJson]. */
    private function http(string $method, string $path, ?array $json = null, bool $auth = true): array
    {
        $headers = ['Accept: application/json', 'Content-Type: application/json'];
        if ($auth) {
            $headers[] = 'Authorization: Bearer ' . $this->token;
        }
        $ch = curl_init($this->base . $path);
        curl_setopt_array($ch, [
            CURLOPT_CUSTOMREQUEST  => $method,
            CURLOPT_HTTPHEADER     => $headers,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 60,
        ]);
        if ($json !== null) {
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($json, JSON_UNESCAPED_SLASHES));
        }
        $raw = curl_exec($ch);
        if ($raw === false) {
            throw new RuntimeException('Airflow request failed: ' . curl_error($ch));
        }
        $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        return [$status, json_decode($raw, true)];
    }

    /** Authenticated API call under /api/v2, with automatic re-login on expired token. */
    public function request(string $method, string $path, ?array $json = null): ?array
    {
        if ($this->token === null) {
            $this->login();
        }
        [$status, $body] = $this->http($method, '/api/v2' . $path, $json);
        if ($status === 401 || $status === 403) {
            $this->login();
            [$status, $body] = $this->http($method, '/api/v2' . $path, $json);
        }
        if ($status >= 400) {
            throw new RuntimeException("Airflow $method $path failed ($status): " . json_encode($body));
        }
        return $body;
    }

    public function listDags(): array
    {
        $dags = [];
        $offset = 0;
        do {
            $page = $this->request('GET', "/dags?limit=100&offset=$offset");
            $dags = array_merge($dags, $page['dags']);
            $offset += count($page['dags']);
        } while ($page['dags'] && $offset < $page['total_entries']);
        return $dags;
    }

    public function trigger(string $dagId, array $conf = [], ?string $runId = null): array
    {
        $body = ['logical_date' => null, 'conf' => (object) $conf];  // (object) so {} not []
        if ($runId !== null) {
            $body['dag_run_id'] = $runId;
        }
        return $this->request('POST', '/dags/' . rawurlencode($dagId) . '/dagRuns', $body);
    }

    public function getRun(string $dagId, string $runId): array
    {
        // rawurlencode() is required: run ids contain ":" and "+"
        return $this->request('GET', '/dags/' . rawurlencode($dagId) . '/dagRuns/' . rawurlencode($runId));
    }

    public function waitForRun(string $dagId, string $runId, int $pollSeconds = 30): array
    {
        while (true) {
            $run = $this->getRun($dagId, $runId);
            if (in_array($run['state'], ['success', 'failed'], true)) {
                return $run;
            }
            sleep($pollSeconds);
        }
    }

    public function taskInstances(string $dagId, string $runId): array
    {
        return $this->request('GET', '/dags/' . rawurlencode($dagId) . '/dagRuns/'
            . rawurlencode($runId) . '/taskInstances')['task_instances'];
    }
}

// ---- usage ----
$client = new AirflowClient(getenv('AIRFLOW_USERNAME'), getenv('AIRFLOW_PASSWORD'));

foreach ($client->listDags() as $dag) {
    echo $dag['dag_id'] . ' ' . ($dag['is_paused'] ? 'paused' : 'active') . PHP_EOL;
}

// $run = $client->trigger('my_dag_id', ['key' => 'value'], 'ext__' . uniqid());
// $final = $client->waitForRun('my_dag_id', $run['dag_run_id']);
// echo $final['state'] . PHP_EOL;
```

> **PHP gotcha:** `json_encode([])` produces `[]`, but Airflow expects `conf` to be a JSON **object**. Always cast it: `(object) $conf`.

#### Laravel (`Http` facade)

```php
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Http;

$base = 'https://datapipelines.afyaanalytics.com';

$token = Cache::remember('airflow_token', now()->addHours(23), fn () =>
    Http::post("$base/auth/token", [
        'username' => config('services.airflow.username'),
        'password' => config('services.airflow.password'),
    ])->throw()->json('access_token')
);

$airflow = Http::withToken($token)->acceptJson()->baseUrl("$base/api/v2");

$dags = $airflow->get('/dags', ['limit' => 100])->throw()->json('dags');

$run = $airflow->post('/dags/my_dag_id/dagRuns', [
    'logical_date' => null,
    'conf'         => (object) ['key' => 'value'],
])->throw()->json();

$status = $airflow->get('/dags/my_dag_id/dagRuns/' . rawurlencode($run['dag_run_id']))
    ->throw()->json('state');
```

If a call returns `401`/`403`, run `Cache::forget('airflow_token')` and retry.

### 5.3 JavaScript / Node 18+ (`fetch`)

```javascript
const BASE = "https://datapipelines.afyaanalytics.com";
let token = null;

async function login() {
  const r = await fetch(`${BASE}/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: process.env.AIRFLOW_USERNAME,
      password: process.env.AIRFLOW_PASSWORD,
    }),
  });
  if (!r.ok) throw new Error(`Login failed: ${r.status}`);
  token = (await r.json()).access_token;
}

async function api(method, path, body) {
  if (!token) await login();
  const call = () =>
    fetch(`${BASE}/api/v2${path}`, {
      method,
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  let r = await call();
  if (r.status === 401 || r.status === 403) { await login(); r = await call(); }
  if (!r.ok) throw new Error(`${method} ${path} -> ${r.status}: ${await r.text()}`);
  return r.status === 204 ? null : r.json();
}

// Examples
const { dags, total_entries } = await api("GET", "/dags?limit=100");
const run = await api("POST", "/dags/my_dag_id/dagRuns", { logical_date: null, conf: {} });
const status = await api("GET",
  `/dags/my_dag_id/dagRuns/${encodeURIComponent(run.dag_run_id)}`);
```

### 5.4 cURL quick test

```bash
BASE=https://datapipelines.afyaanalytics.com
TOKEN=$(curl -s -X POST "$BASE/auth/token" -H "Content-Type: application/json" \
  -d "{\"username\":\"$AIRFLOW_USERNAME\",\"password\":\"$AIRFLOW_PASSWORD\"}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/v2/dags?limit=5"
```

---

## 6. Typical integration flow

```
1. POST /auth/token                         -> cache token (24h)
2. GET  /api/v2/dags                        -> discover available pipelines
3. POST /api/v2/dags/{dag_id}/dagRuns       -> trigger with conf, store dag_run_id
4. GET  /api/v2/dags/{dag_id}/dagRuns/{id}  -> poll until success/failed
5. GET  .../taskInstances (+ /logs)         -> on failure, show which task failed and why
```

## 7. Further reading

- Live Swagger UI for this server (every endpoint, try-it-out): https://datapipelines.afyaanalytics.com/docs
- Official Airflow REST API reference: https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html
- Airflow API authentication (JWT): https://airflow.apache.org/docs/apache-airflow/stable/security/api.html
