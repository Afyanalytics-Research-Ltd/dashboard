// Cube follows the platform warehouse switch (System Settings -> Data
// warehouse) without a restart:
//
//  * The models are written in Snowflake SQL, so Cube always generates
//    Snowflake-dialect queries (dbType 'snowflake').
//  * Snowflake: the stock Snowflake driver.
//  * Databend: AfyaDatabendDriver below sends each generated query to the
//    dashboard's translate endpoint (the same Snowflake->Databend translator
//    the pipelines use) and runs the result on Databend's HTTP API.
//  * The active warehouse is polled from the dashboard every 15 s; Cube keeps
//    one compiled app + query orchestrator per warehouse (contextToAppId /
//    contextToOrchestratorId), so caches and connections never mix.
//
// Env: AFYA_INTERNAL_API_TOKEN, CUBE_AFYA_STATUS_URL (default
// http://web:8000/core/api/warehouse/translate/), DATABEND_HTTP_URL
// (default http://databend:8000), DATABEND_USER/PASSWORD,
// CUBEJS_WAREHOUSE_BACKEND (initial value before the first poll).
const { BaseDriver } = require('@cubejs-backend/base-driver');

const TRANSLATE_URL = process.env.CUBE_AFYA_STATUS_URL || 'http://web:8000/core/api/warehouse/translate/';
const TOKEN = process.env.AFYA_INTERNAL_API_TOKEN || '';
const DATABEND_URL = (process.env.CUBE_DATABEND_HTTP_URL || 'http://databend:8000').replace(/\/$/, '');
const DATABEND_AUTH = 'Basic ' + Buffer.from(
  `${process.env.DATABEND_USER || 'databend'}:${process.env.DATABEND_PASSWORD || ''}`).toString('base64');

let backend = (process.env.CUBEJS_WAREHOUSE_BACKEND || 'snowflake').toLowerCase();

async function pollBackend() {
  if (!TOKEN) return;
  try {
    const r = await fetch(TRANSLATE_URL, { headers: { 'X-Afya-Token': TOKEN } });
    if (r.ok) {
      const j = await r.json();
      if (j.warehouse_backend && j.warehouse_backend !== backend) {
        console.log(`[afya] warehouse switched: ${backend} -> ${j.warehouse_backend}`);
        backend = j.warehouse_backend;
      }
    }
  } catch (e) {
    // dashboard unreachable: keep the last known warehouse
  }
}
pollBackend();
setInterval(pollBackend, 15000).unref();

// Bind Cube's '?' parameters as SQL literals (outside quotes only).
function bindValues(sql, values) {
  if (!values || !values.length) return sql;
  let out = '';
  let i = 0;
  let quote = null;
  for (let p = 0; p < sql.length; p++) {
    const c = sql[p];
    if (quote) {
      out += c;
      if (c === quote) quote = null;
      continue;
    }
    if (c === "'" || c === '"') { quote = c; out += c; continue; }
    if (c === '?' && i < values.length) {
      const v = values[i++];
      if (v === null || v === undefined) out += 'NULL';
      else if (typeof v === 'number' || typeof v === 'bigint') out += String(v);
      else if (typeof v === 'boolean') out += v ? 'TRUE' : 'FALSE';
      else if (v instanceof Date) out += `'${v.toISOString().replace('T', ' ').replace('Z', '')}'`;
      else out += `'${String(v).replace(/\\/g, '\\\\').replace(/'/g, "''")}'`;
      continue;
    }
    out += c;
  }
  return out;
}

class AfyaDatabendDriver extends BaseDriver {
  readOnly() { return true; }

  async testConnection() { await this.runDatabend('SELECT 1'); }

  async translate(sql) {
    const r = await fetch(TRANSLATE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Afya-Token': TOKEN },
      body: JSON.stringify({ sql, database: process.env.CUBEJS_DB_NAME || 'HOSPITALS',
                             schema: process.env.CUBEJS_DB_SCHEMA || null }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(`Databend translation failed: ${j.error || r.status}`);
    return j.statements || [];
  }

  async runDatabend(sql) {
    const headers = { 'Content-Type': 'application/json', Authorization: DATABEND_AUTH };
    let resp = await fetch(`${DATABEND_URL}/v1/query`, {
      method: 'POST', headers,
      body: JSON.stringify({ sql, session: { settings: { timezone: 'UTC', date_format_style: 'Oracle' } },
                             pagination: { wait_time_secs: 30, max_rows_per_page: 50000 } }),
    }).then((r) => r.json());
    let schema = null;
    const rows = [];
    for (;;) {
      if (resp.error) throw new Error(`Databend: ${resp.error.message}`);
      if (resp.schema && !schema) schema = resp.schema;
      if (resp.data) rows.push(...resp.data);
      const next = resp.next_uri;
      if (!next) break;
      const body = await fetch(`${DATABEND_URL}${next}`, { headers }).then((r) => r.json());
      if (next.endsWith('/final')) { if (body.error) throw new Error(`Databend: ${body.error.message}`); break; }
      resp = body;
    }
    const names = (schema || []).map((c) => c.name);
    return rows.map((r) => Object.fromEntries(names.map((n, k) => [n, r[k]])));
  }

  async query(query, values) {
    let rows = [];
    for (const stmt of await this.translate(bindValues(query, values))) {
      rows = await this.runDatabend(stmt);
    }
    return rows;
  }

  async tablesSchema() {
    const rows = await this.runDatabend(
      "SELECT database, \"table\", name, data_type FROM system.columns WHERE database NOT IN " +
      "('system','information_schema','AFYA_SF_COMPAT','default')");
    const out = {};
    for (const r of rows) {
      out[r.database] = out[r.database] || {};
      out[r.database][r.table] = out[r.database][r.table] || [];
      out[r.database][r.table].push({ name: r.name, type: r.data_type, attributes: [] });
    }
    return out;
  }
}

module.exports = {
  // Separate compiled app / orchestrator (caches, connection pools) per warehouse.
  contextToAppId: () => `CUBE_APP_${backend}`,
  contextToOrchestratorId: () => `CUBE_ORCH_${backend}`,
  dbType: () => 'snowflake',
  driverFactory: () => {
    if (backend === 'databend') return new AfyaDatabendDriver();
    // Snowflake exactly as before this file was mounted: configured from the
    // CUBEJS_DB_* / CUBEJS_DB_SNOWFLAKE_* environment variables.
    return { type: 'snowflake' };
  },
};
