# Automation Studio (n8n)

A full alternative front end for n8n, served by the dashboard at **`/automations/`**
(superusers only). It talks to n8n's public API through a Django proxy, so the API key
never reaches the browser.

## Setup

1. In n8n open **Settings → n8n API** and create an API key with all scopes.
2. Add to `.env`:

   ```env
   N8N_API_KEY=...                         # required
   N8N_API_URL=http://n8n-container:5678   # server-side URL (add /n8n if N8N_PATH=/n8n/)
   N8N_PUBLIC_URL=https://example.com/n8n  # browser URL, for "open in n8n" links
   N8N_WEBHOOK_URL=https://example.com/n8n # shown on the webhook panel
   # optional: enables the Run button and the live node catalog (community nodes)
   N8N_STUDIO_EMAIL=automation-bot@example.com
   N8N_STUDIO_PASSWORD=...                 # a non-MFA n8n user
   ```

3. `python manage.py migrate n8n_studio`, then restart the dashboard.
4. Optional nightly backups: schedule `python manage.py n8n_snapshot`.

## What it covers

Every operation in n8n's 2.36.8 OpenAPI spec (114 operations, allowlisted in
`routes.py`) is used somewhere in the UI. Licence-gated features (projects, folders,
variables, SSO, LDAP, log streaming, the security policy) show a clear "needs licence"
panel instead of an error.

| Area | Highlights |
|---|---|
| Editor | Canvas with pan/zoom, minimap, box select, drag-to-connect, inserting or deleting on an edge, AI sub-node ports, stickies, auto-layout, copy/paste compatible with n8n's editor, undo/redo, browser draft recovery, lint panel, find-in-workflow |
| Node panel | Forms built from node schemas, credentials, expressions with sandboxed live preview, drag a field to map it, input/output data (table/JSON/schema/binary), pin/edit pinned data, node settings |
| Runs | Manual run with live canvas status (needs the login), execute up to a node, "debug in editor" (pins a past run's data), webhook tester with curl export |
| Versions | n8n history plus Studio snapshots (every save and every delete, no 24-hour limit), node-level diff with line diff for code, restore, preview |
| Workflows | Search inside nodes, parameters and credentials (`type:` `cred:` `tag:` `node:` `is:` `-neg`), facets, saved views, last-run health, bulk actions, import with credential remapping, `.n8np` packages |
| Executions | Cross-workflow explorer (time window, duration, mode), timeline chart, p50/p95, bulk retry/stop/delete/tag, inspector with waterfall timeline and comparison with another run |
| Failure inbox | Failures grouped by workflow, node and normalised error, with one-click retry |
| Resources | Credentials (usage map, unused/missing, typed forms, test, transfer), data-table spreadsheet (inline edit, filters, dry-run bulk update, upsert, BOM-safe CSV), variables, tags (with merge), projects/members/folders |
| Admin | Users and roles (custom roles, role-mapping rules), insights, security audit, community packages, log streaming, SAML/OIDC/LDAP/OpenTelemetry/security policy, source-control pull, backups and recycle bin, API explorer, system status |

## Tests

```bash
python manage.py test n8n_studio                   # backend (n8n mocked)
node --test n8n_studio/tests/js/*.test.mjs         # pure JS modules (workflow model, diff, lint, cron, csv, …)
```
