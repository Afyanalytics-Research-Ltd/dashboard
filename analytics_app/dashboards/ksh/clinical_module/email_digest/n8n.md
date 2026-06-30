# n8n WORKFLOW — MONTHLY CLINICAL DIGEST
Afya Clinical Analytics · Kisumu Specialists

---

## What you will learn from building this

This workflow teaches you four core n8n concepts:
1. Scheduled triggers — how to run a workflow automatically on a schedule
2. HTTP requests — how to call an external API (Groq) from n8n
3. Data transformation — how to reshape Snowflake output into a prompt
4. Email sending — how to assemble and send a plain text email

---

## Prerequisites

Before building in n8n you need:
- n8n running (cloud at n8n.io free tier, or self-hosted)
- Snowflake credentials (account, username, password, warehouse, database)
- Groq API key (free at console.groq.com — takes 2 minutes)
- Gmail or SMTP credentials for sending email

---

## Workflow overview — 7 nodes in sequence

```
[1] Schedule Trigger
        ↓
[2] Snowflake — Run digest query
        ↓
[3] Code — Reshape data + compute deltas
        ↓
[4] HTTP Request — Call Groq API
        ↓
[5] Code — Extract Groq recommendations
        ↓
[6] Code — Assemble email body
        ↓
[7] Send Email
```

A second branch runs daily for alerts:

```
[1b] Schedule Trigger (daily)
        ↓
[2b] Snowflake — Check alert conditions
        ↓
[3b] IF — Is alert condition true?
        ↓ (yes)
[4b] Send Email (alert)
```

---

## Node 1 — Schedule Trigger

**Type:** Schedule Trigger
**Settings:**
- Trigger interval: Month
- Day of month: 1
- Hour: 8
- Minute: 0
- Timezone: Africa/Nairobi

**What this does:** Wakes up the workflow on the 1st of every month at 08:00 EAT.
Nothing runs until this fires.

---

## Node 2 — Snowflake: Run digest query

**Type:** Snowflake node
**Operation:** Execute Query
**Credentials:** Your Snowflake credentials
**Query:** Paste the full contents of `digest_monthly_summary.sql`

**What this does:** Runs against your Snowflake warehouse and returns two rows —
last month and the month before. Each row has all the metrics the digest needs.

**Output:** Array of two objects, one per row. n8n stores this as
`$json` in subsequent nodes.

---

## Node 3 — Code: Reshape data and compute deltas

**Type:** Code node (JavaScript)
**What this does:** Takes the two Snowflake rows and computes the delta
(change) for each metric. Outputs a single clean object for the Groq prompt.

```javascript
// The Snowflake node returns an array of items
// Each item has a json property with the row data
const rows = $input.all();

// Find last month and prior month rows
const last = rows.find(r => r.json.PERIOD_LABEL === 'last_month')?.json;
const prior = rows.find(r => r.json.PERIOD_LABEL === 'prior_month')?.json;

if (!last || !prior) {
  throw new Error('Missing last_month or prior_month row from Snowflake');
}

// Compute deltas — positive means improvement, negative means decline
const convDelta = (last.CONVERSION_RATE_PCT - prior.CONVERSION_RATE_PCT).toFixed(2);
const retDelta  = (last.RETENTION_RATE_PCT  - prior.RETENTION_RATE_PCT ).toFixed(2);
const workDelta = (last.AVG_VISITS_PER_CLINICIAN - prior.AVG_VISITS_PER_CLINICIAN).toFixed(1);

// Format month label e.g. "April 2026"
const monthLabel = new Date(last.REPORT_MONTH)
  .toLocaleDateString('en-GB', { month: 'long', year: 'numeric' });

const priorMonthLabel = new Date(prior.REPORT_MONTH)
  .toLocaleDateString('en-GB', { month: 'long', year: 'numeric' });

// Direction arrows for email
const arrow = (delta) => parseFloat(delta) > 0 ? '↑' : parseFloat(delta) < 0 ? '↓' : '→';

return [{
  json: {
    // Labels
    last_month_label:              monthLabel,
    prior_month_label:             priorMonthLabel,

    // Last month values
    total_opd_visits:              last.TOTAL_OPD_VISITS,
    total_ipd_admissions:          last.TOTAL_IPD_ADMISSIONS,
    conversion_rate_pct:           last.CONVERSION_RATE_PCT,
    retention_rate_pct:            last.RETENTION_RATE_PCT,
    retention_universe_visits:     last.RETENTION_UNIVERSE_VISITS,
    comorbid_rate_pct:             last.COMORBID_RATE_PCT,
    single_dx_rate_pct:            last.SINGLE_DX_RATE_PCT,
    avg_visits_per_clinician:      last.AVG_VISITS_PER_CLINICIAN,
    active_clinicians:             last.ACTIVE_CLINICIANS,
    wait_time_gap_mins:            last.WAIT_TIME_GAP_MINS,
    strain_signal:                 last.STRAIN_SIGNAL,
    total_escalations:             last.TOTAL_ESCALATIONS,

    // Prior month values
    prior_conversion_rate_pct:     prior.CONVERSION_RATE_PCT,
    prior_retention_rate_pct:      prior.RETENTION_RATE_PCT,
    prior_avg_visits_per_clinician: prior.AVG_VISITS_PER_CLINICIAN,

    // Deltas
    conversion_delta:  `${arrow(convDelta)} ${Math.abs(convDelta)}`,
    retention_delta:   `${arrow(retDelta)}  ${Math.abs(retDelta)}`,
    workload_delta:    `${arrow(workDelta)} ${Math.abs(workDelta)}`,

    // Alert flags — used by daily alert branch
    alert_strain:      last.STRAIN_SIGNAL !== 'AS_EXPECTED',
    alert_comorbid:    last.COMORBID_RATE_PCT < last.SINGLE_DX_RATE_PCT,
    alert_escalation:  last.TOTAL_ESCALATIONS > 15,
  }
}];
```

---

## Node 4 — HTTP Request: Call Groq API

**Type:** HTTP Request node
**Method:** POST
**URL:** `https://api.groq.com/openai/v1/chat/completions`

**Headers:**
```
Authorization: Bearer YOUR_GROQ_API_KEY
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "model": "llama3-70b-8192",
  "max_tokens": 500,
  "temperature": 0.3,
  "messages": [
    {
      "role": "system",
      "content": "You are a clinical analytics advisor for a private hospital in Kenya. Write concise, actionable clinical recommendations in plain English. Output bullet points only."
    },
    {
      "role": "user",
      "content": "={{ $json.groq_prompt }}"
    }
  ]
}
```

**What this does:** Sends the structured data to Groq's Llama 3 70B model.
Temperature 0.3 keeps the output consistent and factual rather than creative.

**Note on the prompt variable:** In Node 3, add a `groq_prompt` field that
assembles the prompt text from `digest_groq_prompt.txt` with the actual
values substituted in. Add this to the return object in Node 3:

```javascript
groq_prompt: `You are a clinical analytics advisor for Kisumu Specialists Hospital in Kenya, a Level 4 private hospital. You are writing the recommendations section of a monthly clinical digest for the Head of Clinician.

Your job is to read the structured data below and write 4-5 bullet point recommendations in plain clinical language. Each bullet should state what the data shows (one sentence, specific numbers) and what clinical action to take (one sentence, specific and actionable). Output bullet points only, starting each with a bullet symbol.

DATA FOR ${monthLabel}:
Overall conversion rate: ${last.CONVERSION_RATE_PCT}% (prior month: ${prior.CONVERSION_RATE_PCT}%)
Retention universe rate: ${last.RETENTION_RATE_PCT}%
Comorbid patient rate: ${last.COMORBID_RATE_PCT}% vs single diagnosis: ${last.SINGLE_DX_RATE_PCT}%
Avg visits per clinician: ${last.AVG_VISITS_PER_CLINICIAN} (prior: ${prior.AVG_VISITS_PER_CLINICIAN})
Wait time gap (admitted vs not admitted): ${last.WAIT_TIME_GAP_MINS} minutes
Strain signal: ${last.STRAIN_SIGNAL}
72-hour escalations: ${last.TOTAL_ESCALATIONS}

CLINICAL CONTEXT:
Retention universe = patients with chronic, oncology, maternal, or mental health conditions — most likely to need inpatient care.
Comorbid patients have 2+ concurrent diagnoses and should convert at higher rates than single-diagnosis patients.
Wait time gap near zero or positive means triage prioritisation is breaking down.
Strain signal HIGH_STRAIN = clinician workload above average AND wait gap collapsed.
72-hour escalations = patients admitted within 72 hours of an OPD visit — likely missed admission decisions.`
```

---

## Node 5 — Code: Extract Groq recommendations

**Type:** Code node (JavaScript)
**What this does:** Pulls the text out of the Groq API response.

```javascript
const response = $input.first().json;
const recommendations = response.choices[0].message.content;
return [{ json: { recommendations } }];
```

---

## Node 6 — Code: Assemble email body

**Type:** Code node (JavaScript)
**What this does:** Builds the plain text email from all the data.

```javascript
const d = $('Code — Reshape data').first().json;
const recs = $input.first().json.recommendations;

const divider = '─'.repeat(45);

const body = `AFYA CLINICAL ANALYTICS — MONTHLY DIGEST
Kisumu Specialists · ${d.last_month_label} · Head of Clinician

${divider}
CONVERSION RATE SUMMARY
${divider}
Overall rate this month:       ${d.conversion_rate_pct}%
vs ${d.prior_month_label}:     ${d.prior_conversion_rate_pct}%  ${d.conversion_delta}pp
Retention universe rate:       ${d.retention_rate_pct}%  ${d.retention_delta}pp vs prior
Strain signal:                 ${d.strain_signal}

${divider}
CLINICAL SIGNALS
${divider}
OPD visits this month:         ${d.total_opd_visits.toLocaleString()}
IPD admissions this month:     ${d.total_ipd_admissions.toLocaleString()}
Comorbid patient rate:         ${d.comorbid_rate_pct}%  (single dx: ${d.single_dx_rate_pct}%)
Active clinicians:             ${d.active_clinicians}
Avg visits per clinician:      ${d.avg_visits_per_clinician}  ${d.workload_delta} vs prior
Wait time gap:                 ${d.wait_time_gap_mins} min (admitted vs not admitted)
72-hour escalations:           ${d.total_escalations}

${divider}
AI RECOMMENDATIONS
${divider}
${recs}

${divider}
Data as of: ${d.last_month_label} | Source: Kisumu Specialists Snowflake
This digest is generated automatically. Reply to your analytics team with questions.`;

return [{ json: { body, subject: `Clinical Digest — ${d.last_month_label} | Kisumu Specialists` } }];
```

---

## Node 7 — Send Email

**Type:** Gmail or Send Email node
**To:** head.of.clinician@kisumu-specialists.co.ke
**Subject:** `={{ $json.subject }}`
**Body:** `={{ $json.body }}`
**Body Content Type:** Text

---

## Daily alert branch — Node 1b to 4b

### Node 1b — Schedule Trigger (daily)
- Trigger interval: Day
- Hour: 7
- Minute: 30
- Timezone: Africa/Nairobi

### Node 2b — Snowflake: Check alert conditions
Run a lightweight version of the digest query for the current month to date:
```sql
SELECT
    ROUND(DIV0(
        COUNT(DISTINCT CASE WHEN a.visit_id IS NOT NULL THEN v.id END),
        COUNT(DISTINCT v.id)
    ) * 100.0, 2) AS conversion_rate_mtd,
    COUNT(DISTINCT CASE
        WHEN DATEDIFF('hour', v.created_at, a.admitted_at) BETWEEN 0 AND 72
        THEN a.visit_id END) AS escalations_mtd
FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS v
LEFT JOIN HOSPITALS.STAGING.STG_INPATIENT_ADMISSIONS a
    ON v.id = a.visit_id
    AND v.source_schema = LOWER(REPLACE(a.source_schema, '_CLEAN', ''))
WHERE v.source_schema = 'kisumu'
  AND DATE_TRUNC('month', v.created_at) = DATE_TRUNC('month', CURRENT_DATE)
```

### Node 3b — IF: Alert condition
**Condition (JavaScript expression):**
```
{{ $json.CONVERSION_RATE_MTD < 4.5 || $json.ESCALATIONS_MTD > 15 }}
```

Thresholds:
- Conversion rate below 4.5% month-to-date → alert
- More than 15 escalations this month → alert

### Node 4b — Send Email (alert)
**Subject:** `⚠ Clinical Alert — Kisumu Specialists`
**Body:**
```
A clinical signal has been detected at Kisumu Specialists that requires your attention.

Conversion rate (month to date): {{ $json.CONVERSION_RATE_MTD }}%
72-hour escalations (month to date): {{ $json.ESCALATIONS_MTD }}

Please review the clinical dashboard for detail.

Source: Kisumu Specialists Snowflake — automated alert
```

---

## How to test before going live

1. In n8n, use the "Execute node" button to run each node individually
   and inspect its output before connecting to the next
2. Test the Snowflake node first — confirm two rows come back with the
   correct column names
3. Test the Code node — check the reshaped object has all fields populated
4. Test the HTTP Request node — paste the Groq API key and send a
   test prompt. Confirm the response has choices[0].message.content
5. Test the email node — send to your own address first before setting
   the Head of Clinician as recipient
6. Once all nodes pass individually, run the full workflow end to end
   using the "Test workflow" button

---

## Cost

- n8n cloud free tier: 5,000 workflow executions per month — sufficient
  for one monthly digest + daily alert checks
- Groq free tier: 14,400 requests per day, 30 requests per minute —
  more than sufficient
- Gmail: free
- Total cost: KES 0