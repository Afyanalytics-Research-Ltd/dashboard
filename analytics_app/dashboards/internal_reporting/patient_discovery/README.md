# patient_discovery

Finds warehouse objects that could become patient-count sources, and says why the rest can't. It's separate from
the counting procedure (`AFYA_REFRESH_PATIENT_COUNTS`), which it never calls or changes.

People answer its questions on the dashboard's **Discovery** tab. Once every answer a source needs is in (what its
identifier means, and that its facility is in scope), the source is registered in `AFYA_PATIENT_COUNT_SOURCES` at
once. Registration is insert-only, with provenance, and never changes or removes existing rows. The source is counted
from the next refresh.

## Where things are stored

All three tables are in `HOSPITALS.ANALYTICS`:

| Table | What |
|---|---|
| `AFYA_PATIENT_DISCOVERY_QUESTIONS` | the current question pack; every `plan` run replaces it |
| `AFYA_PATIENT_DISCOVERY_STATEMENTS` | every answer, append-only, with who, when and on what basis |
| `AFYA_PATIENT_COUNT_SOURCES` | the counting registry; discovery only inserts rows that aren't there yet |

Discovery writes only through `model/store.py`, whose guard allows writes to the two discovery tables and the
insert-if-absent into the registry, and nothing else.

## The rule it enforces

What an identifier column denotes (person, household, insurance member, file ...) comes only from a **namespace
statement** in `model/statements.json`: its meaning, its member columns and its facility scope. That meaning
transfers only along derivation inside that scope:
- **record alignment:** shared records carry the same identifier value, and the values keep the namespace's formats;
- **direct lineage:** the object reads a member directly and holds the same values.

Statistics, column names, layouts, tables named like registers and value overlaps nominate and relate candidates;
they never establish meaning. A dataset is `auto` only when its identifier means `person` and it's eligible (scope,
event classes, decisions D2/D3/D6/D7 in `config.py`).

## Layout

| Path | What |
|---|---|
| `model/concepts.py` | the concepts and question codes |
| `model/measure.py` | aggregate measurements (SQLite and Snowflake dialects) |
| `model/interpret.py` | a pure function: catalogue + observations + statements -> beliefs, questions, gaps |
| `model/planner.py` | measures the missing evidence within limits |
| `model/snowflake.py` | Snowflake access: metadata always; source data only for an approved profile run |
| `model/statements.py`, `model/statements.json` | statements from the registry, `config.py` and people |
| `model/questions.py` | the question pack, what each answer would do, and which sources are due for registration |
| `model/store.py` | the discovery tables, the write guard and the registry insert |
| `run.py` | the command line |
| `../afya_metrics/discovery.py` | the dashboard's Discovery tab |
| `classify.py`, `profile_sql.py`, `config.py` | shared naming, normalisation and decisions |

## Running

From `analytics_app/dashboards/internal_reporting`. Use a Python interpreter your machine's policy allows.

```bash
python -m patient_discovery.run plan --env C:/dev/afya/dashboard/.env --cached-catalogue
```

```bash
python -m patient_discovery.run compile --env C:/dev/afya/dashboard/.env --cached-catalogue
```

```bash
python -m patient_discovery.run profile --env C:/dev/afya/dashboard/.env --cached-catalogue --schemas S1,S2 --budget 1000000 --max-statements 120 --approved
```

```bash
python -m patient_discovery.run answer --by "Your Name" --env C:/dev/afya/dashboard/.env
```

1. **`plan`** is metadata only. It writes `discovery_output/plan.json` (the next measurements with their exact SQL)
   and `discovery_output/questions.json` (the question pack), and publishes the pack to
   `AFYA_PATIENT_DISCOVERY_QUESTIONS` for the Discovery tab. Drop `--cached-catalogue` to re-read the catalogue
   and lineage, which takes about 11 minutes.
2. **`compile`** EXPLAINs every measurement kind and the planned SQL. It reads no table data.
3. **`profile`** reads source data, aggregates only. It needs explicit approval and stays within the given schemas,
   budget and statement cap. Observations are cached in `discovery_output/observations.json`.
4. **`answer`** records the answers filled in to `questions.json`, with who, when and on what basis. It is the
   command-line alternative to the Discovery tab. With `--env`, answers go to `AFYA_PATIENT_DISCOVERY_STATEMENTS`
   and due sources are registered, as on the dashboard; without it, they are appended to `model/statements.json`.

Safety:
- **Session:** role DATAANALYSTS without secondary roles, a 300 s statement timeout, and a query tag.
- **SQL guard:** single aggregate SELECTs only; no `*`, no writes.
- **Personal columns:** names, phones, dates of birth and national IDs are never hashed or profiled; free-text
  columns are never grouped.
- **Caching:** a cached observation is invalid once its object, the raw store a view reads, or the measurement
  version changes. Observations involving a view expire after 28 days.
- **Failures:** a failing measurement is recorded with its values masked, and the rest of the run is kept.
- **Legacy proposals:** the old in-database discovery was retired on 2026-11-09 (procedure dropped, refresh no
  longer calls it). Registry rows it left with status `proposed` are reported and never used.

## Answering questions

Normally on the Discovery tab: enter your name, pick an answer, and record it. Each identifier question shows what
answering "Individual patients" would do for every table, and the original evidence sits under "Technical details".
A question closes once it is answered, or once every table it covers has been set aside as not a patient source.

The pack (on the tab, or in `questions.json` for the command line) groups everything that needs a person, so each
fact is asked once:
- **`meaning`**: one question per facility and identifier column. It lists the member objects and the evidence for
  each. Set `answer` to `person`, `household`, `insurance_member`, `file`, `visit` or `other`. You can prune
  `members`, or set `column` where no candidate exists.
- **`scope`**: is the facility onboarded? Answer `yes` or `no`.
- **Other codes**: answer `not_a_source` to stop asking about those objects.

Answers become namespace, facility and not-a-source statements. The next run uses them, and the derivation rules
carry each meaning to new versions, views and copies without asking again.

## Tests

```bash
python -m unittest discover -s patient_discovery/tests -t .
```

```bash
python -m patient_discovery.tests.mutation_gate
```

```bash
cd patient_counts_regression && python -I test_counting_reference.py
```

1. **Unit tests:** synthetic worlds only.
   - `test_model.py`: scenarios, a leave-one-out replay of the 13 sources with pre-registered expected reasons, and
     the older case catalogues run through the model.
   - `test_operations.py`: the question pack, answers, privacy, cache, failures and the SQL guard.
   - `test_rules.py`: shared naming and normalisation.
2. **Mutation gate:** re-introduces 31 known flaws; each one must fail the suite. It takes about 20 minutes.
3. **Counting regression:** 24 characterisation tests of the refresh's counting semantics.

## Known limitations

- **Household register:** a column holding household numbers, with no finer candidate, is caught only because its
  meaning needs a statement. A person who confirms it wrongly will be believed.
- **Name-based kinds:** the kind of table still comes from its name. Registers and supporting tables are excluded
  by policy on that basis, which can miss a population; it never adds a false count.
- **Copies without evidence:** duplicates with no record identifier and no lineage can't be recognised from data.
- **Views:** they have no row counts, so their cost is estimated at 1,000,000 rows.
