# patient_discovery

Finds warehouse objects that could become patient-count sources, and says why the rest can't. It's separate from
the counting procedure (`AFYA_REFRESH_PATIENT_COUNTS`), which it never calls or changes. Nothing is activated
automatically: a person adds approved sources to `AFYA_PATIENT_COUNT_SOURCES`.

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
| `model/questions.py` | the question pack and the recording of answers |
| `run.py` | the command line |
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
python -m patient_discovery.run answer --by "Your Name"
```

1. **`plan`** is metadata only. It writes `discovery_output/plan.json` (the next measurements with their exact SQL)
   and `discovery_output/questions.json` (the question pack). Drop `--cached-catalogue` to re-read the catalogue
   and lineage, which takes about 11 minutes.
2. **`compile`** EXPLAINs every measurement kind and the planned SQL. It reads no table data.
3. **`profile`** reads source data, aggregates only. It needs explicit approval and stays within the given schemas,
   budget and statement cap. Observations are cached in `discovery_output/observations.json`.
4. **`answer`** records the filled-in questions as statements, with who, when and on what basis.

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

`questions.json` groups everything that needs a person, so each fact is asked once:
- **`meaning`**: one question per facility and identifier column. It lists the member objects and the evidence for
  each. Set `answer` to `person`, `household`, `insurance_member`, `file`, `visit` or `other`. You can prune
  `members`, or set `column` where no candidate exists.
- **`scope`**: is the facility onboarded? Answer `yes` or `no`.
- **Other codes**: answer `not_a_source` to stop asking about those objects.

`answer` appends namespace, facility and not-a-source statements to `model/statements.json`. The next run uses them,
and the derivation rules carry each meaning to new versions, views and copies without asking again.

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
2. **Mutation gate:** re-introduces 24 known flaws; each one must fail the suite.
3. **Counting regression:** 24 characterisation tests of the refresh's counting semantics.

## Known limitations

- **Household register:** a column holding household numbers, with no finer candidate, is caught only because its
  meaning needs a statement. A person who confirms it wrongly will be believed.
- **Name-based kinds:** the kind of table still comes from its name. Registers and supporting tables are excluded
  by policy on that basis, which can miss a population; it never adds a false count.
- **Copies without evidence:** duplicates with no record identifier and no lineage can't be recognised from data.
- **Views:** they have no row counts, so their cost is estimated at 1,000,000 rows.
