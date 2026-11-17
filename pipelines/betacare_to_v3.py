"""
betacare_to_v3.py — Beta Care Hospital's Smart Hospital HMS (PHP / MariaDB) → V3, through the model gateway.

Target (.env or Airflow connection afya_v3_betacare_prod): production org 96 BETA CARE HOSPITAL LTD / facility 94
(its only facility). The org already holds a template catalogue (13.9k procedures, 27k products), 8 wards and 3 stores
set up by staff, and 14 insurers: those are reused, nothing of theirs is changed.

Source: the Snowflake mirror HOSPITALS.BETACARE_HOSPITAL_HISTORICAL_DATA (load_betacare_to_snowflake.py, dump of
2026-11-17). Smart Hospital keys everything by integer ids (patients.id, opd_details.id, ipd_details.id …); each V3 row
carries a uuid5 of its source table + id and is posted with match_on=uuid, so a re-run updates instead of duplicating.

What links to what — Smart Hospital leaves most clinical rows without a visit:
  * prescriptions, OPD charges and IPD charges carry opd_id / ipd_id;
  * clinical notes (`diagnosis`), lab / radiology reports and pharmacy sales carry only the patient and a date. They go
    on the admission the date falls in, else the patient's latest OPD visit up to a few days before (VISIT_WINDOWS),
    else the first one a day after; with none of those they are skipped and counted.
  * a lab / radiology report whose test was also billed as an OPD charge (same patient + charge, 0–3 days before) puts
    its result on that charge's investigation instead of ordering the test a second time.

Steps (STEPS order):
  users                staff → core/users. A staff member whose name matches exactly one existing org-96 user's profile
                       is linked to that user; the rest get an inactive account "bch.<login>" (random password,
                       "...@betacare.migrated.invalid" so no mail goes out) + profile, so records keep their author.
  insurance            insurers named in `organisation`, on patients and in payment methods → core insurance companies
                       (matched to org 96's own), one scheme each; NHIF's variants (Capitation CS / NS, Edu-Afya, Fee for
                       Service, Linda Mama) are schemes of one NHIF company. Ones Beta Care no longer lists are inactive.
  patients / patient_insurance   patients → reception/patient (patient_no = Smart Hospital's number, BCH-<id> when it
                       is blank or shared); insured patients → patient_insurance.
  visits               opd_details → outpatient visits, ipd_details → inpatient visits (unique_id BCH-OPDN… / BCH-IPDN…).
  wards / admissions / ward_stays / discharges   bed groups → org 96's wards (by name; Isolation and Rehabilitation are
                       created, inactive); ipd_details → admissions + ward stays + discharges. An admission Smart Hospital
                       never discharged is closed unless it started in the last OPEN_ADMISSION_DAYS days.
  vitals               triage readings on opd_details / ipd_details → evaluation/vitals.
  doctor_notes / diagnoses   `diagnosis` (Smart Hospital's clinical notes: diagnosis + consultation text) → doctor note
                       + diagnosis; lab_request / rad_request (doctors' free-text requests) → doctor notes.
  catalogue / procedure_prices   charges (+ lab / radiology tests with no charge) → evaluation/procedures BCH-C<id>,
                       priced for facility 94: standard charge = cash, the usual insurer charge = insurance.
  investigations / lab_results   opd_patient_charges + patient_charges → investigations; pathology / radiology reports →
                       investigations (unless matched to a charge) + published investigation_results.
  products / stock     pharmacy → inventory/product BCH-I<id> (an existing org-96 product with exactly the same name is
                       reused); batches with stock left → batch + stock in org 96's Pharmacy store.
  prescriptions        prescription + ipd_prescription_details (doctors' orders; dispensed when the patient bought from
                       the pharmacy 0–2 days later) and pharmacy_bill_detail (what the pharmacy sold, dispensed, priced).
  payments / payment_details   invoicepayment → finance/evaluation_payment: the cash + M-Pesa part. What an insurer paid
                       (NHIF / second-insurer columns) is the insurer settling the invoice: on the invoice, not a receipt.
  invoices             invoices an insurer paid part of, and unpaid ones → finance/invoice (totals from master_bill).
  appointments         appointment → reception/appointment.

    python betacare_to_v3.py --target betacare_prod --steps users patients --limit 3 --dry-run
    python betacare_to_v3.py --target betacare_prod --patients 9289 9288      # canary on two patients
"""
from __future__ import annotations

import argparse
import bisect
import calendar
import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

sys.path.insert(0, "/opt/airflow/dags")
import eagle_to_v3 as _eagle  # noqa: E402
from eagle_to_v3 import TITLE_CODES, clean, gender, num, patient_login, person_name, phone, text_join, vital, words  # noqa: E402


def day(v) -> str | None:
    """eagle's day(), minus Smart Hospital's placeholder dates (1970-01-01, 1922-…): V3 timestamps start in 1970."""
    d = _eagle.day(v)
    return d if d and d >= "1971-01-01" else None


def stamp(d, t=None) -> str | None:
    s = _eagle.stamp(d, t)
    return s if s and s >= "1971-01-01" else None

log = logging.getLogger("betacare_to_v3")

SF = "HOSPITALS.BETACARE_HOSPITAL_HISTORICAL_DATA"
NS = uuid.UUID("3f6c9a1e-8b27-4d5c-a0e4-7c2b9d18f5a6")      # never change: rows are matched by it
TARGETS = ("betacare_prod",)
CHUNK = 20_000
EAT = timezone(timedelta(hours=3))                         # Smart Hospital's unix timestamps → Nairobi wall time
OPEN_ADMISSION_DAYS = 30
ADMISSION_WINDOW_DAYS = 30        # date-only rows go on an admission up to this long after it began (rehab: to discharge)
VISIT_WINDOWS = {"note": (7, 1), "lab": (3, 1), "sale": (2, 0), "charge": (3, 1)}   # (days before, days after)
FACILITY_MODELS = {"patient", "patient_insurance", "visits", "vitals", "doctor_notes", "investigations", "prescriptions",
                   "procedure_prices", "evaluation_payment", "invoice", "admission", "visit_ward", "discharge", "ward",
                   "stock", "batch", "product", "appointment"}
STEPS = ("users", "insurance", "patients", "patient_insurance", "visits", "wards", "admissions", "ward_stays",
         "discharges", "vitals", "doctor_notes", "diagnoses", "catalogue", "procedure_prices", "investigations",
         "lab_results", "products", "stock", "prescriptions", "payments", "payment_details", "invoices", "appointments")

# organisation ids patients still carry after Beta Care deleted the organisation: read off what those patients'
# bills were paid with (org 18: 336 'NHIF Capitation CS' + 81 'NHIF' …, org 21: 'Linda Mama')
DELETED_ORGS = {"1": "NHIF Capitation CS", "7": "NHIF Edu-Afya", "8": "NHIF", "18": "NHIF Capitation CS", "21": "Linda Mama"}
NOT_INSURERS = {"cash", "mpesa", "billing error", "credit billing error", "unable to pay"}
# bed group (Smart Hospital) → the ward staff already set up in V3 (alnum keys); the rest get a ward of their own
WARD_MAP = {"maleward": "maleward", "femalemedicalward": "femaleward", "hdu": "hdu", "nbu": "newbornunit",
            "paediatricward": "paediatricsward", "maternity": "generalmaternity", "privateward": "privaterooms",
            "privateward2": "privaterooms", "rehabilitation2": "rehabilitation"}
# inventory's uuid column is unindexed on prod: match_on=uuid takes >120 s per call against org 96's 27k products
# (nginx 502s / read timeouts, as neorao_to_v3 found) — these are matched through a uuid → id map read up front
NO_MATCH_ON = {"product", "batch", "stock"}
NO_ALLERGY = {"n a", "na", "none", "nil", "no", "0", "nan"}


# ── pure helpers (tests/test_betacare_to_v3.py) ──────────────────────────────
def gid(table: str, key) -> str:
    """Stable uuid of a source row (key = its Smart Hospital id)."""
    return str(uuid.uuid5(NS, f"betacare/{table}/{str(key).strip()}"))


def key(v) -> str:
    """Letters and digits only, lower-case: 'H.D.U' == 'HDU', 'Paediatrics Ward' == 'paediatricsward'."""
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def token_words(v) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(v or "").lower()).split())


def sid(v) -> str | None:
    """A Smart Hospital id as text ('0', '', None → None)."""
    n = num(v)
    return str(int(n)) if n else None


def method_insurers(method) -> list[str]:
    """'Cash, Mpesa, NHIF Capitation CS' → ['NHIF Capitation CS']: the payers in a payment method that are insurers."""
    return [p.strip() for p in (clean(method) or "").split(",") if p.strip() and token_words(p) not in NOT_INSURERS]


def insurer_company(name) -> tuple[str, str]:
    """(company, scheme) for an insurer as Beta Care spells it: NHIF's variants and Linda Mama are schemes of NHIF."""
    n = clean(name) or ""
    t = token_words(n)
    if t.startswith("nhif") or t == "linda mama":
        return "NHIF", n
    return n.rstrip(". "), n.rstrip(". ")


def split_full_name(full) -> tuple[str, str | None, str]:
    parts = (clean(full) or "Unknown").split()
    if len(parts) == 1:
        return parts[0], None, parts[0]
    return parts[0], " ".join(parts[1:-1]) or None, parts[-1]


def bp(v) -> tuple[int | None, int | None]:
    """'128/78mmhg' → (128, 78); anything outside plausible ranges → None."""
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", clean(v) or "")
    if not m:
        return None, None
    s, d = int(m[1]), int(m[2])
    return (s if 40 <= s <= 300 else None), (d if 20 <= d <= 200 else None)


def month_expiry(v) -> str | None:
    """Smart Hospital batch expiry 'Apr/2029' → last day of that month."""
    m = re.match(r"^\s*([A-Za-z]{3})[a-z]*\s*[/-]\s*(\d{4})\s*$", clean(v) or "")
    if not m:
        return day(v)
    try:
        mo = [x.lower() for x in calendar.month_abbr].index(m[1].lower())
    except ValueError:
        return None
    y = int(m[2])
    return date(y, mo, calendar.monthrange(y, mo)[1]).isoformat() if mo and 1900 <= y <= 2100 else None


def epoch_wall(ts) -> str | None:
    """unix seconds → 'YYYY-MM-DD HH:MM:SS' Nairobi wall time (what V3 is sent; see the V3 runbook)."""
    n = num(ts)
    if not n or n < 1e9:
        return None
    return datetime.fromtimestamp(n, EAT).strftime("%Y-%m-%d %H:%M:%S")


def allergies(v) -> str | None:
    s = clean(v)
    return None if not s or token_words(s) in NO_ALLERGY else s


def patient_payload(r: dict, patient_no: str, insurer: tuple | None) -> dict:
    """insurer: (company id or None, company name, scheme name) for an insured patient."""
    first, middle, last = split_full_name(r.get("patient_name"))
    email = clean(r.get("email"))
    if email and "@" not in email:              # the email box often holds the home area ('GITHUNGURI')
        email, area = None, email
    else:
        area = None
    age = " ".join(x for x in (f"{clean(r.get('age'))} y" if clean(r.get("age")) else None,
                               f"{clean(r.get('month'))} m" if clean(r.get("month")) else None) if x)
    when = stamp(r.get("created_at")) or stamp(r.get("admission_date"))
    notes = text_join(("Smart Hospital patient id", r["id"]), ("Smart Hospital patient no", r.get("patient_unique_id")),
                      ("Age at registration", age), ("Insurer", insurer[1] if insurer else None),
                      ("Area", area if area and clean(r.get("address")) else None), ("Note", r.get("note")),
                      ("Guardian address", r.get("guardian_address")), ("Old patient", r.get("old_patient")))
    return {"uuid": gid("patients", r["id"]), "patient_no": patient_no, "first_name": first[:100],
            "middle_name": (middle or "")[:100] or None, "last_name": last[:100], "dob": _eagle.day(r.get("dob")),
            "gender": gender(r.get("gender")), "marital_status": (clean(r.get("marital_status")) or "").lower() or None,
            "blood_group": clean(r.get("blood_group")), "mobile": phone(r.get("mobileno")),
            "email": email, "address": (clean(r.get("address")) or area or "")[:255] or None,
            "allergies": allergies(r.get("known_allergies")),
            "emergency_contact_name": clean(r.get("guardian_name")), "emergency_contact_phone": phone(r.get("guardian_phone")),
            "insurance_status": "insured" if insurer else "uninsured", "insurance_company_id": insurer[0] if insurer else None,
            "registration_date": (when or "")[:10] or None,
            "status": "inactive" if words(r.get("is_active")) == "no" else "active", "notes": notes,
            "created_at": when, "updated_at": when}


def visit_payload(r: dict, kind: str, patient_id: int, *, facility: int, full_name, doctor: int | None,
                  insured: bool) -> dict:
    ip = kind == "ipd"
    when = stamp(r.get("date") if ip else r.get("appointment_date"))
    no = clean(r.get("ipd_no") if ip else r.get("opd_no")) or f"{kind.upper()}{r['id']}"
    casualty = words(r.get("casualty")) == "yes"
    return {"uuid": gid(kind, r["id"]), "patient": patient_id, "clinic": facility, "inpatient": 1 if ip else 0,
            "patient_full_name": (clean(full_name) or "")[:255] or None, "purpose": clean(r.get("case_type")),
            "type_of_visit": "IP Visit" if ip else ("Casualty" if casualty else "OP Visit"), "emergency": 1 if casualty else 0,
            "payment_mode": "insurance" if insured else "cash", "status": "completed", "unique_id": f"BCH-{no}",
            "user": doctor, "created_at": when, "updated_at": when}


PAED = (("Usual weight", "usual"), ("Admission weight", "adwt"), ("Birth weight", "btwt"), ("Head circumference", "head"),
        ("Height/length", "heightl"), ("Weight-for-height z", "whz"), ("Height-for-age", "htage"), ("BMI-for-age", "bmiage"),
        ("Diet", "diet"), ("Breastfeeding", "breast"), ("Clinical", "clinical"), ("Medical history", "med_history"),
        ("Social", "social"), ("Birth order", "btorder"), ("Mother's age", "mum_age"), ("Mother's occupation", "occupation_mum"),
        ("Mother's residence", "residence_mum"), ("Father's age", "dad_age"), ("Father's occupation", "occupation_dad"),
        ("Father's residence", "residence_dad"), ("Siblings' ages", "sib_age"), ("Age (interval)", "ageint"))


def vitals_payload(r: dict, kind: str, visit_id: int, patient_id: int) -> dict | None:
    dropped: list[str] = []
    s, d = bp(r.get("bp"))
    if clean(r.get("bp")) and not (s and d):
        dropped.append(f"bp {clean(r.get('bp'))}")
    height, weight = vital("height", r.get("height"), dropped), vital("weight", r.get("weight"), dropped)
    temp = vital("temperature", r.get("temperature") or r.get("bodytemp"), dropped)
    pulse, resp = vital("pulse", r.get("pulse"), dropped), vital("respiratory_rate", r.get("respiration"), dropped)
    bmi = vital("bmi", r.get("bmi"), dropped)
    notes = text_join(("Remarks", r.get("note_remark") or r.get("note")), *((lbl, r.get(c)) for lbl, c in PAED),
                      ("Not loaded as numbers (out of range)", ", ".join(dropped)))
    symptoms, chronic = clean(r.get("symptoms")), clean(r.get("chronic_illnesses"))
    if not any((s, d, height, weight, temp, pulse, resp, bmi, notes, symptoms, chronic)):
        return None
    when = stamp(r.get("date") if kind == "ipd" else r.get("appointment_date"))
    return {"uuid": gid(f"{kind}-vitals", r["id"]), "visit_id": visit_id, "visit": visit_id, "patient_id": patient_id,
            "height": height, "weight": weight, "temperature": temp, "systolic_bp": s, "diastolic_bp": d,
            "bp_systolic": s, "bp_diastolic": d, "blood_pressure": f"{s}/{d}" if s and d else None,
            "pulse": int(pulse) if pulse else None, "respiratory_rate": int(resp) if resp else None, "bmi": bmi,
            "symptoms": symptoms, "chronic_illnesses": chronic, "allergies": allergies(r.get("known_allergies")),
            "notes": notes, "recorded_at": when, "created_at": when, "updated_at": when}


def clinical_note_payload(r: dict, visit_id: int, patient_id: int, when: str | None, doctor: tuple | None,
                          visit_day: str | None = None) -> dict:
    """A row of Smart Hospital's `diagnosis` table: report_type is the diagnosis, description the consultation notes."""
    text = clean(r.get("description"))
    dx = clean(r.get("report_type"))
    d = day(r.get("report_date"))
    far = d and visit_day and abs((date.fromisoformat(d) - date.fromisoformat(visit_day)).days) > VISIT_WINDOWS["note"][0]
    return {"uuid": gid("diagnosis-note", r["id"]), "visit_id": visit_id, "visit": visit_id, "patient_id": patient_id,
            "presenting_complaints": text or dx, "history_of_present_illness": text, "diagnosis": dx,
            "primary_diagnosis": dx, "doctor_id": doctor[0] if doctor else None, "user": doctor[0] if doctor else None,
            "doctor_name": doctor[1] if doctor else None, "status": "final",
            "remarks": text_join(("Note date", d if far else "not recorded in Smart Hospital (filed on the patient's latest visit)"
                                  if not d else None),
                                 ("Filed on", f"the patient's nearest visit ({visit_day}): Smart Hospital has none near the note's date"
                                  if far else None), ("Attached document", r.get("document"))),
            "created_at": when, "updated_at": when}


def diagnosis_payload(r: dict, visit_id: int, patient_id: int, when: str | None, primary: bool, user: int | None) -> dict | None:
    dx = clean(r.get("report_type"))
    if not dx:
        return None
    return {"uuid": gid("diagnosis", r["id"]), "visit_id": visit_id, "patient_id": patient_id, "diagnosis_text": dx[:500],
            "diagnosis_date": day(r.get("report_date")) or (when or "")[:10] or None, "diagnosis_type": "final",
            "is_primary": 1 if primary else 0, "is_provisional": 0, "status": "confirmed", "recorded_by": user,
            "notes": "From Smart Hospital's clinical notes", "created_at": when, "updated_at": when}


def request_note_payload(kind: str, r: dict, visit_id: int, patient_id: int) -> dict:
    """lab_request / rad_request: a doctor's free-text request for a test."""
    label = "Lab request" if kind == "lab_request" else "Radiology request"
    when = stamp(r.get("report_date"))
    text = text_join((label, r.get("report_type")), ("Clinical notes", r.get("description")))
    return {"uuid": gid(kind, r["id"]), "visit_id": visit_id, "visit": visit_id, "patient_id": patient_id,
            "presenting_complaints": label, "investigations": text, "remarks": text, "status": "final",
            "created_at": when, "updated_at": when}


INVESTIGATION_TYPE = {"laboratory": "laboratory", "radiology": "radiology"}


def investigation_payload(uid: str, visit_id: int, patient_id: int, *, type_: str, procedure, amount, when,
                          admission_id=None, result=None, comments=None) -> dict:
    a = num(amount) or 0.0
    return {"uuid": uid, "visit": visit_id, "patient_id": patient_id, "type": type_, "procedure": procedure,
            "quantity": 1, "price": a, "amount": a, "discount": 0, "ordered": 1, "admission_id": admission_id,
            "cancelled": 0, "result": {"text": result} if result else None, "comments": clean(comments),
            "created_at": when, "updated_at": when}


def prescription_payload(uid: str, visit_id: int, *, drug, drug_name, dosage, notes, when, quantity=None, price=None,
                         dispensed: bool, admission_id=None, store=None, user=None) -> dict:
    q = num(quantity)
    return {"uuid": uid, "visit": visit_id, "drug": drug, "drug_name": (clean(drug_name) or "")[:255] or None,
            "take": clean(dosage), "dosage": clean(dosage), "quantity": q if q is not None else 0,
            "price": num(price), "notes": clean(notes), "status": "active", "dispensed": 1 if dispensed else 0,
            "dispensed_quantity": q if dispensed and q else None, "paid": 1 if dispensed and price is not None else 0,
            "admission_id": admission_id, "store_id": store[0] if store else None, "store_name": store[1] if store else None,
            "user": user, "prescribed_by": user, "created_at": when, "updated_at": when}


def payment_payload(r: dict, *, patient: tuple | None, visit_id, user: int | None) -> dict | None:
    """The patient's own money on an invoicepayment row (cash + M-Pesa); None when an insurer paid all of it."""
    cash, mpesa = num(r.get("cash_amount")) or 0.0, num(r.get("mpesa_amount")) or 0.0
    if cash + mpesa <= 0:
        return None
    when = epoch_wall(r.get("timestamp")) or stamp(r.get("date"))
    if not when:
        return None
    a = round(cash + mpesa, 2)
    return {"uuid": gid("invoicepayment", r["payment_id"]), "receipt": f"BCH-R{r['payment_id']}", "receipt_prefix": "BCH",
            "patient": patient[0] if patient else None, "patient_name": patient[1] if patient else None,
            "patient_no": patient[2] if patient else None, "user": user, "visit": visit_id, "deposit": 0,
            "amount": a, "cash_amount": cash, "mpesa_amount": mpesa, "card_amount": 0, "cheque_amount": 0,
            "total_cash_payment": a, "total_amount_was": a, "discount": 0, "change": 0, "vat_amount": 0,
            "payment_mode": "mpesa" if mpesa >= cash else "cash", "status": "completed",
            "transaction_id": (clean(r.get("transaction_id")) or "")[:255] or None, "receipt_date": when[:10],
            "created_at": when, "updated_at": when}


# ── run ─────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, target: str, dry_run: bool, limit: int, workers: int, only_missing: bool = False):
        from pipelines_common import use_pipelines_dir
        use_pipelines_dir([target])
        from v3_to_v3_client import V3Client
        import snowflake_to_v3_migration as s2v3
        self.target, self.dry_run, self.limit, self.workers, self.only_missing = target, dry_run, limit, workers, only_missing
        self.c = V3Client(target)
        self.c.send_facility_header = True       # rows land in facility 94, not under no facility
        login = self.c.login
        self.c.login = lambda: patient_login(login, self.target)
        self.c._token = (self.c.login(), time.time())
        self.fac = self.c.facility_id
        self.sf = s2v3._snowflake_connect()
        self.today = date.today().isoformat()
        self._cache: dict = {}
        self.failures: dict[str, list] = defaultdict(list)
        self.only_patients: set[str] | None = None      # canary: Smart Hospital patient ids
        log.info("%r user %s", self.c, self.c.user_id)

    # ── source ──
    def q(self, sql: str):
        cur = self.sf.cursor()
        cur.execute(sql)
        cols = [c[0].lower() for c in cur.description]
        while True:
            batch = cur.fetchmany(50_000)
            if not batch:
                return
            for r in batch:
                yield dict(zip(cols, r))

    def table(self, name: str, order: str = "id") -> list[dict]:
        k = ("table", name)
        if k not in self._cache:
            o = f' ORDER BY TRY_TO_NUMBER("{order.upper()}")' if order else ""
            self._cache[k] = list(self.q(f'SELECT * FROM {SF}."{name.upper()}"{o}'))
        return self._cache[k]

    def by_id(self, name: str, col: str = "id") -> dict[str, dict]:
        k = ("by_id", name, col)
        if k not in self._cache:
            self._cache[k] = {sid(r.get(col)): r for r in self.table(name, col) if sid(r.get(col))}
        return self._cache[k]

    def wanted(self, patient_id) -> bool:
        return self.only_patients is None or (sid(patient_id) or "") in self.only_patients

    # ── source visits: what a dated row belongs to ──
    def visit_index(self):
        if "vindex" not in self._cache:
            opd, adm = defaultdict(list), defaultdict(list)
            for r in self.table("opd_details"):
                d = day(r.get("appointment_date"))
                if d and sid(r.get("patient_id")):
                    opd[sid(r["patient_id"])].append((d, sid(r["id"])))
            rehab = {sid(g["id"]) for g in self.table("bed_group") if "rehab" in key(g.get("name"))}
            for r in self.table("ipd_details"):
                d = day(r.get("date"))
                if d and sid(r.get("patient_id")):
                    out = day(r.get("discharged_date")) if words(r.get("discharged")) == "yes" else None
                    # Smart Hospital was cleaned up by discharging old admissions in batches months later
                    # (70 stays > 60 days): only a rehabilitation stay that long is real
                    cap = (date.fromisoformat(d) + timedelta(days=ADMISSION_WINDOW_DAYS)).isoformat()
                    if sid(r.get("bed_group_id")) not in rehab:
                        out = min(out or d, cap)
                    adm[sid(r["patient_id"])].append((d, out or d, sid(r["id"])))
            for v in opd.values():
                v.sort()
            self._cache["vindex"] = (opd, adm)
        return self._cache["vindex"]

    def source_visit(self, patient_id, when, *, opd_id=None, ipd_id=None, window: str = "lab") -> tuple[str, str] | None:
        """('opd' | 'ipd', Smart Hospital id) a row belongs to."""
        if sid(ipd_id) and sid(ipd_id) in self.by_id("ipd_details"):
            return "ipd", sid(ipd_id)
        if sid(opd_id) and sid(opd_id) in self.by_id("opd_details"):
            return "opd", sid(opd_id)
        d, pid = day(when), sid(patient_id)
        opd, adm = self.visit_index()
        if not d and pid and window == "note" and opd.get(pid):     # undated note (placeholder date): latest visit
            return "opd", opd[pid][-1][1]
        if not d or not pid:
            return None
        for a_from, a_to, iid in adm.get(pid, []):
            if a_from <= d <= a_to:
                return "ipd", iid
        back, fwd = VISIT_WINDOWS[window]
        dd = date.fromisoformat(d)
        lo, hi = (dd - timedelta(days=back)).isoformat(), (dd + timedelta(days=fwd)).isoformat()
        vs = opd.get(pid, [])
        i = bisect.bisect_right(vs, (d, "￿"))
        if i and vs[i - 1][0] >= lo:
            return "opd", vs[i - 1][1]
        if i < len(vs) and vs[i][0] <= hi:
            return "opd", vs[i][1]
        if window == "note" and vs:           # a clinical note far from any visit: the patient's nearest one
            j = min(range(len(vs)), key=lambda n: abs((date.fromisoformat(vs[n][0]) - dd).days))
            return "opd", vs[j][1]
        return None

    # ── V3 lookups ──
    def ids_by_uuid(self, svc: str, alias: str, fresh: bool = False) -> dict[str, int]:
        k = ("uuid", svc, alias)
        if fresh or k not in self._cache:
            self._cache[k] = {str(r.get("uuid")).lower(): r["id"] for r in self.c.read_all(svc, alias, workers=8)
                              if r.get("uuid") and not r.get("deleted_at")}
        return self._cache[k]

    def visit_map(self) -> dict[str, tuple[int, int]]:
        if "visits" not in self._cache:
            self._cache["visits"] = {str(v["uuid"]).lower(): (v["id"], v.get("patient"))
                                     for v in self.c.read_all("evaluation", "visits", workers=8) if v.get("uuid")}
        return self._cache["visits"]

    def v3_visit(self, sv: tuple[str, str] | None) -> tuple[int, int] | None:
        return self.visit_map().get(gid(sv[0], sv[1])) if sv else None

    def patient(self, patient_id) -> int | None:
        return self.ids_by_uuid("reception", "patient").get(gid("patients", sid(patient_id))) if sid(patient_id) else None

    def admission(self, ipd_id) -> int | None:
        return self.ids_by_uuid("inpatient", "admission").get(gid("admission", sid(ipd_id))) if sid(ipd_id) else None

    def patient_card(self) -> dict[int, tuple]:
        if "pcard" not in self._cache:
            self._cache["pcard"] = {p["id"]: (p["id"], " ".join(x for x in (p.get("first_name"), p.get("middle_name"),
                                                                            p.get("last_name")) if x), p.get("patient_no"))
                                    for p in self.c.read_all("reception", "patient", workers=8)}
        return self._cache["pcard"]

    # ── posting (insert with match_on=uuid → an existing uuid is updated) ──
    def post(self, svc: str, alias: str, payloads) -> Counter:
        res = Counter()
        if alias in NO_MATCH_ON and not self.dry_run:
            self._known = dict(self.ids_by_uuid(svc, alias, fresh=True))
        have = set(self.ids_by_uuid(svc, alias, fresh=True)) if self.only_missing and not self.dry_run else None
        it, sent = iter(payloads), 0
        while True:
            chunk = list(islice(it, CHUNK))
            if not chunk:
                break
            if have is not None:
                before = len(chunk)
                chunk = [p for p in chunk if p["uuid"] not in have]
                res["already in V3 (skipped)"] += before - len(chunk)
            if self.limit:
                chunk = chunk[:max(self.limit - sent, 0)]
            if not chunk:
                if self.limit and sent >= self.limit:
                    break
                continue
            sent += len(chunk)
            if self.dry_run:
                for p in chunk[:2]:
                    log.info("  dry-run %s/%s: %s", svc, alias, json.dumps(p, default=str)[:900])
                res["would_post"] += len(chunk)
            else:
                res.update(self._post_chunk(svc, alias, chunk))
            if self.limit and sent >= self.limit:
                break
        return res

    def _post_chunk(self, svc: str, alias: str, payloads: list[dict]) -> Counter:
        res, lock = Counter(), threading.Lock()

        def send(data):
            if alias in NO_MATCH_ON:            # uuid → id read up front: update by id, else a plain insert
                rid = self._known.get(data.get("uuid"))
                if rid:
                    r = self.c.post(svc, {"action": "update", "model": alias, "destination_tenant_id": self.c.organization_id,
                                          "ids": [rid], "data": {k: v for k, v in data.items() if k != "uuid"}}, timeout=120)
                else:
                    r = self.c.insert(svc, alias, data, match_on=None)
                ok = r.ok and (r.json() if r.content else {}).get("success", True)
                return ok, None if ok else f"{r.status_code} {r.text[:300]}"
            r = self.c.insert(svc, alias, data, match_on="uuid")
            ok = r.ok and (r.json() if r.content else {}).get("success", True)
            return ok, None if ok else f"{r.status_code} {r.text[:300]}"

        def one(p):
            data = {k: v for k, v in p.items() if v is not None and not k.startswith("_")}
            if alias in FACILITY_MODELS and self.fac:
                data.setdefault("facility_id", self.fac)
            try:
                ok, err = send(data)
                # a number another organisation already uses (unique instance-wide): the fallback in "_alt"
                if not ok and p.get("_alt") and re.search(r"duplicate|already been taken|unique", err or "", re.I):
                    ok, err = send({**data, **p["_alt"]})
                    with lock:
                        res["renamed (number taken)"] += ok
            except Exception as e:
                ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
            with lock:
                res["ok" if ok else "failed"] += 1
                if not ok:
                    self.failures[f"{svc}/{alias}"].append({"uuid": p.get("uuid"), "error": err})
                    if res["failed"] <= 5:
                        log.error("  %s/%s %s: %s", svc, alias, p.get("uuid"), err)
                n = res["ok"] + res["failed"]
                if n % 5000 == 0:
                    log.info("  %s/%s: %d / %d (%d failed)", svc, alias, n, len(payloads), res["failed"])
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(one, payloads))
        return res

    def ref(self, svc: str, alias: str, name: str, extra: dict | None = None, match: str = "name") -> int:
        """A reference row by name: org 96's own is reused, else one is created with a stable uuid."""
        k = ("ref", svc, alias)
        if k not in self._cache:
            self._cache[k] = [r for r in self.c.read_all(svc, alias) if not r.get("deleted_at")]
        rows = self._cache[k]
        hit = next((r["id"] for r in rows if key(r.get(match)) == key(name)), None)
        if hit:
            return hit
        uid = gid(f"{alias}-ref", key(name))
        hit = next((r["id"] for r in rows if str(r.get("uuid")).lower() == uid), None)
        if hit or self.dry_run:
            return hit or 0
        data = {"uuid": uid, match: name, **(extra or {})}
        if alias in FACILITY_MODELS and self.fac:
            data.setdefault("facility_id", self.fac)
        r = self.c.insert(svc, alias, data, match_on="uuid")
        if not r.ok or (r.json() if r.content else {}).get("success") is False:
            raise RuntimeError(f"{svc}/{alias} '{name}': {r.status_code} {r.text[:300]}")
        new_id = ((r.json() or {}).get("data") or {}).get("id")
        if not new_id:
            self._cache[k] = list(self.c.read_all(svc, alias))
            return next(x["id"] for x in self._cache[k] if str(x.get("uuid")).lower() == uid)
        rows.append({"id": new_id, "uuid": uid, match: name})
        return new_id

    # ── staff ──
    def users(self) -> dict:
        existing = self.existing_users()
        why, users, profiles = Counter(), [], []
        for r in self.table("staff"):
            if self.staff_match(r, existing):
                why["matched to an existing org-96 user"] += 1
                continue
            login = re.sub(r"[^a-z0-9._-]", "", (clean(r.get("email")) or "").split("@")[0].lower()) or f"staff{r['id']}"
            users.append({"uuid": gid("staff", r["id"]), "username": f"bch.{login}"[:60],
                          "email": f"{login}.{r['id']}@betacare.migrated.invalid", "password": str(uuid.uuid4()),
                          "enforce_password_reset": 1, "active": 0, "employee_number": f"BCH-S-{r['id']}",
                          "_alt": {"username": f"bch.{login}.{r['id']}"[:60]}})
            why["created (inactive: history only)"] += 1
        res = {"users": dict(self.post("core", "users", users))}
        ids = self.ids_by_uuid("core", "users", fresh=True)
        desig = {sid(d["id"]): clean(d.get("designation")) for d in self.table("staff_designation")}
        for r in self.table("staff"):
            u = ids.get(gid("staff", r["id"]))
            if not u:
                continue
            title, first, middle, last = person_name(f"{clean(r.get('name')) or ''} {clean(r.get('surname')) or ''}")
            email = clean(r.get("email"))
            profiles.append({"uuid": gid("staff-profile", r["id"]), "user_id": u,
                             **({"title": TITLE_CODES[title]} if title in TITLE_CODES else {}),
                             "first_name": first[:100], "middle_name": (middle or "")[:100] or None, "last_name": last[:100],
                             "gender": gender(r.get("gender")), "phone": phone(r.get("contact_no")),
                             "work_email": email if email and "@" in email else None, "staff_number": f"BCH-S-{r['id']}",
                             "job_description": desig.get(sid(r.get("designation"))), "dob": _eagle.day(r.get("dob"))})
        res["profiles"] = dict(self.post("core", "user_profile", profiles))
        return {"staff": len(self.table("staff")), **why, **res}

    def existing_users(self) -> dict[str, int]:
        """'first last' (alnum) → user id, for org-96 users whose profile name is unique."""
        if "existing_users" not in self._cache:
            ours = {gid("staff", r["id"]) for r in self.table("staff")}
            users = {u["id"]: u for u in self.c.read_all("core", "users") if str(u.get("uuid")).lower() not in ours}
            names = defaultdict(set)
            for p in self.c.read_all("core", "user_profile"):
                if p.get("user_id") in users:
                    names[key(f"{p.get('first_name') or ''}{p.get('last_name') or ''}")].add(p["user_id"])
            self._cache["existing_users"] = {k: next(iter(v)) for k, v in names.items() if len(v) == 1 and k}
        return self._cache["existing_users"]

    def staff_match(self, r: dict, existing: dict[str, int]) -> int | None:
        title, first, middle, last = person_name(f"{clean(r.get('name')) or ''} {clean(r.get('surname')) or ''}")
        return existing.get(key(f"{first}{last}")) if clean(r.get("surname")) else None

    def staff_user(self, staff_id) -> int | None:
        """Smart Hospital staff id → V3 user id."""
        if "staff_users" not in self._cache:
            existing, ids = self.existing_users(), self.ids_by_uuid("core", "users")
            self._cache["staff_users"] = {sid(r["id"]): self.staff_match(r, existing) or ids.get(gid("staff", r["id"]))
                                          for r in self.table("staff")}
        return self._cache["staff_users"].get(sid(staff_id))

    def staff_name(self, staff_id) -> str | None:
        r = self.by_id("staff").get(sid(staff_id))
        if not r:
            return None
        return " ".join(x for x in (clean(r.get("name")), clean(r.get("surname"))) if x) or None

    # ── insurance ──
    def insurer_names(self) -> dict[str, str]:
        """Every insurer Beta Care used, by its own spelling → 'active' (still in its organisation list) / 'inactive'."""
        out = {}
        for r in self.table("organisation"):
            if token_words(r.get("organisation_name")) not in NOT_INSURERS:
                out[clean(r["organisation_name"])] = "active"
        for n in DELETED_ORGS.values():
            out.setdefault(n, "inactive")
        for r in self.q(f'SELECT DISTINCT METHOD FROM {SF}."INVOICEPAYMENT"'):
            for n in method_insurers(r.get("method")):
                out.setdefault(n, "inactive")
        return out

    def insurance(self) -> dict:
        from gyno_to_v3 import core_insurer_match, scheme_type
        names = self.insurer_names()
        core = {words(r.get("name")): r["id"] for r in self.c.read_all("core", "insurance_companies")}
        companies: dict[str, dict] = {}
        why = Counter()
        for n, status in names.items():
            comp, _ = insurer_company(n)
            if core_insurer_match(comp, core) or core.get(words(comp)):
                why["insurer matched to org 96's own"] += 1
                continue
            c = companies.setdefault(key(comp), {"uuid": gid("insurance-company", key(comp)), "name": comp[:255],
                                                 "code": f"BCH-INS-{key(comp)[:20].upper()}", "status": status})
            if status == "active":
                c["status"] = "active"
        res = {"companies": dict(self.post("core", "insurance_companies", companies.values()))}
        self._cache.pop("insurers", None)
        schemes = {}
        for n in names:
            comp, scheme = insurer_company(n)
            cid = self.company_id(comp)
            if cid:
                schemes.setdefault(key(f"{comp}|{scheme}"), {
                    "uuid": gid("insurance-scheme", key(f"{comp}|{scheme}")), "company_id": cid, "name": scheme[:255],
                    "code": f"BCH-SCH-{key(scheme)[:20].upper()}", "type": scheme_type(scheme), "status": "active"})
        res["schemes"] = dict(self.post("core", "insurance_schemes", schemes.values()))
        return {"insurers": len(names), **why, "created": len(companies), **res}

    def company_id(self, company) -> int | None:
        from gyno_to_v3 import core_insurer_match
        if "insurers" not in self._cache:
            rows = list(self.c.read_all("core", "insurance_companies"))
            self._cache["insurers"] = ({words(r.get("name")): r["id"] for r in rows},
                                       {str(r.get("uuid")).lower(): r["id"] for r in rows}, {r["id"]: r.get("name") for r in rows})
        core, by_uuid, _ = self._cache["insurers"]
        return core_insurer_match(company, core) or core.get(words(company)) or by_uuid.get(gid("insurance-company", key(company)))

    def org_insurer(self, org_id) -> str | None:
        """A patient's / payment's organisation id → the insurer's name as Beta Care spells it."""
        o = sid(org_id)
        if not o:
            return None
        r = self.by_id("organisation").get(o)
        name = clean(r.get("organisation_name")) if r else DELETED_ORGS.get(o)
        return name if name and token_words(name) not in NOT_INSURERS else None

    def insurer_tuple(self, name) -> tuple | None:
        """(company id, company name, scheme name) for an insurer name, or None."""
        if not name:
            return None
        comp, scheme = insurer_company(name)
        return self.company_id(comp), comp, scheme

    # ── patients ──
    def patient_numbers(self) -> dict[str, str]:
        if "pnos" not in self._cache:
            rows = self.table("patients")
            seen = Counter(clean(r.get("patient_unique_id")) for r in rows if clean(r.get("patient_unique_id")))
            self._cache["pnos"] = {sid(r["id"]): (clean(r.get("patient_unique_id"))
                                                  if seen.get(clean(r.get("patient_unique_id"))) == 1 else f"BCH-{r['id']}")
                                   for r in rows}
        return self._cache["pnos"]

    def patients(self) -> dict:
        nos, why = self.patient_numbers(), Counter()

        def gen():
            for r in self.table("patients"):
                if not self.wanted(r["id"]):
                    continue
                ins = self.insurer_tuple(self.org_insurer(r.get("organisation")))
                why["insured" if ins else "cash"] += 1
                why["numbered BCH-<id> (blank / shared number)"] += nos[sid(r["id"])].startswith("BCH-")
                yield {**patient_payload(r, nos[sid(r["id"])], ins), "_alt": {"patient_no": f"BCH-{r['id']}"}}
        res = self.post("reception", "patient", gen())
        self._cache.pop(("uuid", "reception", "patient"), None)
        return {"source": len(self.table("patients")), **why, **res}

    def patient_insurance(self) -> dict:
        why = Counter()

        def gen():
            for r in self.table("patients"):
                ins = self.insurer_tuple(self.org_insurer(r.get("organisation")))
                if not ins or not self.wanted(r["id"]):
                    continue
                pid = self.patient(r["id"])
                if not pid:
                    why["patient not in V3"] += 1
                    continue
                cid, comp, scheme = ins
                why["linked to insurer" if cid else "insurer not in V3"] += 1
                yield {"uuid": gid("patient-insurance", r["id"]), "patient_id": pid, "insurance_type": "insurance",
                       "insurance_company_id": cid, "insurance_company_name": comp,
                       "insurance_scheme_name": scheme, "status": "active", "is_primary": 1,
                       "member_no": None, "relationship": "principal"}
        res = self.post("reception", "patient_insurance", gen())
        return {**why, **res}

    # ── visits ──
    def bill_insurers(self) -> dict[tuple[str, str], str]:
        """('opd' | 'ipd', id) → the insurer that paid (part of) that visit's bill."""
        if "bill_ins" not in self._cache:
            inv = defaultdict(list)
            for p in self.table("invoicepayment", "payment_id"):
                ins = method_insurers(p.get("method"))
                if (num(p.get("nhif_amount")) or 0) > 0 and not ins:
                    ins = ["NHIF"]
                if (num(p.get("second_insurance_amount")) or 0) > 0:
                    ins.append(self.org_insurer(p.get("second_insurance_id")) or "")
                inv[sid(p.get("invoice_id"))] += [x for x in ins if x]
            bill_inv = defaultdict(list)
            for i in self.table("invoice", "invoice_id"):
                bill_inv[sid(i.get("bill_no"))].append(sid(i.get("invoice_id")))
            out = {}
            for m in self.table("master_bill"):
                names = [n for iid in bill_inv.get(sid(m.get("bill_no")), []) for n in inv.get(iid, [])]
                if names:
                    for kind in ("opd", "ipd"):
                        if sid(m.get(f"{kind}_id")):
                            out[(kind, sid(m[f"{kind}_id"]))] = names[0]
            self._cache["bill_ins"] = out
        return self._cache["bill_ins"]

    def visits(self) -> dict:
        why, ins = Counter(), self.bill_insurers()
        pats = self.by_id("patients")

        def gen():
            for kind, table in (("opd", "opd_details"), ("ipd", "ipd_details")):
                for r in self.table(table):
                    why[f"{table} source"] += 1
                    if not self.wanted(r.get("patient_id")):
                        continue
                    pid = self.patient(r.get("patient_id"))
                    if not pid:
                        why[f"{table}: patient not in Smart Hospital's register (skipped)"] += 1
                        continue
                    yield visit_payload(r, kind, pid, facility=self.fac,
                                        full_name=(pats.get(sid(r.get("patient_id"))) or {}).get("patient_name"),
                                        doctor=self.staff_user(r.get("cons_doctor")), insured=(kind, sid(r["id"])) in ins)
        res = self.post("evaluation", "visits", gen())
        self._cache.pop("visits", None)
        return {**why, **res}

    def per_visit(self, rows, svc: str, alias: str, locate, build) -> dict:
        """rows → payloads for the V3 visit `locate(r)` names (a source visit); `build(r, visit_id, patient_id, sv)`."""
        why = Counter()

        def gen():
            for r in rows:
                why["source"] += 1
                sv = locate(r)
                if sv is None:
                    why["no visit on / just before that date (skipped)"] += 1
                    continue
                v = self.v3_visit(sv)
                if v is None:
                    why["visit not in V3 (skipped)"] += 1
                    continue
                if not self.wanted(self.source_patient(sv)):
                    continue
                p = build(r, v[0], v[1], sv)
                if p is None:
                    why["nothing to load"] += 1
                    continue
                yield p
        res = self.post(svc, alias, gen())
        return {**why, **res}

    def source_patient(self, sv: tuple[str, str]) -> str | None:
        r = self.by_id("opd_details" if sv[0] == "opd" else "ipd_details").get(sv[1]) or {}
        return sid(r.get("patient_id"))

    def visit_stamp(self, sv: tuple[str, str]) -> str | None:
        r = self.by_id("opd_details" if sv[0] == "opd" else "ipd_details").get(sv[1]) or {}
        return stamp(r.get("appointment_date") if sv[0] == "opd" else r.get("date"))

    def visit_doctor(self, sv: tuple[str, str]) -> tuple | None:
        r = self.by_id("opd_details" if sv[0] == "opd" else "ipd_details").get(sv[1]) or {}
        u = self.staff_user(r.get("cons_doctor"))
        return (u, self.staff_name(r.get("cons_doctor"))) if u or self.staff_name(r.get("cons_doctor")) else None

    def vitals(self) -> dict:
        out = {}
        for kind, table in (("opd", "opd_details"), ("ipd", "ipd_details")):
            out[table] = self.per_visit(self.table(table), "evaluation", "vitals", lambda r, k=kind: (k, sid(r["id"])),
                                        lambda r, vid, pid, sv: vitals_payload(r, sv[0], vid, pid))
        return out

    def note_when(self, r_date, sv) -> str | None:
        """A note's time: its visit's, when the note is from the visit's day; else noon on its own day."""
        vs = self.visit_stamp(sv)
        d = day(r_date)
        return vs if vs and (not d or vs[:10] == d) else (f"{d} 12:00:00" if d else vs)

    def doctor_notes(self) -> dict:
        out = {"clinical_notes": self.per_visit(
            self.table("diagnosis"), "evaluation", "doctor_notes",
            lambda r: self.source_visit(r.get("patient_id"), r.get("report_date"), window="note"),
            lambda r, vid, pid, sv: clinical_note_payload(r, vid, pid, self.note_when(r.get("report_date"), sv),
                                                          self.visit_doctor(sv), (self.visit_stamp(sv) or "")[:10] or None))}
        for t in ("lab_request", "rad_request"):
            out[t] = self.per_visit(self.table(t), "evaluation", "doctor_notes",
                                    lambda r: self.source_visit(r.get("patient_id"), r.get("report_date"),
                                                                opd_id=r.get("opd_id"), window="note"),
                                    lambda r, vid, pid, sv, t=t: request_note_payload(t, r, vid, pid))
        return out

    def diagnoses(self) -> dict:
        first: set = set()

        def build(r, vid, pid, sv):
            primary = sv not in first
            first.add(sv)
            doc = self.visit_doctor(sv)
            return diagnosis_payload(r, vid, pid, self.note_when(r.get("report_date"), sv), primary, doc[0] if doc else None)
        return self.per_visit(self.table("diagnosis"), "evaluation", "diagnoses",
                              lambda r: self.source_visit(r.get("patient_id"), r.get("report_date"), window="note"), build)

    # ── inpatient ──
    def wards(self) -> dict:
        v3 = {key(w.get("name")): w["id"] for w in self.c.read_all("inpatient", "ward") if not w.get("deleted_at")}
        pl, why, mapping = [], Counter(), {}
        for g in self.table("bed_group"):
            k = WARD_MAP.get(key(g.get("name")), key(g.get("name")))
            if k in v3:
                mapping[clean(g["name"])] = f"existing ward {v3[k]}"
                continue
            name = "Rehabilitation" if k == "rehabilitation" else clean(g["name"])
            if any(p["uuid"] == gid("ward", k) for p in pl):
                continue
            low = name.lower()
            pl.append({"uuid": gid("ward", k), "name": name[:100], "display_name": name[:150], "code": f"BCH-W-{k}"[:50],
                       "age_group": "adult", "is_active": 0, "is_icu": 0, "is_maternity": int("mater" in low)})
            mapping[clean(g["name"])] = "created (inactive)"
        return {"bed_groups": mapping, **self.post("inpatient", "ward", pl), **why}

    def ward_for(self, bed_group_id) -> int | None:
        if "wardmap" not in self._cache:
            v3 = {key(w.get("name")): w["id"] for w in self.c.read_all("inpatient", "ward") if not w.get("deleted_at")}
            mine = self.ids_by_uuid("inpatient", "ward", fresh=True)
            out = {}
            for g in self.table("bed_group"):
                k = WARD_MAP.get(key(g.get("name")), key(g.get("name")))
                out[sid(g["id"])] = v3.get(k) or mine.get(gid("ward", k))
            self._cache["wardmap"] = out
        return self._cache["wardmap"].get(sid(bed_group_id))

    def admission_state(self, r: dict) -> tuple[str, str | None, bool]:
        """(status, discharge datetime, closed by the migration)."""
        admitted = stamp(r.get("date"))
        if words(r.get("discharged")) == "yes":
            return "discharged", stamp(r.get("discharged_date")) or admitted, False
        recent = admitted and admitted[:10] >= (date.today() - timedelta(days=OPEN_ADMISSION_DAYS)).isoformat()
        return ("admitted", None, False) if recent else ("discharged", admitted, True)

    def admissions(self) -> dict:
        type_id = self.ref("inpatient", "admission_type", "Inpatient Admission", {"code": "BCH-INPATIENT"})
        pats, why, ins = self.by_id("patients"), Counter(), self.bill_insurers()

        def gen():
            for r in self.table("ipd_details"):
                v = self.v3_visit(("ipd", sid(r["id"])))
                if v is None or not self.wanted(r.get("patient_id")):
                    why["IP visit not in V3"] += v is None
                    continue
                status, out, closed = self.admission_state(r)
                why["still admitted" if status == "admitted" else "closed by migration (never discharged)" if closed
                    else "discharged"] += 1
                doc = self.staff_user(r.get("cons_doctor")) or self.c.user_id
                admitted = stamp(r.get("date"))
                bed = (self.by_id("bed").get(sid(r.get("bed"))) or {}).get("name")
                insurer = ins.get(("ipd", sid(r["id"])))
                yield {"uuid": gid("admission", r["id"]), "admission_number": f"BCH-{clean(r.get('ipd_no')) or r['id']}",
                       "inpatient_no": clean(r.get("ipd_no")), "patient_id": v[1], "visit_id": v[0],
                       "patient_name": (clean((pats.get(sid(r.get("patient_id"))) or {}).get("patient_name")) or "")[:255] or None,
                       "admission_type_id": type_id, "admission_date": admitted, "admission_time": (admitted or "")[11:] or None,
                       "admitting_doctor_id": doc, "consultant_id": doc, "reason_for_admission": clean(r.get("symptoms")),
                       "admission_notes": clean(r.get("note")), "admission_source": "direct",
                       "is_emergency": 1 if words(r.get("casualty")) == "yes" else 0,
                       "has_allergies": 1 if allergies(r.get("known_allergies")) else 0,
                       "allergies_details": allergies(r.get("known_allergies")), "status": status, "discharge_date": out,
                       "payment_mode": "insurance" if insurer else "cash", "insurance_scheme_name": insurer,
                       "insurance_limit": num(r.get("credit_limit")),
                       "special_instructions": text_join(("Bed", bed), ("Credit limit", r.get("credit_limit")),
                                                         ("Closed by the migration", "not discharged in Smart Hospital" if closed else None)),
                       "created_by": doc, "created_at": admitted, "updated_at": out or admitted}
        res = self.post("inpatient", "admission", gen())
        self._cache.pop(("uuid", "inpatient", "admission"), None)
        return {"admission_type": type_id, **why, **res}

    def ward_stays(self) -> dict:
        why, pl = Counter(), []
        for r in self.table("ipd_details"):
            a = self.admission(r["id"])
            w = self.ward_for(r.get("bed_group_id"))
            if not a or not w:
                why["no admission / ward"] += 1
                continue
            status, out, _ = self.admission_state(r)
            when = stamp(r.get("date"))
            who = self.staff_user(r.get("cons_doctor")) or self.c.user_id
            bed = (self.by_id("bed").get(sid(r.get("bed"))) or {}).get("name")
            done = status == "discharged"
            pl.append({"uuid": gid("visit_ward", r["id"]), "admission_id": a, "ward_id": w, "allocation_date": when,
                       "transfer_in_date": (when or "")[:10] or None, "transfer_in_time": (when or "")[11:] or None,
                       "allocation_type": "admission", "release_date": out if done else None,
                       "discharge_date": out if done else None, "status": "released" if done else "active",
                       "allocated_by": who, "created_by": who, "notes": text_join(("Bed", bed)),
                       "days_stayed": max((date.fromisoformat(out[:10]) - date.fromisoformat(when[:10])).days, 1)
                       if done and out and when else None, "created_at": when, "updated_at": out or when})
        res = self.post("inpatient", "visit_ward", pl)
        return {**why, **res}

    def discharges(self) -> dict:
        type_id = self.ref("inpatient", "discharge_type", "Patient is Stable", {"code": "PATIENT_IS_STABLE"})
        why, pl = Counter(), []
        for r in self.table("ipd_details"):
            a = self.admission(r["id"])
            status, out, closed = self.admission_state(r)
            if not a or status != "discharged":
                continue
            who = self.staff_user(r.get("cons_doctor")) or self.c.user_id
            text = ("Closed by the migration: the admission was never discharged in Smart Hospital" if closed
                    else "Discharged in Smart Hospital (no discharge summary recorded there)")
            pl.append({"uuid": gid("discharge", r["id"]), "discharge_number": f"DIS-BCH-{clean(r.get('ipd_no')) or r['id']}"[:100],
                       "admission_id": a, "discharge_type_id": type_id, "discharge_date": out,
                       "discharge_time": (out or "")[11:] or None, "discharged_by": who, "discharging_doctor_id": who,
                       "discharging_doctor_name": self.staff_name(r.get("cons_doctor")),
                       "discharge_diagnosis": "See the admission's clinical notes", "discharge_summary": text,
                       "discharge_instructions": text,
                       "patient_condition": "stable", "created_at": out, "updated_at": out})
            why["closed by migration" if closed else "discharged"] += 1
        res = self.post("inpatient", "discharge", pl)
        return {**why, **res}

    # ── catalogue ──
    def catalogue_rows(self) -> list[tuple[str, str, str, str, bool]]:
        """(uuid key, code, name, V3 category, active) for every billable item: charges, tests with no charge, and charge
        ids patients were billed under that the charge list no longer has."""
        out, charges = [], self.by_id("charges")
        for r in self.table("charges"):
            name = clean(r.get("description")) or clean(r.get("code")) or f"Charge {r['id']}"
            out.append((f"charge:{sid(r['id'])}", f"BCH-C{sid(r['id'])}", name,
                        (clean(r.get("charge_type")) or "General").title(), words(r.get("status")) not in ("inactive", "no", "0")))
        for t, cat, prefix in (("pathology", "Laboratory", "L"), ("radio", "Radiology", "R")):
            for r in self.table(t):
                if sid(r.get("charge_id")) not in charges:
                    out.append((f"{t}:{sid(r['id'])}", f"BCH-{prefix}{sid(r['id'])}", clean(r.get("test_name")) or f"Test {r['id']}", cat, True))
        billed = {sid(r.get("charge_id")) for t in ("opd_patient_charges", "patient_charges") for r in self.table(t)}
        for c in sorted(x for x in billed - set(charges) if x):
            out.append((f"charge:{c}", f"BCH-C{c}", f"Charge #{c} (removed from Smart Hospital's charge list)", "General", False))
        return out

    def catalogue(self) -> dict:
        rows = self.catalogue_rows()
        cats = {c: self.ref("evaluation", "procedure_categories", c) for c in sorted({r[3] for r in rows})}
        pl = [{"uuid": gid("procedure", k), "code": code, "name": name[:255], "category": cats[cat], "active": int(active),
               "description": "Beta Care Hospital (Smart Hospital) charge"} for k, code, name, cat, active in rows]
        return {"procedures": len(pl), "categories": cats, **self.post("evaluation", "procedures", pl)}

    def procedure_for(self, charge_id=None, test: tuple[str, str] | None = None) -> int | None:
        procs = self.ids_by_uuid("evaluation", "procedures")
        if test:
            t = self.by_id(test[0]).get(test[1]) or {}
            c = sid(t.get("charge_id"))
            if c and c in self.by_id("charges"):
                return procs.get(gid("procedure", f"charge:{c}"))
            return procs.get(gid("procedure", f"{test[0]}:{test[1]}"))
        return procs.get(gid("procedure", f"charge:{sid(charge_id)}")) if sid(charge_id) else None

    def procedure_prices(self) -> dict:
        procs = self.ids_by_uuid("evaluation", "procedures", fresh=True)
        org = defaultdict(Counter)
        for r in self.table("organisations_charges"):
            if (num(r.get("org_charge")) or 0) > 0:
                org[sid(r.get("charge_id"))][num(r["org_charge"])] += 1
        pl, why = [], Counter()
        for r in self.table("charges"):
            pid = procs.get(gid("procedure", f"charge:{sid(r['id'])}"))
            if not pid:
                why["procedure not in V3"] += 1
                continue
            cash = num(r.get("standard_charge")) or 0.0
            ins = org[sid(r["id"])].most_common(1)[0][0] if org.get(sid(r["id"])) else cash
            pl.append({"uuid": gid("price", f"{sid(r['id'])}|{self.fac}"), "evaluation_procedure_id": pid,
                       "facility_id": self.fac, "facility_name": self.facility_name(), "cash_charge": cash,
                       "insurance_charge": ins, "charge_insurance": 0, "precharge": 0, "status": 1})
        res = self.post("evaluation", "procedure_prices", pl)
        return {**why, **res}

    def facility_name(self) -> str | None:
        if "facility_name" not in self._cache:
            self._cache["facility_name"] = next((f.get("name") for f in self.c.read_all("core", "facilities")
                                                 if f.get("id") == self.fac), None)
        return self._cache["facility_name"]

    # ── investigations ──
    def charge_type(self, charge_id) -> str:
        t = token_words((self.by_id("charges").get(sid(charge_id)) or {}).get("charge_type"))
        return INVESTIGATION_TYPE.get(t, "procedure")

    def report_matches(self) -> dict[tuple[str, str], str]:
        """(pathology_report | radiology_report, id) → the OPD charge that billed the same test: one investigation."""
        if "report_matches" not in self._cache:
            idx = defaultdict(list)
            for r in self.table("opd_patient_charges"):
                d = day(r.get("date"))
                if d:
                    idx[(sid(r.get("patient_id")), sid(r.get("charge_id")))].append((d, sid(r["id"])))
            used, out = set(), {}
            for t, test_t, col in (("pathology_report", "pathology", "pathology_id"), ("radiology_report", "radio", "radiology_id")):
                for r in self.table(t):
                    d = day(r.get("reporting_date")) or day(r.get("created_at"))
                    c = sid((self.by_id(test_t).get(sid(r.get(col))) or {}).get("charge_id"))
                    if not d or not c:
                        continue
                    lo = (date.fromisoformat(d) - timedelta(days=3)).isoformat()
                    hits = sorted((x for x in idx.get((sid(r.get("patient_id")), c), []) if lo <= x[0] <= d and x[1] not in used),
                                  reverse=True)
                    if hits:
                        used.add(hits[0][1])
                        out[(t, sid(r["id"]))] = hits[0][1]
            self._cache["report_matches"] = out
        return self._cache["report_matches"]

    def report_text(self, t: str, r: dict) -> str | None:
        if "report_params" not in self._cache:
            names = {sid(p["id"]): clean(p.get("parameter_name")) for p in self.table("pathology_parameter")}
            params = defaultdict(list)
            for p in self.table("pathology_report_parameterdetails"):
                if clean(p.get("pathology_report_value")):
                    params[sid(p.get("pathology_report_id"))].append(
                        (names.get(sid(p.get("parameter_id"))) or "Value", p["pathology_report_value"]))
            self._cache["report_params"] = params
        params = self._cache["report_params"].get(sid(r["id"]), []) if t == "pathology_report" else []
        return text_join(("Result", r.get("description")), *params, ("Report file", r.get(t)))

    def investigations(self) -> dict:
        self.ids_by_uuid("evaluation", "procedures", fresh=True)
        self.ids_by_uuid("inpatient", "admission", fresh=True)
        out = {}
        for t, kind in (("opd_patient_charges", "opd_charge"), ("patient_charges", "ipd_charge")):
            out[t] = self.per_visit(
                self.table(t), "evaluation", "investigations",
                lambda r: self.source_visit(r.get("patient_id"), r.get("date"), opd_id=r.get("opd_id"),
                                            ipd_id=r.get("ipd_id"), window="charge"),
                lambda r, vid, pid, sv, kind=kind: investigation_payload(
                    gid(kind, r["id"]), vid, pid, type_=self.charge_type(r.get("charge_id")),
                    procedure=self.procedure_for(r.get("charge_id")), amount=r.get("apply_charge"),
                    when=stamp(r.get("created_at")) or stamp(r.get("date")),
                    admission_id=self.admission(sv[1]) if sv[0] == "ipd" else None,
                    comments=text_join(("Smart Hospital bill", r.get("master_bill_id")))))
        matched = self.report_matches()
        for t, test_t, col, type_ in (("pathology_report", "pathology", "pathology_id", "laboratory"),
                                      ("radiology_report", "radio", "radiology_id", "radiology")):
            rows = [r for r in self.table(t) if (t, sid(r["id"])) not in matched]
            out[t] = {"on a billed charge (result only)": sum(1 for k in matched if k[0] == t), **self.per_visit(
                rows, "evaluation", "investigations",
                lambda r: self.source_visit(r.get("patient_id"), r.get("reporting_date") or r.get("created_at"),
                                            opd_id=r.get("opd_id"), ipd_id=r.get("ipd_id"), window="lab"),
                lambda r, vid, pid, sv, t=t, test_t=test_t, col=col, type_=type_: investigation_payload(
                    gid(t, r["id"]), vid, pid, type_=type_, procedure=self.procedure_for(test=(test_t, sid(r.get(col)))),
                    amount=r.get("apply_charge"), when=stamp(r.get("created_at")) or stamp(r.get("reporting_date")),
                    admission_id=self.admission(sv[1]) if sv[0] == "ipd" else None, result=self.report_text(t, r),
                    comments=text_join(("Smart Hospital bill", r.get("bill_no")), ("Reported by", self.staff_name(r.get("generated_by"))))))}
        return out

    def lab_results(self) -> dict:
        inv = self.ids_by_uuid("evaluation", "investigations", fresh=True)
        matched, why, pl = self.report_matches(), Counter(), []
        for t, test_t, col in (("pathology_report", "pathology", "pathology_id"), ("radiology_report", "radio", "radiology_id")):
            for r in self.table(t):
                if not self.wanted(r.get("patient_id")):
                    continue
                m = matched.get((t, sid(r["id"])))
                iid = inv.get(gid("opd_charge", m) if m else gid(t, r["id"]))
                pid = self.patient(r.get("patient_id"))
                if not iid or not pid:
                    why[f"{t}: no investigation in V3"] += 1
                    continue
                when = stamp(r.get("created_at")) or stamp(r.get("reporting_date"))
                test = clean((self.by_id(test_t).get(sid(r.get(col))) or {}).get("test_name"))
                text = self.report_text(t, r)
                by = self.staff_name(r.get("generated_by"))
                pl.append({"uuid": gid(f"{t}-result", r["id"]), "investigation_id": iid, "patient_id": pid,
                           "test_parameter": test, "result_value": (clean(r.get("description")) or "")[:255] or None,
                           "findings": clean(r.get("description")) if t == "radiology_report" else None,
                           "report_text": text, "status": "publish", "result_date": when,
                           "performed_by": self.staff_user(r.get("generated_by")), "performed_by_name": by,
                           "published_on": when, "approved_on": when, "created_at": when, "updated_at": when})
                why[t] += 1
        res = self.post("evaluation", "investigation_results", pl)
        return {**why, **res}

    # ── inventory ──
    def store(self) -> tuple[int, str]:
        """org 96's Pharmacy store (staff made it at onboarding)."""
        if "store" not in self._cache:
            st = [s for s in self.c.read_all("inventory", "store") if not s.get("deleted_at")]
            hit = next((s for s in st if key(s.get("name")) == "pharmacy"), None) or \
                next((s for s in st if s.get("type") == "pharmacy"), None)
            if not hit:
                raise RuntimeError("org 96 has no Pharmacy store")
            self._cache["store"] = (hit["id"], hit.get("name"))
        return self._cache["store"]

    def latest_batch(self) -> dict[str, dict]:
        if "latest_batch" not in self._cache:
            out = {}
            for b in self.table("medicine_batch_details"):
                out[sid(b.get("pharmacy_id"))] = b                 # ordered by id: the newest wins
            self._cache["latest_batch"] = out
        return self._cache["latest_batch"]

    def product_reuse(self) -> dict[str, int]:
        """item id → an org-96 product with exactly its name (only when one product has that name, and one item)."""
        if "reuse" not in self._cache:
            ours = {gid("item", r["id"]) for r in self.table("pharmacy")}
            by = defaultdict(list)
            for p in self.c.read_all("inventory", "product", workers=8):
                if not p.get("deleted_at") and str(p.get("uuid")).lower() not in ours:
                    by[token_words(p.get("generic_name"))].append(p["id"])
            names = Counter(token_words(r.get("medicine_name")) for r in self.table("pharmacy"))
            self._cache["reuse"] = {sid(r["id"]): by[n][0] for r in self.table("pharmacy")
                                    if (n := token_words(r.get("medicine_name"))) and len(by.get(n, [])) == 1 and names[n] == 1}
        return self._cache["reuse"]

    def products(self) -> dict:
        reuse, latest = self.product_reuse(), self.latest_batch()
        cat = self.ref("inventory", "product_category", "Pharmaceuticals", {"is_drug": 1})
        unit = self.ref("inventory", "unit", "Unit")
        pl = []
        for r in self.table("pharmacy"):
            if sid(r["id"]) in reuse:
                continue
            b = latest.get(sid(r["id"])) or {}
            sell = num(b.get("sale_rate")) or num(b.get("mrp"))
            cost = num(b.get("purchase_price"))
            pack = num(r.get("unit_packing"))
            # items from 2021 hold numbers in these boxes ('450316') and 'No Supplier' as the company
            info = {k: v for k in ("medicine_composition", "medicine_company", "medicine_group")
                    if (v := clean(r.get(k))) and not v.isdigit() and token_words(v) != "no supplier"}
            pl.append({"uuid": gid("item", r["id"]), "generic_name": (clean(r.get("medicine_name")) or f"Item {r['id']}")[:255],
                       "code": f"BCH-I{r['id']}", "item_no": f"BCH-I{r['id']}", "category_id": cat, "unit_id": unit,
                       "therapeutic_class": info.get("medicine_group"),
                       "description": text_join(("Composition", info.get("medicine_composition")),
                                                ("Company", info.get("medicine_company")), ("Group", info.get("medicine_group")),
                                                ("Note", r.get("note"))),
                       "pack_size": int(pack) if pack and pack >= 1 else 1,
                       "reorder_level": int(num(r.get("reorder_level")) or 0) or None,
                       "selling_price": sell, "insurance_price": sell, "cost_price": cost, "trade_price": cost,
                       "is_active": 0 if words(r.get("is_active")) == "no" else 1, "track_batch": 1, "track_expiry": 1,
                       "created_at": stamp(r.get("created_at")), "updated_at": stamp(r.get("created_at"))})
        return {"items": len(self.table("pharmacy")), "reused (same name in org 96)": len(reuse),
                **self.post("inventory", "product", pl)}

    def item_product(self, item_id) -> int | None:
        i = sid(item_id)
        if not i:
            return None
        return self.product_reuse().get(i) or self.ids_by_uuid("inventory", "product").get(gid("item", i))

    def stock(self) -> dict:
        self.ids_by_uuid("inventory", "product", fresh=True)
        st, why, batches = self.store(), Counter(), []
        for b in self.table("medicine_batch_details"):
            q = num(b.get("available_quantity")) or 0
            if q <= 0:
                continue
            pid = self.item_product(b.get("pharmacy_id"))
            if not pid:
                why["item deleted from Smart Hospital (stock not loaded)"] += 1
                continue
            cost, sell = num(b.get("purchase_price")) or 0.0, num(b.get("sale_rate")) or num(b.get("mrp"))
            batches.append((b, pid, q, cost, sell))
        res = {"batches": dict(self.post("inventory", "batch", (
            {"uuid": gid("batch", b["id"]), "batch_no": (clean(b.get("batch_no")) or f"BCH-B{b['id']}")[:100], "product_id": pid,
             "store_id": st[0], "expiry_date": month_expiry(b.get("expiry_date")) or day(b.get("expiry_date_format")),
             "cost_price": cost, "selling_price": sell, "quantity_received": num(b.get("quantity")) or q, "quantity_available": q,
             "status": "active", "invoice_no": clean(b.get("supplier_bill_basic_id")),
             "notes": f"Smart Hospital batch {b['id']}, received {day(b.get('inward_date')) or '?'}",
             "created_at": stamp(b.get("created_at")), "updated_at": stamp(b.get("created_at"))}
            for b, pid, q, cost, sell in batches)))}
        bids = self.ids_by_uuid("inventory", "batch", fresh=True)
        res["stock"] = dict(self.post("inventory", "stock", (
            {"uuid": gid("stock", b["id"]), "product_id": pid, "store_id": st[0], "batch_id": bids.get(gid("batch", b["id"])),
             "quantity": q, "available_quantity": q, "reserved_quantity": 0, "average_cost": cost,
             "total_value": round(q * cost, 2), "selling_price": sell}
            for b, pid, q, cost, sell in batches)))
        return {"batches with stock": len(batches), **why, **res}

    def prescriptions(self) -> dict:
        self.ids_by_uuid("inventory", "product", fresh=True)
        self.ids_by_uuid("inpatient", "admission", fresh=True)
        by_name = {}
        for r in self.table("pharmacy"):
            by_name.setdefault(token_words(r.get("medicine_name")), sid(r["id"]))
        sales = defaultdict(list)                  # patient → sale days
        for b in self.table("pharmacy_bill_basic"):
            if day(b.get("date")) and sid(b.get("patient_id")):
                sales[sid(b["patient_id"])].append(day(b["date"]))
        store = self.store()

        def bought(pid, d) -> bool:
            if not d:
                return False
            hi = (date.fromisoformat(d) + timedelta(days=2)).isoformat()
            return any(d <= s <= hi for s in sales.get(pid, []))
        out = {"doctor_orders": self.per_visit(
            self.table("prescription"), "evaluation", "prescriptions",
            lambda r: self.source_visit(None, None, opd_id=r.get("opd_id"), ipd_id=r.get("ipd_id")),
            lambda r, vid, pid, sv: prescription_payload(
                gid("prescription", r["id"]), vid, drug=self.item_product(by_name.get(token_words(r.get("medicine")))),
                drug_name=r.get("medicine"), dosage=r.get("manual_input_dosage") or r.get("dosage"),
                notes=text_join(("Instructions", r.get("instruction"))),
                when=stamp(r.get("datetime")) or self.visit_stamp(sv),
                dispensed=bought(self.source_patient(sv), day(r.get("datetime"))),
                admission_id=self.admission(sv[1]) if sv[0] == "ipd" else None, user=(self.visit_doctor(sv) or (None,))[0]))}
        basic = self.by_id("ipd_prescription_basic")
        out["inpatient_orders"] = self.per_visit(
            self.table("ipd_prescription_details"), "evaluation", "prescriptions",
            lambda r: self.source_visit(None, None, ipd_id=r.get("ipd_id")),
            lambda r, vid, pid, sv: prescription_payload(
                gid("ipd_prescription", r["id"]), vid, drug=self.item_product(by_name.get(token_words(r.get("medicine")))),
                drug_name=r.get("medicine") or r.get("instruction"), dosage=r.get("manual_input_dosage") or r.get("dosage"),
                notes=text_join(("Instructions", r.get("instruction"))),
                when=stamp((basic.get(sid(r.get("basic_id"))) or {}).get("date")) or self.visit_stamp(sv),
                dispensed=True, admission_id=self.admission(sv[1]), user=(self.visit_doctor(sv) or (None,))[0]))
        items, bills = self.by_id("pharmacy"), self.by_id("pharmacy_bill_basic")

        def sale_rows():
            for d in self.table("pharmacy_bill_detail"):
                b = bills.get(sid(d.get("pharmacy_bill_basic_id")))
                if b:
                    yield {**d, "_bill": b}

        def sale(r, vid, pid, sv):
            b, item = r["_bill"], sid(r.get("medicine_name"))
            name = clean((items.get(item) or {}).get("medicine_name")) or f"Item #{item} (removed from Smart Hospital's item list)"
            return prescription_payload(
                gid("pharmacy_bill_detail", r["id"]), vid, drug=self.item_product(item) if item in items else None,
                drug_name=name, dosage=None, quantity=r.get("quantity"), price=r.get("sale_price"),
                notes=text_join(("Pharmacy bill", b.get("bill_no") or b.get("id")), ("Batch", r.get("batch_no")),
                                ("Expiry", r.get("expire_date")), ("Amount", r.get("amount")), ("Doctor", b.get("doctor_name"))),
                when=stamp(b.get("date")), dispensed=True, store=store,
                admission_id=self.admission(sv[1]) if sv[0] == "ipd" else None)
        out["pharmacy_sales"] = self.per_visit(
            sale_rows(), "evaluation", "prescriptions",
            lambda r: self.source_visit(r["_bill"].get("patient_id"), r["_bill"].get("date"), opd_id=r["_bill"].get("opd_id"),
                                        ipd_id=r["_bill"].get("ipd_id"), window="sale"), sale)
        return out

    # ── money ──
    def bill_visit(self, bill_no) -> tuple[int, int] | None:
        m = self.by_id("master_bill", "bill_no").get(sid(bill_no)) or {}
        for kind in ("ipd", "opd"):
            if sid(m.get(f"{kind}_id")):
                v = self.v3_visit((kind, sid(m[f"{kind}_id"])))
                if v:
                    return v
        return None

    def payments(self) -> dict:
        cards, invoices, why = self.patient_card(), self.by_id("invoice", "invoice_id"), Counter()

        def gen():
            for r in self.table("invoicepayment", "payment_id"):
                if not self.wanted(r.get("patient_id")):
                    continue
                inv = invoices.get(sid(r.get("invoice_id"))) or {}
                v = self.bill_visit(inv.get("bill_no"))
                pid = (v[1] if v else None) or self.patient(r.get("patient_id"))
                p = payment_payload(r, patient=cards.get(pid), visit_id=v[0] if v else None, user=self.c.user_id)
                if p is None:
                    why["insurer paid all of it (on the invoice, no receipt)"] += 1
                    continue
                why["no visit (kept, patient-level)"] += v is None
                yield p
        res = self.post("finance", "evaluation_payment", gen())
        return {**why, **res}

    def payment_details(self) -> dict:
        paid = self.ids_by_uuid("finance", "evaluation_payment", fresh=True)
        invoices = self.by_id("invoice", "invoice_id")

        def gen():
            for r in self.table("invoicepayment", "payment_id"):
                pid = paid.get(gid("invoicepayment", r["payment_id"]))
                if not pid:
                    continue
                inv = invoices.get(sid(r.get("invoice_id"))) or {}
                v = self.bill_visit(inv.get("bill_no"))
                a = round((num(r.get("cash_amount")) or 0) + (num(r.get("mpesa_amount")) or 0), 2)
                when = epoch_wall(r.get("timestamp")) or stamp(r.get("date"))
                yield {"uuid": gid("payment-detail", r["payment_id"]), "payment": pid, "visit": v[0] if v else None,
                       "price": a, "amount": a, "name": (clean(inv.get("description")) or f"Payment BCH-R{r['payment_id']}")[:255],
                       "description": text_join(("Method", r.get("method")), ("Invoice", inv.get("bill_no"))),
                       "created_at": when, "updated_at": when}
        return self.post("finance", "evaluation_payment_detail", gen())

    def invoices(self) -> dict:
        pays = defaultdict(list)
        for p in self.table("invoicepayment", "payment_id"):
            pays[sid(p.get("invoice_id"))].append(p)
        bills, cards, why = self.by_id("master_bill", "bill_no"), self.patient_card(), Counter()
        nos = Counter(sid(i.get("bill_no")) for i in self.table("invoice", "invoice_id"))

        def gen():
            for i in self.table("invoice", "invoice_id"):
                if not self.wanted(i.get("patient_id")):
                    continue
                ps = pays.get(sid(i["invoice_id"]), [])
                ins_amt = sum((num(p.get("nhif_amount")) or 0) + (num(p.get("second_insurance_amount")) or 0) for p in ps)
                insurers = [n for p in ps for n in method_insurers(p.get("method"))] + \
                    [self.org_insurer(p.get("second_insurance_id")) for p in ps if (num(p.get("second_insurance_amount")) or 0) > 0]
                insurers = [n for n in insurers if n]
                unpaid = words(i.get("status")) != "paid"
                if not (ins_amt > 0 or insurers) and not unpaid:
                    why["cash, paid (the receipt is the record)"] += 1
                    continue
                m = bills.get(sid(i.get("bill_no"))) or {}
                total = num(m.get("total")) if m else None
                total = total if total is not None else (num(i.get("amount")) or 0.0)
                paid = num(m.get("paid")) if m else sum(num(p.get("amount")) or 0 for p in ps)
                if total <= 0 and not paid:
                    why["empty bill (nothing charged or paid)"] += 1
                    continue
                bal = round(max(total - (paid or 0), 0.0), 2)
                status = "paid" if bal <= 0 and total else "partially_paid" if paid else "pending"
                v = self.bill_visit(i.get("bill_no"))
                pid = (v[1] if v else None) or self.patient(i.get("patient_id"))
                card = cards.get(pid) or (None, None, None)
                ins = self.insurer_tuple(insurers[0]) if insurers else None
                when = stamp(i.get("date"))
                no = sid(i.get("bill_no")) or sid(i["invoice_id"])
                why["insurer" if ins else "cash, unpaid"] += 1
                yield {"uuid": gid("invoice", i["invoice_id"]),
                       "invoice_no": f"BCH-INV-{no}" if nos[sid(i.get('bill_no'))] == 1 else f"BCH-INV-{no}-{i['invoice_id']}",
                       "invoice_date": (when or self.today)[:10], "invoice_type": "inpatient" if sid(m.get("ipd_id")) else "outpatient",
                       "for_cash": 0 if ins else 1, "patient_id": pid, "patient_name": card[1], "patient_no": card[2],
                       "visit_id": v[0] if v else None, "visit": v[0] if v else None, "company_id": ins[0] if ins else None,
                       "subtotal": total, "total_amount": total, "amount": total, "paid_amount": paid or 0, "paid": paid or 0,
                       "balance_amount": bal, "status": status, "payment_status": status, "source": "betacare_migration",
                       "notes": text_join(("Smart Hospital", i.get("description")), ("Insurers", ", ".join(dict.fromkeys(insurers))),
                                          ("Insurer paid", round(ins_amt, 2) if ins_amt else None)),
                       "created_by": self.c.user_id, "created_at": when, "updated_at": when}
        return {**self.post("finance", "invoice", gen()), **why}

    def appointments(self) -> dict:
        why = Counter()

        def gen():
            for r in self.table("appointment"):
                if not self.wanted(r.get("patient_id")):
                    continue
                pid = self.patient(r.get("patient_id"))
                when = stamp(r.get("date"))
                if not pid or not when:
                    why["no patient / date"] += 1
                    continue
                done = words(r.get("appointment_status")) == "approved"
                why[r.get("appointment_status") or "?"] += 1
                yield {"uuid": gid("appointment", r["id"]), "appointment_no": f"BCH-{clean(r.get('appointment_no')) or r['id']}",
                       "patient_id": pid, "doctor_id": self.staff_user(r.get("doctor")), "doctor_name": self.staff_name(r.get("doctor")),
                       "appointment_date": when[:10], "appointment_time": when[11:], "reason": clean(r.get("message")),
                       "notes": text_join(("Source", r.get("source")), ("Smart Hospital status", r.get("appointment_status"))),
                       "status": "completed" if done and when[:10] < self.today else "scheduled",
                       "created_at": stamp(r.get("created_at")) or when, "updated_at": stamp(r.get("created_at")) or when}
        res = self.post("reception", "appointment", gen())
        return {**why, **res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default="betacare_prod", choices=TARGETS)
    ap.add_argument("--steps", nargs="*", default=list(STEPS), choices=STEPS)
    ap.add_argument("--limit", type=int, default=0, help="post at most N rows per step (canary)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-missing", action="store_true", help="post only rows whose uuid V3 doesn't have yet")
    ap.add_argument("--patients", nargs="*", help="canary: only these Smart Hospital patient ids (and their visits, orders, money)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = Run(a.target, a.dry_run, a.limit, a.workers, a.only_missing)
    run.only_patients = set(a.patients) if a.patients else None
    report = {"target": a.target, "org": run.c.organization_id, "facility": run.c.facility_id, "dry_run": a.dry_run,
              "limit": a.limit, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
    out = Path(__file__).resolve().parent / "output" / "betacare_to_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.target}-{time.strftime('%Y%m%d_%H%M%S')}.json"
    for s in STEPS:
        if s in a.steps:
            t0 = time.time()
            try:
                report["steps"][s] = getattr(run, s)()
            except Exception as e:
                log.exception("%s failed", s)
                report["steps"][s] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            report["steps"][s]["seconds"] = round(time.time() - t0)
            log.info("%s: %s", s, json.dumps(report["steps"][s], default=str))
            report["failures"] = {k: v[:200] for k, v in run.failures.items()}
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("report: %s", path)
    sys.exit(1 if any("error" in r for r in report["steps"].values()) else 0)


if __name__ == "__main__":
    main()
