"""
Seeds curated MetricDefinition rows for the clinical/patient domain (OPD-IPD
conversion, case mix, clinical activity/IPD, OPD revisits, retention/LTFU,
chronic disease, maternal health, infectious disease, HAI/SSI) — the 9 cubes
added for the 100-question clinical evaluation test suite.

Unlike the hospital-ops/pharmacy domains, these cubes have no pre-existing
competing catalog metric, so one comprehensive metric per cube (all its
measures + dimensions declared) is enough — there's nothing else already
indexed to lose a retrieval-score contest against.

Idempotent — safe to re-run. Usage:
    python manage.py seed_clinical_metrics
Then:
    python manage.py rebuild_embeddings
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from agents.models import MetricDefinition

METRICS: list[dict] = [
    {
        "metric_id": "opd_ipd_conversion_metrics",
        "name": "OPD to IPD Conversion and 72-Hour Escalation",
        "description": (
            "OPD-to-IPD conversion rate and 72-hour escalation rate, by "
            "age_group, gender, payment_mode, is_comorbidity, and "
            "visit_month. Use for 'OPD to IPD conversion rate', 'which age "
            "group has the highest direct conversion count', 'are insured "
            "patients more likely to convert/escalate than cash patients', "
            "'total OPD visits', 'total direct IPD conversions', 'what "
            "proportion of OPD visits are chronic' questions. total_visits/"
            "distinct_patients/visit_conversions/visit_escalations are "
            "breakdown-level (sum by any dimension); total_opd_visits/"
            "direct_ipd_conversions/escalations_72h/conversion_rate_pct/"
            "escalation_rate_pct are facility-month totals (already "
            "aggregated correctly per month, do not need further summing)."
        ),
        "cube_query": {
            "measures": [
                "rpt_opd_ipd.total_visits",
                "rpt_opd_ipd.distinct_patients",
                "rpt_opd_ipd.visit_conversions",
                "rpt_opd_ipd.visit_escalations",
                "rpt_opd_ipd.total_opd_visits",
                "rpt_opd_ipd.direct_ipd_conversions",
                "rpt_opd_ipd.escalations_72h",
                "rpt_opd_ipd.conversion_rate_pct",
                "rpt_opd_ipd.escalation_rate_pct",
                "rpt_opd_ipd.chronic_opd_pct",
            ],
            "dimensions": [
                "rpt_opd_ipd.age_group",
                "rpt_opd_ipd.gender",
                "rpt_opd_ipd.payment_mode",
                "rpt_opd_ipd.is_comorbidity",
                "rpt_opd_ipd.clean_diagnosis",
            ],
            "timeDimensions": [{"dimension": "rpt_opd_ipd.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "case_mix_metrics",
        "name": "Case Mix — Visits by Diagnosis, Burden Group, Demographics",
        "description": (
            "Visit counts and comorbidity/classification rates broken down "
            "by clean_diagnosis, final_disease_burden_group, age_group, "
            "gender, payment_mode, visit_type, and comorbidity_pair. Use "
            "for 'total visit count', 'which diagnosis has the most "
            "visits', 'which disease burden group has the highest "
            "visits', 'which age group accounts for the most visits', "
            "'percentage of visits with a comorbidity recorded', 'do "
            "inpatient visits have a higher comorbidity rate', 'most "
            "common comorbidity pair', 'proportion classified from ICD-10 "
            "vs doctor notes', 'which payment mode accounts for the most "
            "visits', 'proportion of unclassified visits', 'total distinct "
            "patient count', 'overall visit volume trend' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_case_mix.total_visits",
                "rpt_case_mix.distinct_patients",
                "rpt_case_mix.comorbid_visits",
                "rpt_case_mix.comorbidity_rate_pct",
                "rpt_case_mix.doctor_note_only_visits",
                "rpt_case_mix.doctor_note_only_pct",
                "rpt_case_mix.unclassified_visits",
                "rpt_case_mix.unclassified_pct",
            ],
            "dimensions": [
                "rpt_case_mix.clean_diagnosis",
                "rpt_case_mix.final_disease_burden_group",
                "rpt_case_mix.classification_source",
                "rpt_case_mix.visit_type",
                "rpt_case_mix.age_group",
                "rpt_case_mix.gender",
                "rpt_case_mix.payment_mode",
                "rpt_case_mix.comorbidity_pair",
            ],
            "timeDimensions": [{"dimension": "rpt_case_mix.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "clinical_activity_metrics",
        "name": "Inpatient Clinical Activity — Admissions, LOS, Readmissions",
        "description": (
            "Admissions, discharges, 30-day readmission rate, and average/"
            "median/P25/P75 length of stay, by ward_category, ward_name, "
            "clean_diagnosis, age_group, is_comorbidity, and "
            "admission_month. Use for 'total discharges', 'overall 30-day "
            "readmission rate', 'which ward category/diagnosis/age group "
            "has the highest readmission rate', 'are comorbid patients "
            "readmitted more often', 'average length of stay', 'which "
            "ward has the longest average LOS', 'LOS IQR', 'are there HAI "
            "or SSI cases', 'proportion of readmissions within 7 days' "
            "questions. Use avg_los_days_calc (not the raw stored column) "
            "and readmission_rate_30d_calc_pct (not readmission_rate_30d_pct) "
            "for anything grouped or filtered — those are the "
            "correctly-weighted versions."
        ),
        "cube_query": {
            "measures": [
                "rpt_clinical_activity.total_admissions",
                "rpt_clinical_activity.total_discharges",
                "rpt_clinical_activity.readmissions",
                "rpt_clinical_activity.readmissions_30day",
                "rpt_clinical_activity.readmission_rate_30d_calc_pct",
                "rpt_clinical_activity.avg_los_days_calc",
                "rpt_clinical_activity.median_los_days",
                "rpt_clinical_activity.p25_los_days",
                "rpt_clinical_activity.p75_los_days",
                "rpt_clinical_activity.hai_ssi_count",
            ],
            "dimensions": [
                "rpt_clinical_activity.ward_category",
                "rpt_clinical_activity.ward_name",
                "rpt_clinical_activity.clean_diagnosis",
                "rpt_clinical_activity.age_group",
                "rpt_clinical_activity.is_comorbidity",
                "rpt_clinical_activity.readmission_gap_bucket",
            ],
            "timeDimensions": [{"dimension": "rpt_clinical_activity.admission_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "opd_revisits_metrics",
        "name": "OPD 7-Day Revisits",
        "description": (
            "7-day OPD revisit counts and revisit rate, by clean_diagnosis, "
            "age_group, gender, is_comorbidity, revisit_day_bucket, and "
            "visit_month. Use for 'overall 7-day OPD revisit rate', 'which "
            "diagnosis has the highest 7-day revisit count', 'proportion "
            "of revisits within the first 2 days', 'do comorbid patients "
            "revisit more often', 'are female patients more likely to "
            "revisit' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_opd_revisits.revisit_count",
                "rpt_opd_revisits.distinct_patients",
                "rpt_opd_revisits.revisit_rate_pct",
            ],
            "dimensions": [
                "rpt_opd_revisits.clean_diagnosis",
                "rpt_opd_revisits.age_group",
                "rpt_opd_revisits.gender",
                "rpt_opd_revisits.is_comorbidity",
                "rpt_opd_revisits.revisit_day_bucket",
            ],
            "timeDimensions": [{"dimension": "rpt_opd_revisits.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "retention_ltfu_metrics",
        "name": "Chronic Patient Retention and LTFU (Lost to Follow-Up)",
        "description": (
            "Distinct chronic patient counts and LTFU (lost-to-follow-up, "
            "180+ days) / lapsing (91-180 days) rates, by lifecycle_status, "
            "last_burden_group, payment_mode, last_clinician_user, "
            "exit_visit_bucket, gender, age_group, and "
            "primary_chronic_disease_group. Use for 'how many chronic "
            "patients fall into each lifecycle bucket', 'LTFU rate for "
            "chronic patients', 'which disease burden group/payment mode/"
            "age group has the highest LTFU rate', 'which clinician has "
            "the most LTFU patients', 'at what visit count do patients "
            "drop out', 'is LTFU higher among female patients', "
            "'percentage in the lapsing window' questions. Filter "
            "is_chronic=1 (nearly every question in this domain is scoped "
            "to chronic patients)."
        ),
        "cube_query": {
            "measures": [
                "rpt_retention.total_patients",
                "rpt_retention.ltfu_patients",
                "rpt_retention.ltfu_rate_pct",
                "rpt_retention.lapsing_patients",
                "rpt_retention.lapsing_pct",
            ],
            "dimensions": [
                "rpt_retention.lifecycle_status",
                "rpt_retention.last_burden_group",
                "rpt_retention.last_clean_diagnosis",
                "rpt_retention.payment_mode",
                "rpt_retention.last_clinician_user",
                "rpt_retention.exit_visit_bucket",
                "rpt_retention.gender",
                "rpt_retention.age_group",
                "rpt_retention.primary_chronic_disease_group",
                "rpt_retention.is_chronic",
            ],
            "timeDimensions": [],
            "filters": [{"member": "rpt_retention.is_chronic", "operator": "equals", "values": ["1"]}],
            "limit": 500,
        },
    },
    {
        "metric_id": "chronic_disease_metrics",
        "name": "Chronic Disease Management (Hypertension, Diabetes, etc.)",
        "description": (
            "Chronic disease visit counts, comorbidity rate, and blood-"
            "pressure control status (Controlled/Uncontrolled/Hypertensive "
            "Crisis) for hypertensive patients, by "
            "primary_chronic_disease_group, age_group, gender, "
            "bp_control_status, and visit_month. Use for 'distinct chronic "
            "patients seen', 'chronic disease visit volume trend', 'which "
            "chronic disease group has the most patients', 'percentage of "
            "hypertensive patients with controlled BP', 'are uncontrolled "
            "hypertensive patients skewed to an age group', 'comorbidity "
            "rate among chronic disease patients', 'hypertension visit "
            "volume trend', 'most common drug regimen for uncontrolled "
            "hypertension', 'average systolic BP', 'hypertensive crisis "
            "rate by age group' questions. Filter "
            "primary_chronic_disease_group with 'contains' \"Hypertension\" "
            "for hypertension-specific questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_chronic_disease.total_visits",
                "rpt_chronic_disease.distinct_patients",
                "rpt_chronic_disease.visits_with_comorbidity",
                "rpt_chronic_disease.comorbidity_rate_pct",
                "rpt_chronic_disease.crisis_visits",
                "rpt_chronic_disease.crisis_rate_pct",
                "rpt_chronic_disease.avg_bp_systolic",
                "rpt_chronic_disease.avg_bp_diastolic",
                "rpt_chronic_disease.avg_bmi",
            ],
            "dimensions": [
                "rpt_chronic_disease.primary_chronic_disease_group",
                "rpt_chronic_disease.age_group",
                "rpt_chronic_disease.gender",
                "rpt_chronic_disease.bp_control_status",
                "rpt_chronic_disease.modal_drug_regimen",
                "rpt_chronic_disease.comorbidity_pair",
            ],
            "timeDimensions": [{"dimension": "rpt_chronic_disease.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "maternal_health_metrics",
        "name": "Maternal Health — ANC, Complications, Pregnancy Loss",
        "description": (
            "Maternal visit counts, ANC adherence bands, investigations, "
            "and prescription revenue, by maternal_case_type (ANC, "
            "COMPLICATION / HIGH RISK, PREGNANCY LOSS), age_group, and "
            "visit_month. Use for 'total maternal visits', 'ANC visit "
            "volume trend', 'which maternal case type has the highest "
            "volume', 'percentage of ANC patients in each adherence band', "
            "'proportion of single-visit ANC patients', 'which age group "
            "has the most complication/high-risk visits', 'total "
            "investigations ordered', 'pregnancy loss case volume trend', "
            "'total prescription revenue from maternal visits', "
            "'proportion of total visits that are maternal-related' "
            "questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_maternal.total_visits",
                "rpt_maternal.distinct_patients",
                "rpt_maternal.visits_with_comorbidity",
                "rpt_maternal.single_visit_patients",
                "rpt_maternal.single_visit_pct",
                "rpt_maternal.total_investigations",
                "rpt_maternal.total_investigation_revenue",
                "rpt_maternal.total_prescriptions",
                "rpt_maternal.total_prescription_revenue",
            ],
            "dimensions": [
                "rpt_maternal.maternal_case_type",
                "rpt_maternal.age_group",
                "rpt_maternal.anc_adherence_band",
                "rpt_maternal.clean_diagnosis",
                "rpt_maternal.comorbidity_pair",
            ],
            "timeDimensions": [{"dimension": "rpt_maternal.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "infectious_disease_metrics",
        "name": "Infectious Disease — Malaria, URTI, Sepsis",
        "description": (
            "Infectious disease visit counts and comorbidity rate, by "
            "disease_group (malaria/URTI/sepsis — use 'contains'), "
            "age_group, gender, and visit_month. Use for 'total "
            "infectious disease visits', 'which disease group has the "
            "highest visits', 'malaria/URTI/sepsis case volume trend', "
            "'seasonal pattern in URTI cases', 'which age group has the "
            "highest infectious disease visits', 'are male patients more "
            "affected than female', 'comorbidity rate among infectious "
            "disease patients' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_infectious_disease.total_visits",
                "rpt_infectious_disease.distinct_patients",
                "rpt_infectious_disease.hai_ssi_visits",
                "rpt_infectious_disease.comorbid_visits",
                "rpt_infectious_disease.comorbidity_rate_pct",
            ],
            "dimensions": [
                "rpt_infectious_disease.disease_group",
                "rpt_infectious_disease.final_disease_burden_group",
                "rpt_infectious_disease.age_group",
                "rpt_infectious_disease.gender",
            ],
            "timeDimensions": [{"dimension": "rpt_infectious_disease.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "hai_ssi_infection_metrics",
        "name": "HAI/SSI Hospital-Acquired and Surgical Site Infections",
        "description": (
            "Confirmed HAI (hospital-acquired infection) and SSI "
            "(surgical site infection) visit counts by clean_diagnosis. "
            "Use for 'leading HAI infection type', 'leading SSI type', "
            "'full ranked breakdown of HAI and SSI infections' questions. "
            "Filter hai_ssi_flag = 'HAI' or 'SSI'."
        ),
        "cube_query": {
            "measures": ["rpt_diagnosis_reference.confirmed_visits"],
            "dimensions": [
                "rpt_diagnosis_reference.hai_ssi_flag",
                "rpt_diagnosis_reference.clean_diagnosis",
            ],
            "timeDimensions": [],
            "filters": [],
            "limit": 500,
        },
    },
]


class Command(BaseCommand):
    help = "Seed/update curated MetricDefinition rows for the clinical/patient domain (batch 3)."

    def handle(self, *args, **options):
        created, updated = 0, 0
        for spec in METRICS:
            _obj, was_created = MetricDefinition.objects.update_or_create(
                metric_id=spec["metric_id"],
                defaults={
                    "name": spec["name"],
                    "description": spec["description"],
                    "cube_query": spec["cube_query"],
                    "is_active": True,
                },
            )
            created += was_created
            updated += not was_created

        self.stdout.write(self.style.SUCCESS(
            f"Seeded {len(METRICS)} curated clinical metrics ({created} created, {updated} updated). "
            f"Run 'python manage.py rebuild_embeddings' next so they're searchable."
        ))
