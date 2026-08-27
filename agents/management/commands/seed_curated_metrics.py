"""
Seeds a batch of precisely-scoped MetricDefinition rows for domains that were
previously only reachable through the bare-measure retrieval fallback (which
loses to broad, pre-existing catalog metrics in embedding search — see the
KSH/TENRI and pharmacy/dispensing test-suite runs this was built from).

Each entry here targets ONE tested concept (e.g. "admission TAT", "stockout
rate"), not a whole cube, with a description written to match how the real
test questions are phrased. This is the durable fix for retrieval repeatedly
picking a wrong-but-broader metric over the right-but-uncurated one: a
curated "metric" source embeds as clean name+description text (no cube/field
prefix noise), the same way glossary entries already do, so it scores
reliably above CONFIDENT_SINGLE_SCORE instead of hovering just under it as a
bare measure.

Idempotent — safe to re-run. Uses update_or_create so re-running after
tweaking a description below simply updates the existing row.

Usage:
    python manage.py seed_curated_metrics

After running this, you MUST also run `python manage.py rebuild_embeddings`
so these new metrics are actually searchable.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from agents.models import MetricDefinition

METRICS: list[dict] = [
    # ── Hospital operations ─────────────────────────────────────────────
    {
        "metric_id": "bed_occupancy_payment_mix_metrics",
        "name": "Ward Admissions by Payment Mode (Insured vs Cash)",
        "description": (
            "Insured vs cash admission counts and revenue per ward, ward category, "
            "and facility, by admission month. Use for 'are there wards where "
            "insured admissions exceed cash admissions' and similar insured-vs-cash "
            "comparison questions on BED OCCUPANCY (rpt_bed_occupancy) — NOT the "
            "same-named fields on rpt_specialty_admissions, a different cube."
        ),
        "cube_query": {
            "measures": [
                "rpt_bed_occupancy.insured_admissions",
                "rpt_bed_occupancy.cash_admissions",
                "rpt_bed_occupancy.insured_revenue",
                "rpt_bed_occupancy.cash_revenue",
                "rpt_bed_occupancy.total_admissions",
            ],
            "dimensions": [
                "rpt_bed_occupancy.facility",
                "rpt_bed_occupancy.ward_category",
                "rpt_bed_occupancy.ward_name",
            ],
            "timeDimensions": [{"dimension": "rpt_bed_occupancy.admission_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "avg_los_trend_metrics",
        "name": "Average Length of Stay",
        "description": (
            "Average length of stay (LOS) in days for admitted patients, by facility, "
            "ward category, ward name, and admission month. Use for questions like "
            "'what is the average length of stay', 'has average LOS been increasing or "
            "decreasing', 'average LOS trend over the last N months', broken down by "
            "facility or ward. Distinct from ward-level MEDIAN LOS (see Ward Length of Stay)."
        ),
        "cube_query": {
            "measures": [
                "rpt_bed_occupancy.avg_los_days",
                "rpt_bed_occupancy.total_bed_days",
                "rpt_bed_occupancy.discharged_admissions",
            ],
            "dimensions": [
                "rpt_bed_occupancy.facility",
                "rpt_bed_occupancy.ward_category",
                "rpt_bed_occupancy.ward_name",
            ],
            "timeDimensions": [{"dimension": "rpt_bed_occupancy.admission_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "admission_tat_metrics",
        "name": "Admission Turnaround Time (TAT)",
        "description": (
            "Admission turnaround time in MINUTES (median/P50 and P75), and the "
            "percentage of admissions fast-tracked (TAT under 60 minutes), by month. "
            "KSH-only data (this cube has no facility dimension). Use for 'median "
            "admission TAT', 'is admission TAT improving', 'which month had the lowest "
            "TAT', 'fewer than 50% fast-tracked' questions. Values are MINUTES, not KES."
        ),
        "cube_query": {
            "measures": [
                "rpt_admission_tat.p50_tat_min",
                "rpt_admission_tat.p75_tat_min",
                "rpt_admission_tat.fast_pct",
                "rpt_admission_tat.total_admissions",
                "rpt_admission_tat.fast_track",
            ],
            "dimensions": [],
            "timeDimensions": [{"dimension": "rpt_admission_tat.tat_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "ward_median_los_metrics",
        "name": "Ward Length of Stay (Median)",
        "description": (
            "MEDIAN length of stay in days per ward category, by facility and admission "
            "month. Use for 'median length of stay for the Medical/Maternity ward', "
            "'which ward has the highest median LOS', 'has median LOS exceeded 7 days' "
            "questions. Distinct from the facility-wide AVERAGE LOS trend (see Average "
            "Length of Stay)."
        ),
        "cube_query": {
            "measures": ["rpt_ward_los.median_los_days", "rpt_ward_los.admissions"],
            "dimensions": ["rpt_ward_los.facility", "rpt_ward_los.ward_category"],
            "timeDimensions": [{"dimension": "rpt_ward_los.admission_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "dialysis_ops_metrics",
        "name": "Dialysis Utilisation and Session Mix",
        "description": (
            "Monthly dialysis utilisation rate as a percentage of theoretical capacity, "
            "plus the split of insured vs cash sessions and dialysis revenue. Use for "
            "'current monthly dialysis utilisation rate', 'is dialysis utilisation "
            "improving', 'dropped below 50% of theoretical capacity', 'proportion of "
            "dialysis sessions insured vs cash' questions. Filter is_partial_month=false "
            "for complete months only. KSH only."
        ),
        "cube_query": {
            "measures": [
                "rpt_dialysis_ops.utilisation_pct_theoretical",
                "rpt_dialysis_ops.sessions_billed",
                "rpt_dialysis_ops.sessions_insured",
                "rpt_dialysis_ops.sessions_cash",
                "rpt_dialysis_ops.insured_pct",
                "rpt_dialysis_ops.total_dialysis_revenue",
                "rpt_dialysis_ops.session_fee_revenue",
                "rpt_dialysis_ops.avg_session_fee",
            ],
            "dimensions": ["rpt_dialysis_ops.is_partial_month"],
            "timeDimensions": [{"dimension": "rpt_dialysis_ops.invoice_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "peak_hour_performance_metrics",
        "name": "Peak-Hour Evaluations and Admission Conversion",
        "description": (
            "Emergency department evaluation counts and admission conversion rate, "
            "broken down by time_bucket ('Peak (Mon 14-17h)' vs 'Off-Peak'), doctor "
            "(username), or month. Use for 'how many evaluations during peak hours', "
            "'which doctor handles the most evaluations during peak hours', 'is the "
            "peak-hour admission conversion rate improving', 'are peak-hour patients "
            "more likely to be admitted than off-peak' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_peak_performance.count",
                "rpt_peak_performance.admissions",
                "rpt_peak_performance.conversion_pct",
            ],
            "dimensions": [
                "rpt_peak_performance.time_bucket",
                "rpt_peak_performance.username",
                "rpt_peak_performance.ward_category",
                "rpt_peak_performance.day_name",
            ],
            "timeDimensions": [{"dimension": "rpt_peak_performance.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "patient_return_funnel_metrics",
        "name": "Non-Admitted Peak-Hour Patient Return Funnel",
        "description": (
            "Single-row summary of what happens to non-admitted peak-hour patients "
            "afterward: what percentage return, how many are later admitted on a "
            "return visit, median days to return, and what percentage never return. "
            "Use for 'what percentage of non-admitted peak-hour patients return', "
            "'how many were later admitted on a return visit', 'median days to "
            "return', 'what percentage never return' questions. KSH only."
        ),
        "cube_query": {
            "measures": [
                "rpt_patient_return_funnel.total_non_admitted_peak",
                "rpt_patient_return_funnel.returned",
                "rpt_patient_return_funnel.never_returned",
                "rpt_patient_return_funnel.return_pct",
                "rpt_patient_return_funnel.later_admitted",
                "rpt_patient_return_funnel.admitted_of_returned_pct",
                "rpt_patient_return_funnel.median_days_to_return",
                "rpt_patient_return_funnel.never_returned_pct",
            ],
            "dimensions": [],
            "timeDimensions": [],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "insurance_collection_metrics",
        "name": "Insurance Collection and Dispatch Rate",
        "description": (
            "Insurance AR collection rate (% of billed amounts collected) and invoice "
            "dispatch rate, by facility, insurer, and invoice month. Use for 'overall "
            "insurance collection rate', 'which insurer has the highest collection "
            "rate', 'insurers with a dispatch rate below 80%', 'is outstanding AR "
            "increasing' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_insurance_ar.collection_rate_calc_pct",
                "rpt_insurance_ar.total_billed",
                "rpt_insurance_ar.total_collected",
                "rpt_insurance_ar.total_outstanding",
                "rpt_insurance_ar.dispatch_rate_calc_pct",
                "rpt_insurance_ar.invoices",
                "rpt_insurance_ar.dispatched_invoices",
            ],
            "dimensions": ["rpt_insurance_ar.facility", "rpt_insurance_ar.insurer"],
            "timeDimensions": [{"dimension": "rpt_insurance_ar.invoice_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "ar_aging_metrics",
        "name": "AR Aging by Bucket",
        "description": (
            "Outstanding accounts receivable broken down by aging bucket (e.g. '90+' "
            "days), facility, and insurer, plus overpayment amounts. Use for 'total "
            "outstanding AR in the 90+ day bucket', 'which insurer has the highest "
            "90+ day outstanding balance', 'insurers with more than KES 1M in "
            "overpayments' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_ar_aging.total_outstanding",
                "rpt_ar_aging.overpayment_kes",
                "rpt_ar_aging.overpaid_invoices",
            ],
            "dimensions": ["rpt_ar_aging.facility", "rpt_ar_aging.insurer", "rpt_ar_aging.aging_bucket"],
            "timeDimensions": [{"dimension": "rpt_ar_aging.invoice_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "doctor_conversion_rate_metrics",
        "name": "Doctor Admission Conversion Rate",
        "description": (
            "Per-doctor (and facility-wide) admission conversion rate — the percentage "
            "of a doctor's evaluations that resulted in an admission — plus evaluation "
            "and admission counts, by username and visit month. Use for 'facility-wide "
            "admission conversion rate', 'which doctor has the highest conversion rate', "
            "'doctors with over 100 evaluations but a conversion rate below 5%' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_doctor_performance.conversion_rate_calc_pct",
                "rpt_doctor_performance.evaluations",
                "rpt_doctor_performance.admissions",
            ],
            "dimensions": ["rpt_doctor_performance.username"],
            "timeDimensions": [{"dimension": "rpt_doctor_performance.visit_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "theatre_utilization_metrics",
        "name": "Theatre Utilization",
        "description": (
            "Theatre session completion rate, emergency vs elective session split, "
            "revenue per session, and insured/cash revenue, by facility, theatre name, "
            "theatre type, and session month. Use for 'theatre completion rate', 'how "
            "has theatre revenue trended', 'which theatre has the highest average "
            "session revenue', 'proportion of emergency vs elective sessions' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_theatre_utilization.completion_rate_calc_pct",
                "rpt_theatre_utilization.total_sessions",
                "rpt_theatre_utilization.completed_sessions",
                "rpt_theatre_utilization.emergency_sessions",
                "rpt_theatre_utilization.elective_sessions",
                "rpt_theatre_utilization.emergency_pct",
                "rpt_theatre_utilization.total_revenue",
                "rpt_theatre_utilization.avg_revenue_per_session",
                "rpt_theatre_utilization.insured_revenue",
                "rpt_theatre_utilization.cash_revenue",
            ],
            "dimensions": [
                "rpt_theatre_utilization.facility",
                "rpt_theatre_utilization.theatre_name",
                "rpt_theatre_utilization.theatre_type",
                "rpt_theatre_utilization.booking_status",
            ],
            "timeDimensions": [{"dimension": "rpt_theatre_utilization.session_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "dialysis_sessions_metrics",
        "name": "Dialysis Sessions, Duration, and Revenue",
        "description": (
            "Dialysis session counts, distinct patients, average session duration in "
            "hours, and revenue per session, by facility, ward, and session month. Use "
            "for 'distinct dialysis patients seen last month', 'average dialysis "
            "session duration', 'dialysis revenue per session' questions comparing "
            "facilities like TENRI vs KSH."
        ),
        "cube_query": {
            "measures": [
                "rpt_dialysis.total_sessions",
                "rpt_dialysis.distinct_patients",
                "rpt_dialysis.avg_duration_hrs",
                "rpt_dialysis.total_dialysis_revenue",
                "rpt_dialysis.revenue_per_session",
                "rpt_dialysis.insured_revenue",
                "rpt_dialysis.cash_revenue",
            ],
            "dimensions": ["rpt_dialysis.facility", "rpt_dialysis.dialysis_ward"],
            "timeDimensions": [{"dimension": "rpt_dialysis.session_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "procedure_revenue_metrics",
        "name": "Procedure Revenue by Category and Insurer",
        "description": (
            "Total procedure/billing revenue, broken down by revenue_category (e.g. "
            "Pharmacy, Investigations, Rebate), insurer, payment_mode, and facility. "
            "Use for 'top revenue category', 'pharmacy revenue trend', 'which insurer "
            "contributes the most insured revenue', 'total rebate/contra-revenue "
            "exposure at each facility' questions. Filter revenue_category='Rebate' "
            "for rebate exposure, or payment_mode='insured' for insured-revenue "
            "questions."
        ),
        "cube_query": {
            "measures": ["rpt_procedure_revenue.total_revenue", "rpt_procedure_revenue.avg_item_amount", "rpt_procedure_revenue.line_items"],
            "dimensions": [
                "rpt_procedure_revenue.facility",
                "rpt_procedure_revenue.insurer",
                "rpt_procedure_revenue.payment_mode",
                "rpt_procedure_revenue.revenue_category",
            ],
            "timeDimensions": [{"dimension": "rpt_procedure_revenue.revenue_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "lab_monthly_metrics",
        "name": "Lab Monthly Volume and Abnormal Rate",
        "description": (
            "Total lab components processed and the abnormal lab result rate, by "
            "month. KSH only. Use for 'abnormal lab result rate last month', 'is the "
            "abnormal lab rate increasing', 'which month had the highest total lab "
            "components processed', 'months where abnormal rate exceeded 40%' "
            "questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_lab_monthly.total_components",
                "rpt_lab_monthly.abnormal_pct",
                "rpt_lab_monthly.abnormal_count",
                "rpt_lab_monthly.distinct_visits",
            ],
            "dimensions": [],
            "timeDimensions": [{"dimension": "rpt_lab_monthly.lab_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "imaging_ops_metrics",
        "name": "Imaging Revenue by Modality",
        "description": (
            "Imaging revenue and average revenue per session, by facility, modality "
            "(e.g. 'CT / Angio', 'Ultrasound'), and revenue month. Use for 'total "
            "imaging revenue this month', 'CT/Angio revenue trend', 'which imaging "
            "modality generates the most revenue', 'average ultrasound session "
            "revenue between TENRI and KSH' questions."
        ),
        "cube_query": {
            "measures": ["rpt_imaging_ops.revenue", "rpt_imaging_ops.avg_per_session", "rpt_imaging_ops.sessions"],
            "dimensions": ["rpt_imaging_ops.facility", "rpt_imaging_ops.modality"],
            "timeDimensions": [{"dimension": "rpt_imaging_ops.revenue_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "readmissions_30day_metrics",
        "name": "30-Day Readmissions",
        "description": (
            "30-day readmission rate and counts, by facility, ward category, "
            "discharge type, payment mode, and admission month, plus insured "
            "revenue at risk from 30-day readmissions. Use for 'overall 30-day "
            "readmission rate', 'is the 30-day readmission rate improving', "
            "'which ward category has the highest insured revenue at risk', "
            "'discharge pathways with a readmission rate above 10%' questions."
        ),
        "cube_query": {
            "measures": [
                "rpt_readmissions.total_admissions",
                "rpt_readmissions.readmissions",
                "rpt_readmissions.readmission_rate_calc_pct",
                "rpt_readmissions.readmissions_30day",
                "rpt_readmissions.readmission_30day_rate_calc_pct",
                "rpt_readmissions.insured_30day_revenue_at_risk",
                "rpt_readmissions.avg_days_between_admissions",
            ],
            "dimensions": [
                "rpt_readmissions.facility",
                "rpt_readmissions.discharge_type",
                "rpt_readmissions.payment_mode",
                "rpt_readmissions.ward_category",
            ],
            "timeDimensions": [{"dimension": "rpt_readmissions.admission_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    # ── Pharmacy / dispensing ────────────────────────────────────────────
    {
        "metric_id": "drug_stockout_rate_metrics",
        "name": "Drug Stockout Rate",
        "description": (
            "Stockout FREQUENCY / RATE — the share of dispensing months a product, "
            "therapeutic class, or facility spent at zero stock-on-hand (days at "
            "zero stock divided by total transaction days/months). Use for 'stockout "
            "rate for antimalarials', 'which drugs are stocking out', 'how many days "
            "did we have zero stock last month', 'stockout rate by therapeutic "
            "class', 'which product has the highest stockout rate', 'stockout trends "
            "month-over-month' questions. Break down by canonical_product_taxonomy."
            "canonical_name or therapeutic_class (joined via canonical_product_id), "
            "or by fact_dispensing.source_schema for facility (values are lowercase, "
            "e.g. 'kisumu')."
        ),
        "cube_query": {
            "measures": [
                "fact_dispensing.days_at_zero_stock",
                "fact_dispensing.total_transaction_days",
                "fact_dispensing.stockout_frequency",
            ],
            "dimensions": [
                "fact_dispensing.source_schema",
                "canonical_product_taxonomy.canonical_name",
                "canonical_product_taxonomy.therapeutic_class",
            ],
            "timeDimensions": [{"dimension": "fact_dispensing.dispensing_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "negative_stock_dispensing_metrics",
        "name": "Negative Stock Dispensing Rate",
        "description": (
            "Count and rate of dispense events made from already-negative stock, by "
            "product, therapeutic class, facility, or month. Use for 'which drugs are "
            "dispensed from negative stock', 'how many dispenses from negative stock', "
            "'negative stock rate by product', 'which therapeutic class has the most "
            "negative stock dispenses', 'negative stock trend' questions."
        ),
        "cube_query": {
            "measures": [
                "fact_dispensing.dispensed_from_negative_stock_count",
                "fact_dispensing.negative_stock_rate",
                "fact_dispensing.count",
            ],
            "dimensions": [
                "fact_dispensing.source_schema",
                "canonical_product_taxonomy.canonical_name",
                "canonical_product_taxonomy.therapeutic_class",
            ],
            "timeDimensions": [{"dimension": "fact_dispensing.dispensing_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "drug_consumption_metrics",
        "name": "Drug Consumption (Units Dispensed)",
        "description": (
            "Total units dispensed and dispense-event counts, by product "
            "(canonical_name), therapeutic class/subclass, facility, or month. Use "
            "for 'how many ARVs were dispensed', 'total antimalarial units dispensed "
            "this month', 'which drug was dispensed the most', 'antibiotic "
            "consumption by units', 'total units dispensed by therapeutic class', "
            "'opioid dispenses', 'antimalarial units vs ARV units' questions."
        ),
        "cube_query": {
            "measures": ["fact_dispensing.quantity_dispensed", "fact_dispensing.count"],
            "dimensions": [
                "fact_dispensing.source_schema",
                "canonical_product_taxonomy.canonical_name",
                "canonical_product_taxonomy.therapeutic_class",
                "canonical_product_taxonomy.therapeutic_subclass",
            ],
            "timeDimensions": [{"dimension": "fact_dispensing.dispensing_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "drug_cost_metrics",
        "name": "Drug Cost and Spend",
        "description": (
            "Total drug spend (KES), average unit price, average cost per dispense "
            "line, and cost per patient, by product, therapeutic class, facility, or "
            "month. Use for 'total drug spend this month', 'which therapeutic class "
            "costs the most', 'average unit price by product', 'most expensive "
            "drug', 'drug cost breakdown by therapeutic class', 'cost of ARVs last "
            "quarter', 'average line total per dispense', 'cost per patient per "
            "month' questions."
        ),
        "cube_query": {
            "measures": [
                "fact_dispensing.line_total",
                "fact_dispensing.avg_unit_price",
                "fact_dispensing.avg_line_total",
                "fact_dispensing.cost_per_patient",
                "fact_dispensing.distinct_patients",
                "fact_dispensing.count",
            ],
            "dimensions": [
                "fact_dispensing.source_schema",
                "canonical_product_taxonomy.canonical_name",
                "canonical_product_taxonomy.therapeutic_class",
            ],
            "timeDimensions": [{"dimension": "fact_dispensing.dispensing_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "patient_therapeutic_coverage_metrics",
        "name": "Patient Therapeutic Coverage (ARV/Antimalarial/Antibiotic/etc.)",
        "description": (
            "Per-VISIT flags for whether a dispensing visit included an ARV, "
            "antimalarial, antibiotic, reserve (AMR) antibiotic, chronic-disease "
            "drug, opioid, or obstetric drug — plus distinct patient counts and "
            "average drugs dispensed per visit. Use for 'how many patients received "
            "ARVs', 'distinct patients dispensed last quarter', 'visits that "
            "included an antimalarial', 'patient count by therapeutic flag', "
            "'patients who received reserve antibiotics', 'visits with obstetric "
            "drugs', 'which therapeutic flag is most common', 'average drugs per "
            "visit' questions. facility filter is fact_patient_dispensing."
            "source_schema (lowercase, e.g. 'kisumu')."
        ),
        "cube_query": {
            "measures": [
                "fact_patient_dispensing.count",
                "fact_patient_dispensing.distinct_patients",
                "fact_patient_dispensing.has_arv",
                "fact_patient_dispensing.has_antimalarial",
                "fact_patient_dispensing.has_antibiotic",
                "fact_patient_dispensing.has_chronic_drug",
                "fact_patient_dispensing.has_reserve_antibiotic",
                "fact_patient_dispensing.has_obstetric_drug",
                "fact_patient_dispensing.has_opioid",
                "fact_patient_dispensing.avg_drugs_per_visit",
                "fact_patient_dispensing.unclassified_drug_count",
            ],
            "dimensions": ["fact_patient_dispensing.source_schema"],
            "timeDimensions": [{"dimension": "fact_patient_dispensing.dispensing_month", "granularity": "month"}],
            "filters": [],
            "limit": 500,
        },
    },
    {
        "metric_id": "product_taxonomy_quality_metrics",
        "name": "Product Taxonomy Data Quality",
        "description": (
            "Data-quality signals for the drug product taxonomy mapping: products "
            "not yet mapped to a canonical name (inn_map_status='needs_review'), "
            "low-confidence mappings (match_confidence below 60), and products with "
            "no therapeutic_class assigned. Use for 'products not mapped to "
            "canonical names', 'how many products need taxonomy review', 'low "
            "confidence product mappings', 'unclassified drugs dispensed', "
            "'products with unknown therapeutic class', 'unmapped products by "
            "category', 'data quality gaps in the product taxonomy' questions."
        ),
        "cube_query": {
            "measures": ["canonical_product_taxonomy.count", "canonical_product_taxonomy.match_confidence"],
            "dimensions": [
                "canonical_product_taxonomy.facility",
                "canonical_product_taxonomy.inn_map_status",
                "canonical_product_taxonomy.product_category",
                "canonical_product_taxonomy.product_name",
                "canonical_product_taxonomy.canonical_name",
                "canonical_product_taxonomy.match_type",
                "canonical_product_taxonomy.therapeutic_class",
            ],
            "timeDimensions": [],
            "filters": [],
            "limit": 500,
        },
    },
]


class Command(BaseCommand):
    help = "Seed/update curated MetricDefinition rows for hospital-ops and pharmacy domains."

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
            f"Seeded {len(METRICS)} curated metrics ({created} created, {updated} updated). "
            f"Run 'python manage.py rebuild_embeddings' next so they're searchable."
        ))
