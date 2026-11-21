-- Disease category x ward x year, using all_diagnoses. No arrays, no FLATTEN.
WITH cats(category, pattern) AS (
    SELECT column1, column2 FROM VALUES
        ('Infectious disease',  '.*(pneumonia|sepsis|malaria|gastritis|urinary tract infection).*'),
        ('Respiratory',         '.*(pneumonia|respiratory|asthma|tonsil|bronch).*'),
        ('NCD',                 '.*(hypertension|diabetes|chronic kidney|epilepsy).*'),
        ('Injury & trauma',     '.*(fracture|wound|burn|injury).*'),
        ('Maternal & obstetric','.*(pregnan|obstetric|abortion|puerperi).*'),
        ('Mental health',       '.*(schizophrenia|alcohol|bipolar|depression).*'),
        ('Malaria',             '.*(malaria|plasmodium).*')
),
w AS (
    SELECT county, sub_county, ward, visit_year, unique_patient_id, all_diagnoses,
           -- ward total computed once per visit row, before the join to categories
           COUNT(*) OVER (PARTITION BY county, sub_county, ward, visit_year) AS ward_visits
    FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
    WHERE (is_default_code_pair = FALSE OR is_default_code_pair IS NULL)
      AND ward IS NOT NULL
      -- AND geo_confidence IN ('High', 'Medium')
)
SELECT w.county, w.sub_county, w.ward, w.visit_year, c.category,
       COUNT(*)                                       AS category_visits,
       COUNT(DISTINCT w.unique_patient_id)            AS category_patients,
       ANY_VALUE(w.ward_visits)                       AS ward_visits,
       ROUND(COUNT(*) * 100.0 / ANY_VALUE(w.ward_visits), 2) AS pct_of_ward_visits
FROM w
JOIN cats c ON REGEXP_LIKE(COALESCE(w.all_diagnoses, ''), c.pattern, 'is')
GROUP BY w.county, w.sub_county, w.ward, w.visit_year, c.category
ORDER BY w.visit_year, c.category, category_visits DESC;
