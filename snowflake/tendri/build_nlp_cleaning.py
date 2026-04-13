"""
xana_build_nlp_cleaning.py  —  Xana Pipeline
══════════════════════════════════════════════════════════════════════════════
Step 3: NLP diagnosis cleaning.

READS FROM
──────────
    HOSPITALS.STAGING.stg_doctor_notes   — free-text diagnosis column
    HOSPITALS.STAGING.stg_icd10          — structured ICD-10 codes + hierarchy

Both tables are populated by build_staging.py. stg_icd10 already has the
full ICD-10 hierarchy pre-joined from evaluation_doctor_notes_merged
(variation → type → subcategory → category). No further joins needed here.

MERGE STRATEGY
──────────────
The previous approach (Tendri pipeline) joined doctor_notes and icd10 inside
_build_pivoted_icd_work() using a window function over stg_unified_icd10 on
every run — expensive at scale.

Here the merge is already done at ingest time:
  STAGING.stg_icd10  already has icd10_note_code, icd10_variation_name etc.
  per sk_visit_id (one row per visit, primary code only via ROW_NUMBER).

This file just needs to:
  1. Read stg_doctor_notes for free-text
  2. Read stg_icd10 for the pre-resolved ICD-10 name (rank-1 code per visit)
  3. Coalesce: ICD-10 name first, NLP gap-fill second
  4. Run taxonomy categorisation on the coalesced string
  5. Write to HOSPITALS.REPORTING.fact_doctor_notes_cleaned

WRITES TO
──────────
    HOSPITALS.REPORTING.fact_doctor_notes_cleaned

USAGE
──────
    python xana_build_nlp_cleaning.py           # incremental (new visits only)
    python xana_build_nlp_cleaning.py --full    # full rebuild
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import re
import traceback
from datetime import datetime
from typing import List, Optional

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas # type: ignore
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import STAGING_SCHEMA, REPORTING_SCHEMA
from db import build_mysql_engine

SF_TARGET_DB     = "HOSPITALS"
SF_TARGET_SCHEMA = "TENRI_RAW"

log = logging.getLogger(__name__)

# ── Table references ──────────────────────────────────────────────────────────
SOURCE_NOTES  = f"`{STAGING_SCHEMA}`.`stg_doctor_notes`"
SOURCE_ICD10  = f"`{STAGING_SCHEMA}`.`stg_icd10`"
TARGET_SCHEMA = REPORTING_SCHEMA
TARGET_TABLE  = "fact_doctor_notes_cleaned"
CHUNK_SIZE    = 50_000

_SHADOW_SUFFIX = "_new"
_WORK_ICD      = "_work_icd_pivot"
_WORK_NEW_IDS  = "_work_new_note_ids"


# ══════════════════════════════════════════════════════════════════════════════
#  REGEX CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RULE_OUT_RE = re.compile(
    r"r\s*/\s*o\.?|r\.o\.?|rule\s*out|^\?|\?\s*[a-z]",
    re.IGNORECASE,
)

INVALID_RE_SIMPLE = re.compile(
    r"^[=\-_/;,.!?@#%^*\\|<>\s]+$"
    r"|^[a-z]{1,2}$"
    r"|^[0-9\s]+$",
    re.IGNORECASE,
)
INVALID_RE_REPEAT = re.compile(r"^([a-z])\1+$", re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
#  ABBREVIATION EXPANSION  — applied before categorisation
# ══════════════════════════════════════════════════════════════════════════════

_RAW_REPLACEMENTS: list[tuple[str, str]] = [
    # Cardiovascular
    (r"\bHTN\b",                     "hypertension"),
    (r"\bHBP\b",                     "high blood pressure"),
    (r"\bCCF\b",                     "congestive cardiac failure"),
    (r"\bCHF\b",                     "congestive heart failure"),
    (r"\bCAD\b",                     "coronary artery disease"),
    (r"\bIHD\b",                     "ischaemic heart disease"),
    (r"\bMI\b",                      "myocardial infarction"),
    (r"\bNSTEMI\b",                  "non-ST-elevation myocardial infarction"),
    (r"\bSTEMI\b",                   "ST-elevation myocardial infarction"),
    (r"\bAFib\b",                    "atrial fibrillation"),
    (r"\bAF\b(?!\s*fever)",          "atrial fibrillation"),
    (r"\bCVA\b",                     "cerebrovascular accident"),
    (r"\bCVD\b",                     "cardiovascular disease"),
    (r"\bPE\b(?!\s*tube)",           "pulmonary embolism"),
    (r"\bDVT\b",                     "deep vein thrombosis"),
    (r"\bPAD\b",                     "peripheral arterial disease"),
    # Diabetes
    (r"\bT2DM\b",                    "type 2 diabetes mellitus"),
    (r"\bT1DM\b",                    "type 1 diabetes mellitus"),
    (r"\bDM\s*[Tt]ype\s*2\b",       "type 2 diabetes mellitus"),
    (r"\bDM\s*[Tt]ype\s*1\b",       "type 1 diabetes mellitus"),
    (r"\bDM2\b",                     "type 2 diabetes mellitus"),
    (r"\bDM1\b",                     "type 1 diabetes mellitus"),
    (r"\bDM\b",                      "diabetes mellitus"),
    (r"\bNIDDM\b",                   "non-insulin dependent diabetes mellitus"),
    (r"\bIDDM\b",                    "insulin dependent diabetes mellitus"),
    (r"\bGDM\b",                     "gestational diabetes mellitus"),
    (r"\bDKA\b",                     "diabetic ketoacidosis"),
    # Respiratory
    (r"\bCOPD\b",                    "chronic obstructive pulmonary disease"),
    (r"\bURTI\b",                    "upper respiratory tract infection"),
    (r"\bLRTI\b",                    "lower respiratory tract infection"),
    (r"\bARI\b",                     "acute respiratory infection"),
    (r"\bARS\b",                     "acute rhinosinusitis"),
    (r"\bPTB\b",                     "pulmonary tuberculosis"),
    (r"\bTBM\b",                     "tuberculous meningitis"),
    (r"\bTB\b",                      "tuberculosis"),
    # Renal
    (r"\bCKD\b",                     "chronic kidney disease"),
    (r"\bAKI\b",                     "acute kidney injury"),
    (r"\bARF\b",                     "acute renal failure"),
    (r"\bCRF\b",                     "chronic renal failure"),
    (r"\bESRD\b",                    "end stage renal disease"),
    # HIV / ARV
    (r"\bRVD\b",                     "retroviral disease"),
    (r"\bARV\b",                     "antiretroviral"),
    (r"\bHAART\b",                   "highly active antiretroviral therapy"),
    (r"\bPMTCT\b",                   "prevention of mother to child transmission"),
    (r"\bVL\b(?=\s)",                "viral load"),
    # Gastrointestinal
    (r"\bGE\b(?!\s*[A-Z]{2})",      "gastroenteritis"),
    (r"\bGERD\b",                    "gastroesophageal reflux disease"),
    (r"\bGORD\b",                    "gastro-oesophageal reflux disease"),
    (r"\bPUD\b",                     "peptic ulcer disease"),
    (r"\bIBS\b",                     "irritable bowel syndrome"),
    (r"\bIBD\b",                     "inflammatory bowel disease"),
    # Musculoskeletal
    (r"\bOA\b",                      "osteoarthritis"),
    (r"\bRA\b",                      "rheumatoid arthritis"),
    (r"\bLBP\b",                     "lower back pain"),
    (r"\bMSK\b",                     "musculoskeletal"),
    (r"\bSLE\b",                     "systemic lupus erythematosus"),
    (r"\bAS\b(?=\s)",                "ankylosing spondylitis"),
    # Urinary
    (r"\bUTI\b",                     "urinary tract infection"),
    (r"\bBPH\b",                     "benign prostatic hyperplasia"),
    (r"\bSTUI\b",                    "lower urinary tract symptoms"),
    # Gynaecological
    (r"\bPID\b",                     "pelvic inflammatory disease"),
    (r"\bSTI\b",                     "sexually transmitted infection"),
    (r"\bSTD\b",                     "sexually transmitted disease"),
    (r"\bFP\b(?!\w)",                "family planning"),
    (r"\bFGM\b",                     "female genital mutilation"),
    (r"\bVVC\b",                     "vulvovaginal candidiasis"),
    # Maternal / Obstetric
    (r"\bANC\b(?!\w)",               "antenatal care"),
    (r"\bPNC\b(?!\w)",               "postnatal care"),
    (r"\bEDD\b",                     "expected date of delivery"),
    (r"\bC/S\b",                     "caesarean section"),
    (r"\bLSCS\b",                    "lower segment caesarean section"),
    (r"\bSVD\b",                     "spontaneous vaginal delivery"),
    (r"\bNVD\b",                     "normal vaginal delivery"),
    (r"\bPPH\b",                     "postpartum haemorrhage"),
    (r"\bGHT\b",                     "gestational hypertension"),
    (r"\bPIH\b",                     "pregnancy-induced hypertension"),
    (r"\bPET\b",                     "pre-eclampsia"),
    (r"\bECL\b",                     "eclampsia"),
    (r"\bHELLP\b",                   "HELLP syndrome"),
    (r"\bROM\b",                     "rupture of membranes"),
    (r"\bPROM\b",                    "premature rupture of membranes"),
    # Child / Nutrition
    (r"\bEPI\b",                     "immunisation"),
    (r"\bGMC\b",                     "growth monitoring"),
    (r"\bSAM\b",                     "severe acute malnutrition"),
    (r"\bMAM\b",                     "moderate acute malnutrition"),
    (r"\bMUAC\b",                    "mid-upper arm circumference"),
    # Haematology / Sickle Cell
    (r"\bSCD\b",                     "sickle cell disease"),
    (r"\bSCA\b",                     "sickle cell anaemia"),
    (r"\bHbSS\b",                    "sickle cell disease"),
    (r"\bHbSC\b",                    "sickle cell disease SC"),
    (r"\bSS\s+disease\b",            "sickle cell disease"),
    (r"\bVOC\b",                     "vaso-occlusive crisis"),
    (r"\bSCC\b(?!\s+cancer)",        "sickle cell crisis"),
    (r"\bACS\b(?!\s+coronary)",      "acute chest syndrome"),
    (r"\bDIC\b",                     "disseminated intravascular coagulation"),
    # Mental health
    (r"\bMDD\b",                     "major depressive disorder"),
    (r"\bBPD\b",                     "borderline personality disorder"),
    (r"\bPTSD\b",                    "post-traumatic stress disorder"),
    (r"\bADHD\b",                    "attention deficit hyperactivity disorder"),
    (r"\bOCD\b",                     "obsessive compulsive disorder"),
    # Neurological
    (r"\bTIA\b",                     "transient ischaemic attack"),
    (r"\bMS\b(?=\s)",                "multiple sclerosis"),
    (r"\bMND\b",                     "motor neurone disease"),
    # Oncology
    (r"\bCa\b(?=\s+\w)",             "cancer"),
    (r"\bCRC\b",                     "colorectal cancer"),
    (r"\bNHL\b",                     "non-Hodgkin lymphoma"),
    (r"\bCLL\b",                     "chronic lymphocytic leukaemia"),
    (r"\bCML\b",                     "chronic myeloid leukaemia"),
    (r"\bAML\b",                     "acute myeloid leukaemia"),
    # Miscellaneous clinical
    (r"\bHP\b(?!\s*pylori)",         "hypertension"),
    (r"\bF/U\b",                     "follow-up"),
    (r"\bR/O\b",                     "rule out"),
    (r"\bR\.O\.\b",                  "rule out"),
    (r"\bHx\b",                      "history"),
    (r"\bDx\b",                      "diagnosis"),
    (r"\bTx\b",                      "treatment"),
    (r"\bRx\b",                      "prescription"),
]

_ABBR_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _RAW_REPLACEMENTS
]


def _expand_abbreviations(text: str) -> str:
    """Expand clinical abbreviations before categorisation."""
    if not text:
        return text
    for pattern, replacement in _ABBR_COMPILED:
        text = pattern.sub(replacement, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  NLP TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

DISEASE_PATTERNS: list[tuple[str, list[str]]] = [
    # ── NCDs ──────────────────────────────────────────────────────────────────
    ("NCD - Hypertension", [
        r"\bhypertension\b", r"\bhigh blood pressure\b", r"\bessential hypertension\b",
        r"\bprimary hypertension\b", r"\bsecondary hypertension\b",
        r"\bgestational hypertension\b", r"\bpregnancy.induced hypertension\b",
        r"\bpre.eclampsia\b", r"\beclampsia\b",
    ]),
    ("NCD - Diabetes", [
        r"\bdiabetes\b", r"\bdiabetic\b", r"\bdiabetes mellitus\b",
        r"\btype\s*[12]\s*diabetes\b", r"\bnon.insulin dependent\b",
        r"\binsulin dependent\b", r"\bhyperglycaemi\b", r"\bhyperglycemi\b",
        r"\bdiabetic ketoacidosis\b", r"\bhypoglycaemi\b", r"\bhypoglycemi\b",
        r"\bgestational diabetes\b",
    ]),
    ("NCD - Cardiovascular", [
        r"\bcardiovascular\b", r"\bheart\s+disease\b", r"\bheart\s+failure\b",
        r"\bcardiac\s+failure\b", r"\bcongestive\s+cardiac\b",
        r"\bcongestive\s+heart\b", r"\bcoronary\s+artery\b",
        r"\bischaemic\s+heart\b", r"\bischemic\s+heart\b",
        r"\bangina\b", r"\batrial\s+fibrillation\b", r"\barrhythmia\b",
        r"\bmyocardial\b", r"\bcardiomyopathy\b", r"\bpericarditi\b",
        r"\bvalvular\s+heart\b", r"\bmitral\b", r"\baortic\s+stenosis\b",
        r"\bpulmonary\s+embolism\b", r"\bdeep\s+vein\s+thrombosis\b",
        r"\bperipheral\s+arterial\b", r"\bperipheral\s+vascular\b",
        r"\bheart\s+block\b",
    ]),
    ("NCD - Cerebrovascular", [
        r"\bstroke\b", r"\bcerebrovascular\b", r"\bcerebral\s+infarct\b",
        r"\btransient\s+ischaemic\b", r"\btransient\s+ischemic\b",
        r"\btia\b(?!\w)", r"\bintracranial\s+haemorrhage\b",
        r"\bsubarachnoid\s+haemorrhage\b", r"\bhemiplegia\b", r"\bhemiparesis\b",
    ]),
    ("NCD - Respiratory Chronic", [
        r"\basthma\b", r"\bchronic\s+obstructive\b",
        r"\bchronic\s+bronchitis\b", r"\bpulmonary\s+fibrosis\b",
        r"\brespiratory\s+failure\b", r"\bbronchiectasis\b",
        r"\binterstitial\s+lung\b", r"\bpulmonary\s+hypertension\b",
        r"\bobstructive\s+sleep\s+apn\b",
    ]),
    ("NCD - Renal", [
        r"\bchronic\s+kidney\b", r"\bchronic\s+renal\b",
        r"\bend\s+stage\s+renal\b", r"\brenal\s+failure\b",
        r"\bkidney\s+failure\b", r"\bnephrotic\b", r"\bnephropathy\b",
        r"\bnephritis\b", r"\bdialysis\b", r"\brenal\s+transplant\b",
    ]),
    ("NCD - Mental Health", [
        r"\bdepression\b", r"\bdepressive\s+disorder\b", r"\banxiety\b",
        r"\bschizophrenia\b", r"\bbipolar\b", r"\bpsychosis\b",
        r"\bpsychotic\b", r"\bpsychiatric\b", r"\bmental\s+health\b",
        r"\bmental\s+illness\b", r"\bpost.traumatic\b", r"\bpanic\s+disorder\b",
        r"\bobsessive.compulsive\b", r"\beating\s+disorder\b",
        r"\bpersonality\s+disorder\b", r"\bdementia\b", r"\balzheimer\b",
    ]),
    ("NCD - Substance Use", [
        r"\bsubstance\s+abuse\b", r"\bsubstance\s+use\b",
        r"\balcohol\s+use\s+disorder\b", r"\balcohol\s+abuse\b",
        r"\bdrug\s+abuse\b", r"\bdrug\s+dependence\b",
        r"\baddiction\b", r"\bwithdrawal\b",
    ]),
    ("NCD - Cancer / Oncology", [
        r"\bcancer\b", r"\bcarcinoma\b", r"\btumou?r\b", r"\bneoplasm\b",
        r"\blymphoma\b", r"\bleukaemia\b", r"\bleukemia\b", r"\bsarcoma\b",
        r"\bmelanoma\b", r"\boncolog\b", r"\bmalignant\b", r"\bmetastati\b",
        r"\bbreast\s+cancer\b", r"\bcervical\s+cancer\b",
        r"\bprostate\s+cancer\b", r"\bcolorectal\s+cancer\b",
        r"\blung\s+cancer\b", r"\bkaposi\b", r"\bchemo\b", r"\bradiation\s+therapy\b",
    ]),
    ("NCD - Neurological", [
        r"\bepilepsy\b", r"\bepileptic\b", r"\bseizure\b",
        r"\bparkinson\b", r"\bneuropathy\b", r"\bneurological\b",
        r"\bmultiple\s+sclerosis\b", r"\bmotor\s+neurone\b",
        r"\bguillain.barr\b", r"\bmeningitis\b(?!.*tubercul)",
        r"\bencephaliti\b", r"\bhydrocephalus\b", r"\bbrain\s+tumou?r\b",
        r"\bneurocysticercosis\b",
    ]),
    ("NCD - Haematology", [
        r"\bsickle\s*cell\b", r"\bvaso.occlusive\b", r"\bsickle\s+cell\s+crisis\b",
        r"\banaemia\b", r"\banaemic\b", r"\banemia\b", r"\banemic\b",
        r"\bhaemophilia\b", r"\bthalassaemi\b",
        r"\bhaematological\b", r"\bblood\s+disorder\b",
        r"\bdisseminated\s+intravascular\b",
        r"\baplastic\s+anaemia\b", r"\biron\s+deficiency\s+anaemia\b",
    ]),
    ("NCD - Endocrine", [
        r"\bthyroid\b", r"\bhypothyroidism\b", r"\bhyperthyroidism\b",
        r"\bthyrotoxicosis\b", r"\bgoitre\b", r"\bgout\b", r"\bendocrine\b",
        r"\bmetabolic\s+syndrome\b", r"\bobesity\b", r"\boverweight\b",
        r"\badrenal\s+insufficiency\b", r"\bcushing\b", r"\baddison\b",
    ]),
    ("NCD - Musculoskeletal", [
        r"\barthritis\b", r"\brheumatoid\b", r"\bosteoporosis\b",
        r"\bosteoarthritis\b", r"\bankylosing\s+spondylitis\b",
        r"\bsystemic\s+lupus\b", r"\blupus\b", r"\bback\s+pain\b",
        r"\bjoint\s+pain\b", r"\bmusculoskeletal\b", r"\bfibromyalgia\b",
        r"\bgout\s+arthritis\b", r"\bspinal\s+stenosis\b",
        r"\bdegenerative\s+disc\b",
    ]),
    ("NCD - Digestive", [
        r"\bpeptic\s+ulcer\b", r"\bgastric\s+ulcer\b", r"\bduodenal\s+ulcer\b",
        r"\bcrohn\b", r"\bulcerative\s+colitis\b", r"\birritable\s+bowel\b",
        r"\bgastro.oesophageal\s+reflux\b", r"\bgastroesophageal\s+reflux\b",
        r"\bhepatic\b", r"\bchronic\s+liver\b", r"\bcirrhosis\b",
        r"\bpancreatitis\b(?!\s*acute)", r"\bchronic\s+pancreatitis\b",
        r"\bcoeliac\b", r"\binflammatory\s+bowel\b",
        r"\bhepatitis\s+b\b", r"\bhepatitis\s+c\b",
    ]),
    ("NCD - Ophthalmic", [
        r"\bglaucoma\b", r"\bcataract\b", r"\bdiabetic\s+retinopathy\b",
        r"\bmacular\s+degeneration\b", r"\bretinal\b(?!\s+infection)",
        r"\bvisual\s+impairment\b", r"\bblindn\b", r"\buveitis\b(?!\s+infect)",
        r"\boptic\s+neuritis\b",
    ]),
    ("NCD - Urological", [
        r"\bbenign\s+prostatic\b", r"\bprostate\s+hyper\b",
        r"\boveractive\s+bladder\b", r"\bincontinence\b", r"\bhydronephrosis\b",
        r"\bkidney\s+ston\b", r"\brenal\s+calculi\b", r"\bnephrolithiasis\b",
    ]),
    ("NCD - Dermatological", [
        r"\beczema\b", r"\bpsoriasis\b", r"\bdermatitis\b",
        r"\bchronic\s+skin\b", r"\bichthyosis\b", r"\bvitiligo\b",
        r"\balopecia\b", r"\brosacea\b",
    ]),
    # ── Communicable Diseases ─────────────────────────────────────────────────
    ("Communicable - Malaria", [
        r"\bmalaria\b", r"\bfalciparum\b", r"\bvivax\b", r"\bplasmodium\b",
        r"\buncomplicated\s+malaria\b", r"\bsevere\s+malaria\b",
        r"\bcomplicated\s+malaria\b", r"\bcerebral\s+malaria\b",
    ]),
    ("Communicable - Tuberculosis", [
        r"\btuberculosis\b", r"\bpulmonary\s+tuberculosis\b",
        r"\btb\s+disease\b", r"\btb\s+treatment\b", r"\bspinal\s+tuberculosis\b",
        r"\btuberculous\s+meningitis\b", r"\btuberculous\b",
        r"\bextra.pulmonary\s+tb\b", r"\bmdr.tb\b", r"\bxdr.tb\b",
        r"\blatent\s+tb\b",
    ]),
    ("Communicable - Respiratory Infection", [
        r"\bpneumonia\b", r"\bbronchitis\b(?!\s+chronic)",
        r"\bupper\s+respiratory\s+tract\s+infection\b",
        r"\blower\s+respiratory\s+tract\s+infection\b",
        r"\bacute\s+respiratory\s+infection\b",
        r"\bcovid.?19\b", r"\bcoronavirus\b", r"\bsars.cov\b",
        r"\binfluenza\b", r"\bflu\b(?!\w)", r"\bsinusitis\b",
        r"\btonsillitis\b", r"\bpharyngitis\b", r"\blaryngitis\b",
        r"\bepiglottitis\b", r"\bpleural\s+effusion\b",
        r"\bempyema\b", r"\blung\s+abscess\b",
    ]),
    ("Communicable - Gastrointestinal", [
        r"\bdiarrhoea\b", r"\bdiarrhea\b", r"\bgastroenteritis\b",
        r"\bgastritis\b(?!\s+chronic)", r"\bdysentery\b",
        r"\bfood\s+poisoning\b", r"\btyphoid\b", r"\bcholera\b",
        r"\brotavirus\b", r"\bnoro\s*virus\b", r"\bhep\s*a\b",
        r"\bhepatitis\s+a\b", r"\bgiardia\b", r"\bamoebias\b",
        r"\benteric\s+fever\b", r"\babdominal\s+infection\b",
    ]),
    ("Communicable - HIV/AIDS", [
        r"\bhiv\b", r"\baids\b", r"\bretroviral\s+disease\b",
        r"\bantiretroviral\b", r"\bhaart\b", r"\bcd4\b",
        r"\bopportunistic\s+infection\b",
        r"\bprevention\s+of\s+mother\s+to\s+child\b",
    ]),
    ("Communicable - Urinary Tract Infection", [
        r"\burinary\s+tract\s+infection\b", r"\bcystitis\b",
        r"\bpyelonephritis\b", r"\burethritis\b",
        r"\brecurrent\s+uti\b",
    ]),
    ("Communicable - Gynaecological Infection", [
        r"\bsexually\s+transmitted\b", r"\bgonorrhoea\b", r"\bgonorrhea\b",
        r"\bchlamydia\b", r"\bsyphilis\b", r"\bvaginitis\b",
        r"\bcervicitis\b", r"\bpelvic\s+inflammatory\b",
        r"\bvulvovaginal\s+candidiasis\b", r"\btrichomoniasis\b",
        r"\bgenital\s+herpes\b", r"\bgenital\s+warts\b",
        r"\bcondyloma\b", r"\bhpv\b",
    ]),
    ("Communicable - Skin Infection", [
        r"\bcellulitis\b", r"\bimpetigo\b", r"\bfungal\s+infection\b",
        r"\btinea\b", r"\bscabies\b", r"\bringworm\b",
        r"\bskin\s+abscess\b", r"\bboil\b", r"\bfuruncle\b",
        r"\bcarbuncle\b", r"\bherpes\s+zoster\b", r"\bshingles\b",
        r"\bchicken\s*pox\b", r"\bvaricella\b",
    ]),
    ("Communicable - Eye Infection", [
        r"\bconjunctivitis\b", r"\beye\s+infection\b",
        r"\btrachoma\b", r"\bkeratitis\b", r"\bendophthalmitis\b",
        r"\bdacryocystitis\b", r"\borbital\s+cellulitis\b",
    ]),
    ("Communicable - Other", [
        r"\bsepsis\b", r"\bsepticaemia\b", r"\bsepticemia\b",
        r"\bmeningitis\b(?!.*tubercul)", r"\bencephalitis\b",
        r"\bbrucellosis\b", r"\bleptospirosis\b", r"\bdengue\b",
        r"\byellow\s+fever\b", r"\btyphus\b", r"\brickettsial\b",
        r"\binfection\b(?!\s*(urinary|skin|eye|tract))",
        r"\bfever\b(?!\s*(hay|dengue|yellow|rheumatic))", r"\bviral\b", r"\bbacterial\b",
    ]),
    # ── MNCH ──────────────────────────────────────────────────────────────────
    ("MNCH - Maternal: ANC", [
        r"\bantenatal\b", r"\bprenatal\b", r"\bantenatal\s+care\b",
        r"\bprenatal\s+care\b", r"\bpregnancy\s+review\b",
        r"\bpregnancy\s+check\b", r"\bpregnancy\s+visit\b",
        r"\bexpected\s+date\s+of\s+delivery\b",
    ]),
    ("MNCH - Maternal: Obstetric Complication", [
        r"\bpre.eclampsia\b", r"\beclampsia\b", r"\bhellp\b",
        r"\bantepartum\s+haemorrhage\b", r"\bplacenta\s+praevia\b",
        r"\bplacental\s+abruption\b", r"\bruptured\s+uterus\b",
        r"\bectopic\s+pregnancy\b", r"\bmiscarriage\b", r"\babortion\b",
        r"\bthreatened\s+abortion\b", r"\bmolar\s+pregnancy\b",
        r"\bpremature\s+labour\b", r"\bpreterm\s+labour\b",
        r"\bpremature\s+rupture\b", r"\bprolapsed\s+cord\b",
    ]),
    ("MNCH - Maternal: Intrapartum", [
        r"\blabour\b(?!\s+pain)", r"\blabor\b", r"\bdelivery\b",
        r"\bbirth\b", r"\bintrapartum\b",
        r"\bcaesarean\s+section\b", r"\bc.section\b", r"\blscs\b",
        r"\bspontaneous\s+vaginal\b", r"\bnormal\s+vaginal\b",
        r"\bpostpartum\b", r"\bpost.partum\b",
    ]),
    ("MNCH - Maternal: Postnatal", [
        r"\bpostnatal\b", r"\bpostnatal\s+care\b", r"\bpuerperium\b",
        r"\bpostpartum\s+haemorrhage\b", r"\bpuerperal\b",
        r"\bbreastfeeding\b", r"\blactation\b",
        r"\bpost.partum\s+depression\b",
    ]),
    ("MNCH - Neonatal", [
        r"\bneonatal\b", r"\bnewborn\b", r"\bneonate\b",
        r"\blow\s+birth\s+weight\b", r"\bprematur\b(?!\s+rupture)",
        r"\bneonatal\s+jaundice\b", r"\bneonatal\s+sepsis\b",
        r"\bbirth\s+asphyxia\b", r"\brespiratory\s+distress\s+syndrome\b",
        r"\bneonatal\s+infection\b",
    ]),
    ("MNCH - Child Health", [
        r"\bpaediatric\b", r"\bpediatric\b", r"\bchild\s+health\b",
        r"\bimmunisation\b", r"\bvaccination\b", r"\bgrowth\s+monitor\b",
        r"\bwell\s+baby\b", r"\bwell.child\b", r"\bschool\s+health\b",
        r"\bchild\s+welfare\b",
    ]),
    # ── Nutrition ─────────────────────────────────────────────────────────────
    ("Nutrition & Deficiency", [
        r"\bmalnutrition\b", r"\bsevere\s+acute\s+malnutrition\b",
        r"\bmoderate\s+acute\s+malnutrition\b", r"\bwasting\b(?!\s+disease)",
        r"\bstunting\b", r"\bkwashiorkor\b", r"\bmarasmus\b",
        r"\biron\s+deficiency\b", r"\bvitamin\s+[a-d]\s+deficiency\b",
        r"\brickets\b", r"\bscurvy\b", r"\bpellagra\b",
        r"\banaemia\s+of\s+deficiency\b", r"\banaemia\s+in\s+pregnancy\b",
        r"\bnutritional\b",
    ]),
    # ── Reproductive Health ────────────────────────────────────────────────────
    ("Reproductive Health", [
        r"\bcontraception\b", r"\bfamily\s+planning\b",
        r"\breproductive\s+health\b", r"\bfertility\b",
        r"\bgynaecolog\b", r"\bgynecolog\b", r"\buterine\b",
        r"\bovarian\b", r"\bmenstrual\b", r"\bdysmenorrhoea\b",
        r"\bamenorrhoea\b", r"\bpcos\b", r"\bendometriosis\b",
        r"\bfibroids\b", r"\buterine\s+fibroid\b", r"\binfertility\b",
        r"\bmenopausal\b", r"\bmenopause\b", r"\bcervical\s+screen\b",
        r"\bpap\s+smear\b", r"\bcolposcopy\b",
    ]),
    # ── Injury & Violence ─────────────────────────────────────────────────────
    ("Injury & Violence", [
        r"\btrauma\b", r"\binjury\b", r"\bfracture\b", r"\blaceration\b",
        r"\bwound\b", r"\bbite\b", r"\bburn\b", r"\bassault\b",
        r"\baccident\b", r"\bfall\b", r"\bsprain\b", r"\bdislocation\b",
        r"\bhead\s+injury\b", r"\bcranial\s+trauma\b",
        r"\bdomestic\s+violence\b", r"\bsexual\s+assault\b",
        r"\broad\s+traffic\b", r"\brtc\b(?!\w)",
    ]),
    # ── Surgical / Procedural ─────────────────────────────────────────────────
    ("Surgical / Procedural", [
        r"\bpre.?op\b", r"\bpost.?op\b", r"\bpre.?operative\b",
        r"\bpost.?operative\b", r"\bsurgery\b", r"\bsurgical\b",
        r"\bappendectomy\b", r"\bappendicitisi?\b", r"\bhernia\b",
        r"\blaparotomy\b", r"\bintestinal\s+obstruction\b",
        r"\bowound\s+review\b", r"\bdebridement\b",
        r"\bwound\s+dressing\b", r"\bsuture\s+removal\b",
        r"\bfistula\b", r"\babscess\b(?!\s+orbital|\s+brain)",
    ]),
    # ── Dental / Oral Health ─────────────────────────────────────────────────
    ("Dental / Oral Health", [
        r"\bdental\b", r"\btooth\b", r"\bteeth\b", r"\boral\b",
        r"\btoothache\b", r"\bperiodontal\b", r"\bgum\s+disease\b",
        r"\bdentition\b", r"\bextraction\b(?!\s+kidney|\s+tooth)",
        r"\bcaries\b", r"\bdecay\b(?!\s+mental)",
    ]),
    # ── Screening / Prevention ────────────────────────────────────────────────
    ("Screening / Prevention", [
        r"\bscreening\b", r"\bhealth\s+check\b", r"\broutine\s+check\b",
        r"\bpreventive\b", r"\bprevention\b", r"\bimmunisation\s+visit\b",
        r"\bvaccine\b", r"\bbp\s+check\b", r"\bblood\s+pressure\s+check\b",
        r"\bblood\s+glucose\s+check\b", r"\bcholesterol\s+check\b",
        r"\bcancer\s+screening\b",
    ]),
    # ── Medication / Investigation ────────────────────────────────────────────
    ("Medication / Investigation", [
        r"\bprescription\b", r"\bmedication\s+refill\b",
        r"\bdrug\s+refill\b", r"\bmedication\s+pickup\b",
        r"\bblood\s+test\b", r"\blab\s+result\b",
        r"\binvestigation\s+only\b", r"\bfollow.up\s+result\b",
        r"\bx.ray\b", r"\bultrasound\b", r"\becho\b(?!\w)",
        r"\bct\s+scan\b", r"\bmri\b",
    ]),
]

# Priority order for split_and_categorise — lower number = higher priority
CATEGORY_PRIORITY: dict[str, int] = {
    "NCD - Hypertension":                    10,
    "NCD - Diabetes":                        11,
    "NCD - Cardiovascular":                  12,
    "NCD - Cerebrovascular":                 13,
    "NCD - Renal":                           14,
    "NCD - Cancer / Oncology":               15,
    "NCD - Haematology":                     16,
    "NCD - Neurological":                    17,
    "NCD - Mental Health":                   18,
    "NCD - Substance Use":                   19,
    "NCD - Respiratory Chronic":             20,
    "NCD - Digestive":                       21,
    "NCD - Endocrine":                       22,
    "NCD - Musculoskeletal":                 23,
    "NCD - Urological":                      24,
    "NCD - Ophthalmic":                      25,
    "NCD - Dermatological":                  26,
    "Communicable - Malaria":                30,
    "Communicable - Tuberculosis":           31,
    "Communicable - HIV/AIDS":               32,
    "Communicable - Respiratory Infection":  33,
    "Communicable - Gastrointestinal":       34,
    "Communicable - Urinary Tract Infection":35,
    "Communicable - Gynaecological Infection":36,
    "Communicable - Skin Infection":         37,
    "Communicable - Eye Infection":          38,
    "Communicable - Other":                  39,
    "MNCH - Maternal: ANC":                  40,
    "MNCH - Maternal: Obstetric Complication":41,
    "MNCH - Maternal: Intrapartum":          42,
    "MNCH - Maternal: Postnatal":            43,
    "MNCH - Neonatal":                       44,
    "MNCH - Child Health":                   45,
    "Nutrition & Deficiency":                50,
    "Reproductive Health":                   55,
    "Injury & Violence":                     60,
    "Surgical / Procedural":                 65,
    "Dental / Oral Health":                  70,
    "Screening / Prevention":                75,
    "Medication / Investigation":            80,
}

CHRONIC_CATEGORIES: frozenset[str] = frozenset({
    "NCD - Hypertension",
    "NCD - Diabetes",
    "NCD - Cardiovascular",
    "NCD - Cerebrovascular",
    "NCD - Respiratory Chronic",
    "NCD - Renal",
    "NCD - Mental Health",
    "NCD - Cancer / Oncology",
    "NCD - Neurological",
    "NCD - Haematology",
    "NCD - Endocrine",
    "NCD - Musculoskeletal",
    "NCD - Digestive",
    "NCD - Ophthalmic",
    "NCD - Urological",
    "NCD - Dermatological",
    "Communicable - HIV/AIDS",
    "Communicable - Tuberculosis",
    "NCD - Substance Use",
})

CHRONIC_KEYWORDS = re.compile(
    r"\bchronic\b|\blife.?long\b|\blifetime\b|\bpermanent\b"
    r"|\bmaintenance\b|\blong.?term\b|\bon.?treatment\b",
    re.IGNORECASE,
)

_COMPILED = [
    (label, [re.compile(p, re.IGNORECASE) for p in patterns])
    for label, patterns in DISEASE_PATTERNS
]


# ══════════════════════════════════════════════════════════════════════════════
#  SERIES-LEVEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _to_none(value):
    if value is None:
        return None
    s = str(value).strip()
    return None if s in ("", "None", "nan", "NULL") else s


def clean_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[^\x20-\x7E\xA0-\xFF]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned = cleaned.where(cleaned.str.len() > 2, other=None)
    return cleaned


def flag_rule_out(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(lambda t: 1 if RULE_OUT_RE.search(t) else 0)


def flag_invalid(series: pd.Series) -> pd.Series:
    def _check(t):
        if not t or len(t.strip()) < 2:
            return 1
        if INVALID_RE_SIMPLE.match(t):
            return 1
        if INVALID_RE_REPEAT.match(t.strip()):
            return 1
        return 0
    return series.fillna("").apply(_check)


def categorise_series(series: pd.Series) -> pd.Series:
    """Return the highest-priority matching disease category for each text value."""
    def _cat(text: str) -> Optional[str]:
        if not text:
            return None
        expanded = _expand_abbreviations(text)
        best_cat = None
        best_pri = 999
        for label, patterns in _COMPILED:
            for pat in patterns:
                if pat.search(expanded):
                    pri = CATEGORY_PRIORITY.get(label, 99)
                    if pri < best_pri:
                        best_pri = pri
                        best_cat = label
                    break
        return best_cat
    return series.fillna("").apply(_cat)


def split_and_categorise(cleaned: pd.Series) -> pd.DataFrame:
    """
    Expand abbreviations, split on '/', categorise each segment, then resolve
    by CATEGORY_PRIORITY (lowest number = highest clinical priority).
    Returns a DataFrame with columns: primary_disease, comorbidity_1, comorbidity_2.
    """
    def _priority(cat) -> int:
        return CATEGORY_PRIORITY.get(cat, 99)

    def _split(text: str) -> tuple:
        if not text:
            return None, None, None
        expanded = _expand_abbreviations(text)
        parts = [p.strip() for p in expanded.split("/") if p.strip()]
        cats = []
        for part in parts:
            cat = None
            for label, patterns in _COMPILED:
                for pat in patterns:
                    if pat.search(part):
                        cat = label
                        break
                if cat:
                    break
            if cat and cat not in cats:
                cats.append(cat)
        cats.sort(key=_priority)
        return (
            cats[0] if len(cats) > 0 else None,
            cats[1] if len(cats) > 1 else None,
            cats[2] if len(cats) > 2 else None,
        )

    rows = cleaned.fillna("").apply(_split)
    return pd.DataFrame(
        rows.tolist(),
        columns=["primary_disease", "comorbidity_1", "comorbidity_2"],
        index=cleaned.index,
    )


def is_chronic_series(primary_disease: pd.Series, cleaned: pd.Series) -> pd.Series:
    def _check(row):
        cat, text = row
        if cat in CHRONIC_CATEGORIES:
            return 1
        if text and CHRONIC_KEYWORDS.search(str(text)):
            return 1
        return 0
    return pd.Series(
        list(map(_check, zip(primary_disease.fillna(""), cleaned.fillna("")))),
        index=primary_disease.index,
    )


def _parse_nlp_terms(cleaned_text: str) -> List[str]:
    if not cleaned_text:
        return []
    return [t.strip() for t in cleaned_text.split("/") if t.strip()]


def _already_covered(nlp_term: str, icd10_terms: List[str]) -> bool:
    nlp_lower = nlp_term.lower()
    for icd_term in icd10_terms:
        icd_lower = icd_term.lower()
        if nlp_lower in icd_lower or icd_lower in nlp_lower:
            return True
        nlp_words = set(nlp_lower.split())
        icd_words = set(icd_lower.split())
        if nlp_words & icd_words:
            return True
    return False


def build_coalesced_diagnosis(
    raw_diagnosis: Optional[str],
    icd10_name_1: Optional[str],
    icd10_name_2: Optional[str],
    icd10_name_3: Optional[str],
) -> str:
    """
    Coalesce ICD-10 names (authoritative) + NLP gap-fill.
    ICD-10 names come first; NLP terms added only if not already covered.
    """
    icd10_terms = [
        t for t in [icd10_name_1, icd10_name_2, icd10_name_3]
        if t and str(t).strip() not in ("", "None", "nan")
    ]

    if not raw_diagnosis or str(raw_diagnosis).strip() in ("", "None", "nan"):
        return " / ".join(icd10_terms) if icd10_terms else ""

    cleaned = clean_series(pd.Series([raw_diagnosis])).iloc[0]
    if not cleaned:
        return " / ".join(icd10_terms) if icd10_terms else ""

    nlp_terms = _parse_nlp_terms(cleaned)
    gap_fill = [t for t in nlp_terms if not _already_covered(t, icd10_terms)]

    all_terms = icd10_terms + gap_fill
    return " / ".join(all_terms) if all_terms else cleaned


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET TABLE DDL
# ══════════════════════════════════════════════════════════════════════════════

def _target_ddl(table_name: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS `{TARGET_SCHEMA}`.`{table_name}` (
        visit_id                VARCHAR(100) NOT NULL,
        source_schema           VARCHAR(50)  NOT NULL,
        raw_visit_id            INT,
        raw_diagnosis           TEXT,
        nlp_cleaned_diagnosis   TEXT,
        coalesced_diagnosis     TEXT,
        primary_disease         VARCHAR(200),
        comorbidity_1           VARCHAR(200),
        comorbidity_2           VARCHAR(200),
        nlp_primary_icd10_code  VARCHAR(20),
        nlp_comorbidity_1_icd10 VARCHAR(20),
        nlp_comorbidity_2_icd10 VARCHAR(20),
        is_rule_out             TINYINT(1) DEFAULT 0,
        is_invalid              TINYINT(1) DEFAULT 0,
        is_chronic              TINYINT(1) DEFAULT 0,
        clinical_status         VARCHAR(50),
        refreshed_at            DATETIME,
        PRIMARY KEY (visit_id, source_schema),
        INDEX idx_primary_disease  (primary_disease),
        INDEX idx_source_schema    (source_schema),
        INDEX idx_is_chronic       (is_chronic)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """


# ══════════════════════════════════════════════════════════════════════════════
#  WORK TABLES
# ══════════════════════════════════════════════════════════════════════════════

def _build_icd_pivot_work(engine: Engine) -> None:
    """
    Materialise one ICD-10 row per visit from stg_icd10.
    stg_icd10 already has one row per visit (primary code only from merged table).
    We just pull the three name columns directly — no window function needed.
    """
    with engine.begin() as conn:
        conn.execute(text(
            f"DROP TABLE IF EXISTS `{STAGING_SCHEMA}`.`{_WORK_ICD}`"
        ))
        conn.execute(text(f"""
            CREATE TABLE `{STAGING_SCHEMA}`.`{_WORK_ICD}` (
                sk_visit_id   VARCHAR(100) NOT NULL PRIMARY KEY,
                source_schema VARCHAR(50),
                icd10_variation_code_1  VARCHAR(20),
                icd10_variation_name_1  VARCHAR(500),
                icd10_variation_code_2  VARCHAR(20),
                icd10_variation_name_2  VARCHAR(500),
                icd10_variation_code_3  VARCHAR(20),
                icd10_variation_name_3  VARCHAR(500)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """))
        # stg_icd10 has one row per visit (primary code resolved at ingest).
        # icd10_note_code = d1 fallback or structured note code (COALESCE at merge).
        # icd10_variation_name = the human-readable name.
        # Slots 2 and 3 are NULL here — the merged table only carries rank-1.
        # If multiple ICD-10 codes per visit are needed in future, extend
        # evaluation_doctor_notes_merged to carry rank-2 and rank-3.
        conn.execute(text(f"""
            INSERT INTO `{STAGING_SCHEMA}`.`{_WORK_ICD}`
         SELECT
                sk_visit_id,
                source_schema,
                icd10_variation_code   AS icd10_variation_code_1,
                icd10_variation_name   AS icd10_variation_name_1,
                NULL                   AS icd10_variation_code_2,
                NULL                   AS icd10_variation_name_2,
                NULL                   AS icd10_variation_code_3,
                NULL                   AS icd10_variation_name_3
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY sk_visit_id
                        ORDER BY note_created_at ASC, raw_note_id ASC
                    ) AS rn
                FROM TENRI_RAW.STG_ICD10
                WHERE icd10_variation_code IS NOT NULL
            ) ranked
            WHERE rn = 1

        """))
    log.info("  ICD pivot work table ready")


def _build_new_visits_work(engine: Engine, existing_ids: set) -> int:
    """Visits in stg_doctor_notes not yet written to Snowflake target."""
    with engine.begin() as conn:
        conn.execute(text(
            f"DROP TABLE IF EXISTS `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}`"
        ))
        conn.execute(text(f"""
            CREATE TABLE `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}` (
                sk_visit_id VARCHAR(100) NOT NULL PRIMARY KEY
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """))
        conn.execute(text(f"""
            INSERT IGNORE INTO `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}`
            SELECT DISTINCT sk_visit_id
            FROM {SOURCE_NOTES}
            WHERE diagnosis IS NOT NULL AND diagnosis != ''
        """))

    # Remove visits already in Snowflake
    if existing_ids:
        batch_size = 1000
        ids_list = list(existing_ids)
        with engine.begin() as conn:
            for i in range(0, len(ids_list), batch_size):
                batch = ids_list[i:i + batch_size]
                placeholders = ", ".join(f"'{v}'" for v in batch)
                conn.execute(text(
                    f"DELETE FROM `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}` "
                    f"WHERE sk_visit_id IN ({placeholders})"
                ))

    with engine.connect() as conn:
        count = conn.execute(text(
            f"SELECT COUNT(*) FROM `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}`"
        )).scalar() or 0
    log.info("  New visits to process: %d", count)
    return count


def _drop_work_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            f"DROP TABLE IF EXISTS `{STAGING_SCHEMA}`.`{_WORK_ICD}`"
        ))
        conn.execute(text(
            f"DROP TABLE IF EXISTS `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}`"
        ))


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNK QUERY
#  Joins stg_doctor_notes with _work_icd_pivot on sk_visit_id.
#  No 5-table join — icd10_name_1 already resolved at ingest time.
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_query(limit: int, offset: int, incremental: bool) -> str:
    if incremental:
        visit_window = f"""
            SELECT n.sk_visit_id
            FROM {SOURCE_NOTES} n
            JOIN `{STAGING_SCHEMA}`.`{_WORK_NEW_IDS}` nw
              ON nw.sk_visit_id = n.sk_visit_id
            WHERE n.diagnosis IS NOT NULL AND n.diagnosis != ''
            ORDER BY n.sk_visit_id
            LIMIT {limit} OFFSET {offset}
        """
    else:
        visit_window = f"""
            SELECT sk_visit_id
            FROM {SOURCE_NOTES}
            WHERE diagnosis IS NOT NULL AND diagnosis != ''
            ORDER BY sk_visit_id
            LIMIT {limit} OFFSET {offset}
        """

    return f"""
        WITH visit_window AS ({visit_window})
        SELECT
            n.source_schema,
            n.sk_visit_id                                           AS visit_id,
            CAST(SUBSTRING_INDEX(n.sk_visit_id, '|', -1) AS UNSIGNED) AS raw_visit_id,
            n.diagnosis                     AS raw_diagnosis,
            pi.icd10_code_1                 AS nlp_primary_icd10_code,
            pi.icd10_name_1                 AS icd10_name_1,
            pi.icd10_code_2                 AS nlp_comorbidity_1_icd10,
            pi.icd10_name_2                 AS icd10_name_2,
            pi.icd10_code_3                 AS nlp_comorbidity_2_icd10,
            pi.icd10_name_3                 AS icd10_name_3
        FROM {SOURCE_NOTES} n
        JOIN visit_window vw ON vw.sk_visit_id = n.sk_visit_id
        LEFT JOIN `{STAGING_SCHEMA}`.`{_WORK_ICD}` pi
               ON pi.sk_visit_id = n.sk_visit_id
    """


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS CHUNK
# ══════════════════════════════════════════════════════════════════════════════

def _process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    raw = df["raw_diagnosis"].apply(_to_none)
    cleaned = clean_series(raw.fillna(""))

    icd_name_1 = df.get("icd10_name_1", pd.Series([None] * len(df))).apply(_to_none)
    icd_name_2 = df.get("icd10_name_2", pd.Series([None] * len(df))).apply(_to_none)
    icd_name_3 = df.get("icd10_name_3", pd.Series([None] * len(df))).apply(_to_none)

    coalesced = pd.Series([
        build_coalesced_diagnosis(r, i1, i2, i3)
        for r, i1, i2, i3 in zip(
            raw.fillna(""), icd_name_1.fillna(""),
            icd_name_2.fillna(""), icd_name_3.fillna("")
        )
    ], index=df.index)

    cats = split_and_categorise(coalesced)

    is_chronic = is_chronic_series(cats["primary_disease"], coalesced)
    is_rule_out = flag_rule_out(raw.fillna(""))
    is_invalid  = flag_invalid(cleaned)

    _CLINICAL_STATUS_RULES = [
        # (status_label, compiled_regex)
        ("Suspected / Rule Out", re.compile(
            r"\br\s*/\s*o\.?\b|r\.o\.?\b|rule\s*out|^\?|\?\s*[a-z]"
            r"|\bsuspected\b|\bprobable\b|\bpossible\b",
            re.IGNORECASE
        )),
        ("Follow-Up", re.compile(
            r"\bfollow.?up\b|\bf/u\b|\breview\b|\breviewed\b"
            r"|\bmonitoring\b|\bcheck.up\b|\brepeat\b|\brefill\b"
            r"|\bmedication\s+pickup\b",
            re.IGNORECASE
        )),
        ("High-Risk Pregnancy", re.compile(
            r"\bpre.eclampsia\b|\beclampsia\b|\bhellp\b"
            r"|\bhigh.risk\s+preg\b|\bgestational\s+diabetes\b"
            r"|\bantepartum\s+haem\b|\bplacenta\s+praevia\b",
            re.IGNORECASE
        )),
        ("ANC", re.compile(
            r"\bantenatal\b|\bprenatal\b|\bantenatal\s+care\b"
            r"|\bpregnancy\s+review\b|\bpregnancy\s+check\b",
            re.IGNORECASE
        )),
        ("Intrapartum", re.compile(
            r"\blabou?r\b|\bdelivery\b|\bintrapartum\b"
            r"|\bcaesarean\b|\bc.section\b|\bsvd\b|\bnvd\b",
            re.IGNORECASE
        )),
        ("Postnatal", re.compile(
            r"\bpostnatal\b|\bpuerperium\b|\bpostpartum\b|\bpnc\b"
            r"|\bbreastfeeding\b|\blactation\s+problem\b",
            re.IGNORECASE
        )),
        ("Immunisation", re.compile(
            r"\bimmunisation\b|\bimmunization\b|\bvaccinat\b"
            r"|\bepi\b(?!\w)|\bwell.baby\b|\bwell.child\b",
            re.IGNORECASE
        )),
        ("Investigation Only", re.compile(
            r"\binvestigation\s+only\b|\blab\s+result\b|\btest\s+result\b"
            r"|\bx.ray\b|\bultrasound\b|\becho\b(?!\w)|\bct\s+scan\b",
            re.IGNORECASE
        )),
        ("New Presentation", re.compile(r".", re.IGNORECASE)),  # catch-all
    ]

    def _get_clinical_status(cleaned_text: str) -> str:
        if not cleaned_text or not cleaned_text.strip():
            return "Unknown"
        expanded = _expand_abbreviations(cleaned_text)
        for status_label, pat in _CLINICAL_STATUS_RULES:
            if pat.search(expanded):
                return status_label
        return "Unknown"

    status = pd.Series(
        [_get_clinical_status(t) for t in cleaned.fillna("")],
        index=df.index,
    )

    out = pd.DataFrame({
        "visit_id":                 df["visit_id"],
        "source_schema":            df["source_schema"].apply(_to_none),
        "raw_visit_id":             df["raw_visit_id"],
        "raw_diagnosis":            raw,
        "nlp_cleaned_diagnosis":    cleaned,
        "coalesced_diagnosis":      coalesced,
        "primary_disease":          cats["primary_disease"],
        "comorbidity_1":            cats["comorbidity_1"],
        "comorbidity_2":            cats["comorbidity_2"],
        "nlp_primary_icd10_code":   df.get("nlp_primary_icd10_code", pd.Series([None]*len(df))).apply(_to_none),
        "nlp_comorbidity_1_icd10":  df.get("nlp_comorbidity_1_icd10", pd.Series([None]*len(df))).apply(_to_none),
        "nlp_comorbidity_2_icd10":  df.get("nlp_comorbidity_2_icd10", pd.Series([None]*len(df))).apply(_to_none),
        "is_rule_out":              is_rule_out,
        "is_invalid":               is_invalid,
        "is_chronic":               is_chronic,
        "clinical_status":          status,
        "refreshed_at":             datetime.now(),
    })
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  UPSERT
# ══════════════════════════════════════════════════════════════════════════════

def _sf_existing_visit_ids(sf_conn) -> set:
    """Return set of visit_ids already written to Snowflake target table."""
    try:
        cur = sf_conn.cursor()
        cur.execute(
            f"SELECT DISTINCT VISIT_ID "
            f"FROM {SF_TARGET_DB}.{SF_TARGET_SCHEMA}.{TARGET_TABLE.upper()}"
        )
        ids = {row[0] for row in cur.fetchall()}
        cur.close()
        return ids
    except Exception:
        return set()


def _write_chunk_sf(sf_conn, df: pd.DataFrame, first_chunk: bool, full: bool) -> None:
    """Write a processed chunk to Snowflake TENRI_RAW."""
    if df.empty:
        return
    df_sf = df.copy()
    df_sf.columns = [c.upper() for c in df_sf.columns]
    # Convert datetime columns to string so pyarrow doesn't mistype them
    dt_cols = [c for c in df_sf.columns if pd.api.types.is_datetime64_any_dtype(df_sf[c])]
    for col in dt_cols:
        df_sf[col] = df_sf[col].astype(str).replace("NaT", None)
    # Force all object columns to string so pyarrow doesn't misidentify
    # mixed-null string columns (e.g. ICD10 codes) as FIXED/numeric
    for col in df_sf.select_dtypes(include=["object"]).columns:
        df_sf[col] = df_sf[col].apply(lambda x: None if pd.isnull(x) else str(x))
    log.info("  [DEBUG Step3] dtypes before write_pandas:\n%s", df_sf.dtypes.to_string())
    write_pandas(
        conn=sf_conn,
        df=df_sf,
        table_name=TARGET_TABLE.upper(),
        database=SF_TARGET_DB,
        schema=SF_TARGET_SCHEMA,
        auto_create_table=first_chunk,
        overwrite=(full and first_chunk),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def refresh_nlp(engine: Engine, full: bool = False, sf_conn=None) -> None:
    """
    Entry point called from run_pipeline.py Step 3.
    full=True: process all visits and overwrite Snowflake target.
    full=False: process only visits not yet in Snowflake (incremental).
    """
    log.info("=" * 60)
    log.info("Step 3: NLP cleaning (%s)", "FULL" if full else "INCREMENTAL")
    log.info("=" * 60)

    incremental = not full

    try:
        _build_icd_pivot_work(engine)

        if incremental:
            existing_ids = _sf_existing_visit_ids(sf_conn) if sf_conn else set()
            new_count = _build_new_visits_work(engine, existing_ids)
            if new_count == 0:
                log.info("  Nothing to process — target is up to date")
                return
        else:
            with engine.connect() as conn:
                new_count = conn.execute(text(
                    f"SELECT COUNT(DISTINCT sk_visit_id) FROM {SOURCE_NOTES} "
                    f"WHERE diagnosis IS NOT NULL AND diagnosis != ''"
                )).scalar() or 0
            log.info("  Total visits to process: %d", new_count)

        processed = 0
        offset = 0
        first_chunk = True
        while True:
            sql = _chunk_query(CHUNK_SIZE, offset, incremental)
            with engine.connect() as conn:
                chunk = pd.read_sql(text(sql), conn)
            if chunk.empty:
                break

            result = _process_chunk(chunk)
            if sf_conn:
                _write_chunk_sf(sf_conn, result, first_chunk, full)
            first_chunk = False

            processed += len(chunk)
            offset += CHUNK_SIZE
            log.info("  Processed %d / %d", min(processed, new_count), new_count)

    finally:
        _drop_work_tables(engine)

    log.info("=" * 60)
    log.info("Step 3 complete — %d rows written to Snowflake %s.%s",
             processed, SF_TARGET_SCHEMA, TARGET_TABLE)
    log.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Xana: NLP diagnosis cleaning")
    parser.add_argument("--full", action="store_true", help="Full rebuild")
    args = parser.parse_args()
    engine = build_mysql_engine()
    refresh_nlp(engine, full=args.full)