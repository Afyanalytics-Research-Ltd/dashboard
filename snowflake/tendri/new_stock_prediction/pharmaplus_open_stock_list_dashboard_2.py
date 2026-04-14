"""
Pharmaplus Chain Pharmacies — Afya Analytics Platform
======================================================
Tab 1: Market Opportunity
Tab 2: Customer Spend Behaviour
Tab 3: Opening Stock Recommendations

Run:  streamlit run pharmaplus_dashboard_v2.py
Data: data_export.pkl from pharmaplus_model notebook
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import plotly.io as pio

# ── GLOBAL PLOTLY TEMPLATE — all text Afya Blue ───────────────────────────────
pio.templates["afya"] = pio.templates["plotly_white"]
pio.templates["afya"].layout.font = dict(
    family="Montserrat, sans-serif", color="#0072CE", size=11)
pio.templates["afya"].layout.legend.font = dict(
    family="Montserrat, sans-serif", color="#0072CE", size=10)
pio.templates["afya"].layout.xaxis.tickfont  = dict(color="#0072CE", size=10)
pio.templates["afya"].layout.xaxis.title.font = dict(color="#0072CE", size=11)
pio.templates["afya"].layout.yaxis.tickfont  = dict(color="#0072CE", size=10)
pio.templates["afya"].layout.yaxis.title.font = dict(color="#0072CE", size=11)
pio.templates["afya"].layout.xaxis.gridcolor = "#EBF3FB"
pio.templates["afya"].layout.yaxis.gridcolor = "#EBF3FB"
pio.templates["afya"].layout.paper_bgcolor   = "#fff"
pio.templates["afya"].layout.plot_bgcolor    = "#fff"
pio.templates.default = "afya"

# ── THERAPEUTIC GROUP MAP ─────────────────────────────────────────────────────
THERAPEUTIC_GROUP_MAP = {
    "Analgesic / antipyretic":"Analgesics","NSAID":"Analgesics","NSAID + analgesic":"Analgesics",
    "NSAID + analgesic combination":"Analgesics","NSAID + enzyme":"Analgesics",
    "NSAID + muscle relaxant":"Analgesics","NSAID — COX-2 selective":"Analgesics",
    "NSAID — fenamate":"Analgesics","NSAID — injectable":"Analgesics","NSAID — ketoprofen":"Analgesics",
    "NSAID — lornoxicam":"Analgesics","NSAID — topical":"Analgesics","Topical NSAID":"Analgesics",
    "Topical analgesic":"Analgesics","Topical analgesic — cooling":"Analgesics",
    "Topical analgesic — warming":"Analgesics","Topical anti-inflammatory":"Analgesics",
    "Enzyme anti-inflammatory":"Analgesics","Enzyme + analgesic":"Analgesics",
    "Non-opioid analgesic":"Analgesics","Opioid analgesic (mild)":"Analgesics",
    "Opioid analgesic (strong)":"Analgesics","Opioid analgesic (weak)":"Analgesics",
    "Opioid + antiemetic combination":"Analgesics","Opioid antagonist":"Analgesics",
    "Sedative / analgesic":"Analgesics","Muscle relaxant":"Analgesics",
    "Topical muscle relaxant":"Analgesics",
    "Antibiotic / antiprotozoal":"Antibiotics","Antibiotic — aminoglycoside":"Antibiotics",
    "Antibiotic — aminopenicillin":"Antibiotics","Antibiotic — beta-lactam combination":"Antibiotics",
    "Antibiotic — cephalosporin (1st gen)":"Antibiotics","Antibiotic — cephalosporin (2nd gen)":"Antibiotics",
    "Antibiotic — cephalosporin (3rd gen)":"Antibiotics","Antibiotic — fluoroquinolone":"Antibiotics",
    "Antibiotic — glycopeptide":"Antibiotics","Antibiotic — lincosamide":"Antibiotics",
    "Antibiotic — macrolide":"Antibiotics","Antibiotic — nitrofuran":"Antibiotics",
    "Antibiotic — oxazolidinone":"Antibiotics","Antibiotic — penicillin combination":"Antibiotics",
    "Antibiotic — penicillin injectable":"Antibiotics","Antibiotic — penicillinase-resistant":"Antibiotics",
    "Antibiotic — rifamycin":"Antibiotics","Antibiotic — sulfonamide combination":"Antibiotics",
    "Antibiotic — tetracycline":"Antibiotics","Antibiotic — topical":"Antibiotics",
    "Antibiotic — topical combination":"Antibiotics","Antimicrobial — topical":"Antibiotics",
    "Antimicrobial — topical silver":"Antibiotics",
    "Antihypertensive — ACE + diuretic":"Antihypertensives","Antihypertensive — ACE+CCB combo":"Antihypertensives",
    "Antihypertensive — ACE+diuretic":"Antihypertensives","Antihypertensive — ARB":"Antihypertensives",
    "Antihypertensive — ARB + diuretic":"Antihypertensives","Antihypertensive — ARB+CCB combo":"Antihypertensives",
    "Antihypertensive — CCB":"Antihypertensives","Antihypertensive — beta-blocker":"Antihypertensives",
    "Antihypertensive — beta-blocker combination":"Antihypertensives",
    "Antihypertensive — central":"Antihypertensives","Antihypertensive — combination":"Antihypertensives",
    "Antihypertensive — diuretic":"Antihypertensives","Antihypertensive — injectable":"Antihypertensives",
    "Antihypertensive — vasodilator":"Antihypertensives","Beta-blocker":"Antihypertensives",
    "Beta-blocker (alpha + beta)":"Antihypertensives","Diuretic — combination":"Antihypertensives",
    "Diuretic — loop":"Antihypertensives","Diuretic — potassium sparing":"Antihypertensives",
    "Anti-anginal — ranolazine":"Antihypertensives","Antiarrhythmic":"Antihypertensives",
    "Cardiac glycoside":"Antihypertensives","Cerebral vasodilator":"Antihypertensives",
    "Vascular tonic":"Antihypertensives",
    "Biguanide — antidiabetic":"Antidiabetics","Sulfonylurea — antidiabetic":"Antidiabetics",
    "DPP-4 inhibitor":"Antidiabetics","DPP-4 inhibitor + biguanide":"Antidiabetics",
    "DPP-4 inhibitor — antidiabetic":"Antidiabetics","DPP-4 + biguanide combination":"Antidiabetics",
    "Dipeptidyl peptidase inhibitor":"Antidiabetics","SGLT2 inhibitor — antidiabetic":"Antidiabetics",
    "Insulin":"Antidiabetics","Antidiabetic supplement":"Antidiabetics",
    "Lipase inhibitor — antiobesity":"Antidiabetics",
    "Antimalarial":"Antimalarials","Antimalarial / DMARD":"Antimalarials",
    "Antimalarial — ACT":"Antimalarials","Antimalarial — SP":"Antimalarials",
    "Antiviral — topical":"Antivirals & Antifungals","Antiviral — topical/oral/IV":"Antivirals & Antifungals",
    "Antifungal — allylamine":"Antivirals & Antifungals","Antifungal — azole":"Antivirals & Antifungals",
    "Antifungal — imidazole":"Antivirals & Antifungals","Antifungal — polyene":"Antivirals & Antifungals",
    "Antifungal — topical":"Antivirals & Antifungals","Antifungal — triazole":"Antivirals & Antifungals",
    "Antifungal — vaginal":"Antivirals & Antifungals","Topical antifungal":"Antivirals & Antifungals",
    "Topical antifungal + steroid":"Antivirals & Antifungals",
    "Topical antifungal combination":"Antivirals & Antifungals","Anthelmintic":"Antivirals & Antifungals",
    "Anthelmintic / immunomodulator":"Antivirals & Antifungals",
    "Proton pump inhibitor (PPI)":"GI Agents","H2 antagonist":"GI Agents","Antacid":"GI Agents",
    "Antacid / GI preparation":"GI Agents","Antacid / alginate":"GI Agents",
    "Antacid suspension":"GI Agents","GI mucosal protectant":"GI Agents","Antispasmodic":"GI Agents",
    "Antispasmodic drops":"GI Agents","Antispasmodic — IBS":"GI Agents",
    "Antiemetic — 5-HT3 antagonist":"GI Agents","Antiemetic — dopamine antagonist":"GI Agents",
    "Antidiarrhoeal":"GI Agents","Adsorbent / antidiarrhoeal":"GI Agents","Antiflatulent":"GI Agents",
    "Prokinetic":"GI Agents","Digestive enzyme":"GI Agents","Laxative — stimulant":"GI Agents",
    "Hepatoprotective":"GI Agents","Oral antiseptic gel":"GI Agents",
    "Gingival hyaluronic acid gel":"GI Agents",
    "Bronchodilator — SABA":"Respiratory","Bronchodilator — LABA":"Respiratory",
    "Bronchodilator + corticosteroid combination":"Respiratory","LABA + ICS combination":"Respiratory",
    "Leukotriene antagonist":"Respiratory","Corticosteroid — inhaled":"Respiratory",
    "Corticosteroid — inhaled / nasal":"Respiratory","Intranasal corticosteroid":"Respiratory",
    "Intranasal corticosteroid + antihistamine":"Respiratory","Expectorant":"Respiratory",
    "Expectorant / bronchodilator":"Respiratory","Expectorant / mucolytic":"Respiratory",
    "Mucolytic":"Respiratory","Cough preparation":"Respiratory","Cough suppressant":"Respiratory",
    "Cold / flu preparation":"Respiratory","Nasal decongestant":"Respiratory",
    "Nasal drops / decongestant":"Respiratory","Nasal saline":"Respiratory",
    "SSRI antidepressant":"CNS & Mental Health","SNRI antidepressant":"CNS & Mental Health",
    "Tricyclic antidepressant":"CNS & Mental Health","Antidepressant — NaSSA":"CNS & Mental Health",
    "Antidepressant + antipsychotic combo":"CNS & Mental Health","Antipsychotic":"CNS & Mental Health",
    "Antipsychotic / antidepressant":"CNS & Mental Health","Antipsychotic — atypical":"CNS & Mental Health",
    "Antipsychotic — injectable depot":"CNS & Mental Health","Antipsychotic — typical":"CNS & Mental Health",
    "Anticonvulsant":"CNS & Mental Health","Anticonvulsant / neuropathic pain":"CNS & Mental Health",
    "Anticonvulsant / sedative":"CNS & Mental Health","Benzodiazepine":"CNS & Mental Health",
    "Benzodiazepine — anticonvulsant":"CNS & Mental Health","Antivertigo":"CNS & Mental Health",
    "Antiparkinsonian":"CNS & Mental Health","CNS stimulant":"CNS & Mental Health",
    "Nootropic supplement":"CNS & Mental Health","Triptan — antimigraine":"CNS & Mental Health",
    "Anticholinergic":"CNS & Mental Health","Acetylcholinesterase inhibitor":"CNS & Mental Health",
    "Cholinergic agonist":"CNS & Mental Health","Antiplatelet + antidepressant":"CNS & Mental Health",
    "Nerve growth factor supplement":"CNS & Mental Health",
    "Joint supplement":"Musculoskeletal","Joint supplement + anti-inflammatory":"Musculoskeletal",
    "Bone health supplement":"Musculoskeletal","Bisphosphonate":"Musculoskeletal",
    "Bisphosphonate — injectable":"Musculoskeletal","Xanthine oxidase inhibitor":"Musculoskeletal",
    "Uricosuric / xanthine oxidase":"Musculoskeletal","DMARD":"Musculoskeletal",
    "Antihistamine":"Antihistamines & Allergy","Antihistamine + decongestant":"Antihistamines & Allergy",
    "Antihistamine + leukotriene combo":"Antihistamines & Allergy",
    "Antihistamine / antivertigo":"Antihistamines & Allergy",
    "Antihistamine nasal spray":"Antihistamines & Allergy",
    "Antihistamine — 1st generation":"Antihistamines & Allergy",
    "Antihistamine — 2nd generation":"Antihistamines & Allergy",
    "Decongestant + antihistamine":"Antihistamines & Allergy",
    "Decongestant / antihistamine":"Antihistamines & Allergy",
    "Combined oral contraceptive":"Hormones & Contraceptives",
    "Emergency contraceptive":"Hormones & Contraceptives",
    "Injectable contraceptive / steroid":"Hormones & Contraceptives","Progestogen":"Hormones & Contraceptives",
    "Progestogen supplement":"Hormones & Contraceptives","Progestogen — oral":"Hormones & Contraceptives",
    "Oestrogen — HRT":"Hormones & Contraceptives","Menopause supplement":"Hormones & Contraceptives",
    "Uterotonic":"Hormones & Contraceptives","Thyroid hormone":"Hormones & Contraceptives",
    "GnRH agonist — injectable depot":"Hormones & Contraceptives",
    "Phytogenic uterine tonic":"Hormones & Contraceptives","Prostaglandin":"Hormones & Contraceptives",
    "Corticosteroid — oral":"Corticosteroids","Corticosteroid — oral/injectable":"Corticosteroids",
    "Corticosteroid — injectable":"Corticosteroids","Corticosteroid + antihistamine":"Corticosteroids",
    "Corticosteroid + keratolytic":"Corticosteroids",
    "Corticosteroid — topical":"Dermatologicals","Corticosteroid — topical (mild)":"Dermatologicals",
    "Corticosteroid — topical (potent)":"Dermatologicals",
    "Corticosteroid — topical (very potent)":"Dermatologicals",
    "Ophthalmic / topical corticosteroid":"Dermatologicals",
    "Topical calcineurin inhibitor + steroid":"Dermatologicals",
    "Topical corticosteroid":"Dermatologicals","Topical corticosteroid combination":"Dermatologicals",
    "Emollient":"Dermatologicals","Emollient / antimicrobial":"Dermatologicals",
    "Emollient / skin barrier cream":"Dermatologicals","Retinoid — topical":"Dermatologicals",
    "Depigmenting agent":"Dermatologicals","Scar treatment":"Dermatologicals",
    "Topical anti-infective":"Dermatologicals","Topical antiseptic — ear":"Dermatologicals",
    "Topical skin preparation":"Dermatologicals","Topical oral analgesic / antiseptic":"Dermatologicals",
    "Topical anticoagulant":"Dermatologicals","Hyaluronic acid gel":"Dermatologicals",
    "Vitamin E topical":"Dermatologicals","Wound healing — topical":"Dermatologicals",
    "Wound debriding agent":"Dermatologicals","Wound hydrogel":"Dermatologicals",
    "Hair growth supplement":"Dermatologicals","Skin / hair / nail supplement":"Dermatologicals",
    "Ophthalmic — anti-infective":"Ophthalmics","Ophthalmic — antiallergic":"Ophthalmics",
    "Ophthalmic — antibiotic":"Ophthalmics","Ophthalmic — antibiotic + steroid":"Ophthalmics",
    "Ophthalmic — antifungal":"Ophthalmics","Ophthalmic — antihistamine":"Ophthalmics",
    "Ophthalmic — antiseptic":"Ophthalmics","Ophthalmic — beta-blocker":"Ophthalmics",
    "Ophthalmic — corticosteroid":"Ophthalmics","Ophthalmic — cycloplegic":"Ophthalmics",
    "Ophthalmic — fluoroquinolone":"Ophthalmics","Ophthalmic — glaucoma":"Ophthalmics",
    "Ophthalmic — glaucoma combination":"Ophthalmics","Ophthalmic — lubricant":"Ophthalmics",
    "Ophthalmic — NSAID":"Ophthalmics","Ophthalmic — prostaglandin":"Ophthalmics",
    "Local anaesthetic — ophthalmic":"Ophthalmics","Eye health supplement":"Ophthalmics",
    "Carbonic anhydrase inhibitor":"Ophthalmics",
    "Ear drops":"Ear, Nose & Throat","Ear wax softener":"Ear, Nose & Throat",
    "Otic — antibiotic + steroid":"Ear, Nose & Throat",
    "Calcium supplement":"Vitamins & Supplements","Calcium + vitamin D":"Vitamins & Supplements",
    "Calcium + vitamin D + magnesium":"Vitamins & Supplements",
    "Calcium + vitamin D supplement":"Vitamins & Supplements",
    "Calcium + multivitamin":"Vitamins & Supplements","Iron supplement":"Vitamins & Supplements",
    "Iron + folic acid":"Vitamins & Supplements","Iron + haematinic":"Vitamins & Supplements",
    "Magnesium supplement":"Vitamins & Supplements","Potassium supplement":"Vitamins & Supplements",
    "Vitamin B complex":"Vitamins & Supplements","Vitamin B complex / neuropathy":"Vitamins & Supplements",
    "Vitamin B complex — injectable":"Vitamins & Supplements",
    "Vitamin B complex — neuropathy":"Vitamins & Supplements",
    "Vitamin B12 — injectable":"Vitamins & Supplements","Vitamin E supplement":"Vitamins & Supplements",
    "Vitamin supplement":"Vitamins & Supplements","Vitamins & Supplements":"Vitamins & Supplements",
    "Multivitamin supplement":"Vitamins & Supplements","Multivitamin supplement — male":"Vitamins & Supplements",
    "Multivitamin — chewable":"Vitamins & Supplements","Multivitamin — neurological":"Vitamins & Supplements",
    "Multivitamin — paediatric":"Vitamins & Supplements","Multivitamin / omega":"Vitamins & Supplements",
    "Micronutrient supplement":"Vitamins & Supplements","Omega fatty acid supplement":"Vitamins & Supplements",
    "Prenatal omega supplement":"Vitamins & Supplements","Prenatal vitamin supplement":"Vitamins & Supplements",
    "Paediatric multivitamin":"Vitamins & Supplements","Paediatric vitamin D drops":"Vitamins & Supplements",
    "Probiotic":"Vitamins & Supplements","Nutritional supplement":"Vitamins & Supplements",
    "Nutritional supplement — fertility":"Vitamins & Supplements",
    "Inositol + micronutrient":"Vitamins & Supplements",
    "Appetite stimulant supplement":"Vitamins & Supplements","Cranberry supplement":"Vitamins & Supplements",
    "Female health supplement":"Vitamins & Supplements","Fertility supplement":"Vitamins & Supplements",
    "Male fertility supplement":"Vitamins & Supplements",
    "Anticoagulant":"Anticoagulants","Anticoagulant — DOAC":"Anticoagulants",
    "Anticoagulant — LMWH":"Anticoagulants","Anticoagulant — vitamin K antagonist":"Anticoagulants",
    "Antiplatelet":"Anticoagulants","Antifibrinolytic":"Anticoagulants","Haemostatic":"Anticoagulants",
    "Local anaesthetic":"Anaesthetics","Intravenous anaesthetic":"Anaesthetics",
    "Dissociative anaesthetic":"Anaesthetics","Neuromuscular blocking agent":"Anaesthetics",
    "Antimuscarinic — bladder":"Urology","Urinary antispasmodic":"Urology",
    "Alpha-blocker — uroselective":"Urology","Alpha-blocker + 5-alpha-reductase":"Urology",
    "Alpha-blocker combination":"Urology","PDE5 inhibitor":"Urology",
    "Phosphodiesterase inhibitor":"Urology",
    "IV fluid — carbohydrate":"IV & Hospital Fluids","IV fluid / nasal saline":"IV & Hospital Fluids",
    "IV colloid — volume expansion":"IV & Hospital Fluids","IV diluent":"IV & Hospital Fluids",
    "IV nutrition — total parenteral":"IV & Hospital Fluids",
    "Lung surfactant — injectable":"IV & Hospital Fluids","Vasopressor / inotrope":"IV & Hospital Fluids",
    "Vasopressor / decongestant":"IV & Hospital Fluids",
    "Erythropoietin — injectable":"IV & Hospital Fluids","Immunoglobulin — anti-D":"IV & Hospital Fluids",
    "Sympathomimetic":"IV & Hospital Fluids",
    "Alkylating agent":"Oncology","Taxane — antineoplastic":"Oncology",
    "Platinum — antineoplastic":"Oncology","Antimetabolite":"Oncology",
    "Antimetabolite — oral":"Oncology","Immunosuppressant":"Oncology","Sclerosant":"Oncology",
    "Statin":"Statins & Lipid",
    "Beauty Products":"Beauty Products","Body Building":"Body Building",
}

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
LOGO_PATH = None
for _n in ["pharmaplus_logo.jpg","pharmaplus_logo.png","afya_logo.png"]:
    _p = os.path.join(os.path.dirname(__file__), _n)
    if os.path.exists(_p): LOGO_PATH = _p; break
logo_img = Image.open(LOGO_PATH) if LOGO_PATH else None

st.set_page_config(
    page_title="Afya Analytics · Pharmaplus",
    page_icon=logo_img or "💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PALETTE ───────────────────────────────────────────────────────────────────
AFYA_BLUE = "#0072CE"
TEAL      = "#0BB99F"
COOL_BLUE = "#003467"
ORANGE    = "#f5a623"
CORAL     = "#e05c5c"
PURPLE    = "#7b5ea7"
GRAY      = "#adb5bd"
MUTED     = "#0072CE"
BORDER    = "#cce0f5"
BG_LIGHT  = "#F4F8FC"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, CORAL, PURPLE, ORANGE]

NEW_CATS  = ["Beauty Products","Vitamins & Supplements","Body Building"]
NEW_COLOR = {"Beauty Products":TEAL,"Vitamins & Supplements":AFYA_BLUE,"Body Building":ORANGE}
NEW_ICON  = {"Beauty Products":"✨","Vitamins & Supplements":"💊","Body Building":"💪"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
:root{color-scheme:light only!important}
html,body,[class*="css"],[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:#fff!important;color:#003467!important;
  font-family:'Montserrat',sans-serif!important;color-scheme:light!important}
[data-testid="stSidebar"]{background:#003467!important;border-right:none!important}
[data-testid="stSidebar"] *{color:#fff!important}
.stTabs [data-baseweb="tab-list"]{background:#F4F8FC;border-radius:10px;padding:4px;
  border:1px solid #D6E4F0;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#0072CE;
  font-size:.83rem;font-weight:600;padding:.4rem 1.2rem;border:none;
  font-family:'Montserrat',sans-serif}
.stTabs [aria-selected="true"]{background:#0072CE!important;color:#fff!important}
.kcard{background:#fff;border:1.5px solid #D6E4F0;border-radius:10px;
  padding:.9rem 1rem .7rem;box-shadow:0 2px 8px rgba(0,114,206,.06)}
.chart-card{background:#fff;border:1px solid #D6E4F0;border-radius:10px;
  padding:1rem 1.25rem;margin-bottom:.75rem;box-shadow:0 1px 4px rgba(0,114,206,.05)}
.card-title{font-size:.72rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
  color:#0072CE;margin-bottom:.65rem;font-family:'Montserrat',sans-serif}
.sh{font-size:.68rem;font-weight:800;color:#0072CE;text-transform:uppercase;
  letter-spacing:2.5px;padding:6px 0;border-bottom:2px solid #EBF3FB;margin:1.4rem 0 .9rem}
.sh-teal{font-size:.68rem;font-weight:800;color:#0BB99F;text-transform:uppercase;
  letter-spacing:2.5px;padding:6px 0;border-bottom:2px solid #d0f5f0;margin:1.4rem 0 .9rem}
.info-strip{padding:10px 14px;background:#F4F8FC;border-left:3px solid #0072CE;
  border-radius:4px;font-size:12px;color:#0072CE;margin-bottom:10px}
.ext-signal{padding:12px 16px;background:#FFF8EC;border:1px solid #f5a62340;
  border-left:4px solid #f5a623;border-radius:8px;margin-bottom:12px}
#MainMenu,footer,header{visibility:hidden}

/* ── Widget labels — sliders, selectboxes, text inputs, expanders ── */
[data-testid="stSlider"] label p,
[data-testid="stSlider"] label,
div[data-testid="stSliderLabel"] p,
[data-testid="stSelectbox"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label p,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label p,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label,
[data-testid="stMultiSelect"] label p,
[data-testid="stMultiSelect"] label,
[data-testid="stDateInput"] label p,
[data-testid="stDateInput"] label,
.stSlider label, .stSelectbox label,
.stTextInput label, .stNumberInput label,
.stExpander summary p,
.stExpander [data-testid="stExpanderToggleIcon"] ~ div p,
div[class*="stSlider"] p,
div[class*="stSelectbox"] p,
div[class*="stTextInput"] p,
div[class*="stNumberInput"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"],
p[kind="label"] {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
  font-weight:600!important;
}

/* Slider value bubble and track */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] output {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
}

/* Selectbox / input text inside box */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stTextInput"] input {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
}

/* Expander header — fix icon overlap and color */
[data-testid="stExpander"] details summary {
  display:flex!important;
  align-items:center!important;
  gap:8px!important;
  list-style:none!important;
}
[data-testid="stExpander"] details summary::-webkit-details-marker {
  display:none!important;
}
[data-testid="stExpander"] details summary::marker {
  display:none!important;
  content:""!important;
}
[data-testid="stExpander"] details summary svg {
  flex-shrink:0!important;
  color:#0072CE!important;
  fill:#0072CE!important;
}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] summary > div > p,
[data-testid="stExpander"] summary > div > span,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] + div p {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
  font-weight:700!important;
  margin:0!important;
  padding:0!important;
}

/* Slider thumb — fill circle with Afya Blue */
[data-testid="stSlider"] div[role="slider"] {
  background-color:#0072CE!important;
  border-color:#0072CE!important;
}

/* Slider filled portion of track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] > div > div {
  background-color:#0072CE!important;
}

/* Slider value numbers and labels */
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
  font-weight:600!important;
}


/* Expander — force flex layout to prevent arrow/text overlap */
[data-testid="stExpander"] details > summary {
  display:flex!important;
  flex-direction:row!important;
  align-items:center!important;
  justify-content:space-between!important;
  padding:0.75rem 1rem!important;
  cursor:pointer!important;
  list-style:none!important;
}
[data-testid="stExpander"] details > summary > div {
  flex:1!important;
}
[data-testid="stExpander"] details > summary > div p {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
  font-weight:700!important;
  font-size:.88rem!important;
  margin:0!important;
}
[data-testid="stExpander"] details > summary > svg {
  flex-shrink:0!important;
  width:18px!important;
  height:18px!important;
  color:#0072CE!important;
  fill:#0072CE!important;
}

/* Toggle switch — Afya Blue */
[data-testid="stToggle"] label p,
[data-testid="stToggle"] label span,
[data-testid="stToggle"] p {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
  font-weight:700!important;
  font-size:.85rem!important;
}
[data-testid="stToggle"] input:checked + div {
  background-color:#0072CE!important;
}
/* Native range input thumb fallback */
input[type="range"]::-webkit-slider-thumb {
  background:#0072CE!important;
  border-color:#0072CE!important;
  -webkit-appearance:none!important;
}
input[type="range"]::-moz-range-thumb {
  background:#0072CE!important;
  border-color:#0072CE!important;
}
input[type="range"]::-webkit-slider-runnable-track {
  background:linear-gradient(#0072CE,#0072CE) no-repeat!important;
}

/* Dataframe column headers */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] .dvn-scroller th {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
  font-weight:700!important;
}

/* Caption / help text */
[data-testid="stCaptionContainer"] p,
small, .stCaption {
  color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt_ksh(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if abs(v) >= 1_000_000: return f"KES {v/1_000_000:.1f}M"
    if abs(v) >= 1_000:     return f"KES {v/1_000:.1f}K"
    return f"KES {v:,.0f}"

CHART_LAYOUT = dict(
    plot_bgcolor="#fff",paper_bgcolor="#fff",
    font=dict(family="Montserrat, sans-serif",size=11,color="#0072CE"),
    margin=dict(t=10,b=10,l=0,r=10),
    legend=dict(
        orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
        font=dict(family="Montserrat, sans-serif",size=10,color="#0072CE"),
        title=dict(font=dict(family="Montserrat, sans-serif",size=10,color="#0072CE")),
        bgcolor="rgba(0,0,0,0)",
    ),
    colorway=["#0BB99F","#0072CE","#003467","#f5a623","#e05c5c","#7b5ea7"],
)
AXIS = dict(
    showgrid=True,gridcolor="#EBF3FB",zeroline=False,
    color="#0072CE",
    tickfont=dict(color="#0072CE",size=10,family="Montserrat, sans-serif"),
    title_font=dict(color="#0072CE",size=11,family="Montserrat, sans-serif"),
    title_standoff=8,
)

def kpi_card(col, label, value, accent, sub=""):
    col.markdown(f"""
    <div class="kcard" style="border-color:{accent}">
      <div style="color:{MUTED};font-size:.6rem;font-weight:700;letter-spacing:.12em;
                  text-transform:uppercase;margin-bottom:.35rem">{label}</div>
      <div style="color:{COOL_BLUE};font-size:1.55rem;font-weight:800;line-height:1">{value}</div>
      <div style="color:{accent};font-size:.68rem;margin-top:.3rem;font-weight:600">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sh(text, teal=False):
    cls = "sh-teal" if teal else "sh"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def info(text, color=AFYA_BLUE):
    st.markdown(
        f'<div class="info-strip" style="border-color:{color}">{text}</div>',
        unsafe_allow_html=True)

def ext_signal(source, stat, insight):
    st.markdown(f"""
    <div class="ext-signal">
      <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                  letter-spacing:.1em;color:{ORANGE};margin-bottom:4px">🌐 External Signal — {source}</div>
      <div style="font-size:.9rem;font-weight:700;color:{COOL_BLUE};margin-bottom:3px">{stat}</div>
      <div style="font-size:.78rem;color:rgba(0,52,103,.65);line-height:1.55">{insight}</div>
    </div>""", unsafe_allow_html=True)

# ── GENERIC DETECTION (for pharma product names) ──────────────────────────────
GENERIC_TERMS = [
    "amoxicillin","ampicillin","azithromycin","ciprofloxacin","metronidazole",
    "doxycycline","cotrimoxazole","artemether","quinine","metformin","glibenclamide",
    "insulin","amlodipine","atenolol","losartan","lisinopril","omeprazole",
    "pantoprazole","ibuprofen","paracetamol","diclofenac","salbutamol","prednisolone",
    "vitamin c","vitamin d","zinc sulfate","calcium carbonate","magnesium oxide",
    "omega 3","fish oil","multivitamin","biotin","folic acid","ferrous sulfate",
    "b complex","cod liver oil","collagen peptide","glycerin","petroleum jelly",
    "aloe vera gel","salicylic acid","benzoyl peroxide","whey protein concentrate",
    "creatine monohydrate","bcaa","glutamine powder",
]
KNOWN_BRANDS = [
    "augmentin","coartem","flagyl","ciproxin","zithromax","glucophage","norvasc",
    "cozaar","zestril","losec","voltaren","redoxon","berocca","centrum","seven seas",
    "supradyn","vitabiotics","perfectil","nourkrin","neocell","neurobion","becosules",
    "slow-mag","blackmores","solgar","nature made","nordic naturals","nivea",
    "neutrogena","cerave","olay","vaseline","la roche-posay","bioderma","eucerin",
    "garnier","maybelline","revlon","loreal","mac cosmetics","nyx","rimmel","cantu",
    "dark and lovely","pantene","dove","sunsilk","schwarzkopf","tresemme",
    "the ordinary","cosrx","optimum nutrition","usn","bsn","muscletech","evox",
    "dymatize","myprotein","serious mass","scivation","cellucor","ghost",
]

def infer_generic(product_name):
    name = str(product_name).lower()
    if any(b in name for b in KNOWN_BRANDS): return False
    if any(g in name for g in GENERIC_TERMS): return True
    return None  # unknown — exclude from analysis

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="color:rgba(255,255,255,.55);font-size:.65rem;font-weight:800;'
                'letter-spacing:.15em;text-transform:uppercase">PHARMAPLUS · AFYA ANALYTICS</span>',
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,.12);margin:.6rem 0'>",
                unsafe_allow_html=True)
    PKL_PATH = st.text_input(
        "Data file path",
        value="C:/Users/Mercy/Documents/Tendri/Snowflake Pulls/Xana/snowflake/tendri/pickle_file/data_export.pkl",
        label_visibility="visible",
    )
    if st.button("↺  Reload data"):
        st.cache_data.clear(); st.rerun()
    st.markdown("<hr style='border-color:rgba(255,255,255,.12);margin:.6rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<span style="color:rgba(255,255,255,.4);font-size:.65rem">Afya Analytics Platform</span>',
                unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pkl(path):
    with open(path,"rb") as f: return pickle.load(f)

if not os.path.exists(PKL_PATH):
    st.error(f"Data file not found: `{PKL_PATH}`. Run the notebook and save data_export.pkl.")
    st.stop()

with st.spinner("Loading data…"):
    data = load_pkl(PKL_PATH)

disp        = data["disp"].copy()
inv         = data["inv"].copy()
pat         = data["pat"].copy()
diag_df     = data["diag_df"].copy()
disp_df     = data["disp_df"].copy()
pred_output = data["pred"].copy()
product_intel = data.get("product_intel", pd.DataFrame())

disp["date"]         = pd.to_datetime(disp["date"])
inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
diag_df["monthly"]   = pd.to_datetime(diag_df["monthly"])
disp_df["months"]    = pd.to_datetime(disp_df["months"])
if "facility_id" in diag_df.columns:
    diag_df["facility_id"] = diag_df["facility_id"].astype(int)
max_date = disp["date"].max()

EXCLUDE = [7]
disp    = disp[~disp["facility_id"].isin(EXCLUDE)]
inv     = inv[~inv["facility_id"].isin(EXCLUDE)]
diag_df = diag_df[~diag_df["facility_id"].isin(EXCLUDE)] if "facility_id" in diag_df.columns else diag_df
disp_df = disp_df[~disp_df["facility_id"].isin(EXCLUDE)]

# backward compat
if "therapeutic_group" not in disp_df.columns:
    if "correct_therapeutic_class" in disp_df.columns:
        disp_df["therapeutic_group"] = disp_df["correct_therapeutic_class"].map(THERAPEUTIC_GROUP_MAP)
    else:
        disp_df["therapeutic_group"] = disp_df.get("new_category_name", pd.NA)

if "product_name" not in disp_df.columns:
    _pn = disp[["product_id","product_name"]].drop_duplicates()
    disp_df = disp_df.merge(_pn, on="product_id", how="left")

# ── OPENING STOCK DATA ────────────────────────────────────────────────────────
has_products = "pred_products" in data
prod_out = data["pred_products"].copy() if has_products else pd.DataFrame()

avg_price = (disp.groupby("product_id")["unit_selling_price"]
             .mean().reset_index().rename(columns={"unit_selling_price":"avg_price"}))

if has_products and not prod_out.empty:
    prod_out["Dead Stock Risk"] = prod_out["Dead Stock Risk"].map(
        {True:"Yes",False:"No","⚠️ Yes":"Yes","✓ No":"No"}).fillna(prod_out["Dead Stock Risk"])
    if "product_id" in prod_out.columns:
        prod_out = prod_out.merge(avg_price, on="product_id", how="left")
    elif "Product" in prod_out.columns:
        pn_pr = (disp[["product_name","unit_selling_price"]].dropna()
                 .groupby("product_name")["unit_selling_price"].median().reset_index()
                 .rename(columns={"product_name":"Product","unit_selling_price":"avg_price"}))
        prod_out = prod_out.merge(pn_pr, on="Product", how="left")
    if "avg_price" in prod_out.columns:
        cc = "Category" if "Category" in prod_out.columns else "therapeutic_group"
        prod_out["avg_price"] = (prod_out.groupby(cc)["avg_price"]
                                  .transform(lambda x: x.fillna(x.median())))
        prod_out["avg_price"] = prod_out["avg_price"].fillna(prod_out["avg_price"].median())
        qc = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
        prod_out["est_revenue"] = (prod_out[qc] * prod_out["avg_price"]).round(0)
    else:
        prod_out["avg_price"] = 0; prod_out["est_revenue"] = 0

cat_out = pred_output.copy()
cat_out["Dead Stock Risk"] = cat_out["Dead Stock Risk"].map(
    {True:"Yes",False:"No"}).fillna(cat_out["Dead Stock Risk"])

if has_products and not prod_out.empty and "est_revenue" in prod_out.columns:
    cc = "Category" if "Category" in prod_out.columns else "therapeutic_group"
    qc = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
    cat_rev = (prod_out.groupby(cc)
               .agg(total_units=(qc,"sum"),est_revenue=("est_revenue","sum"),
                    n_products=(cc,"count")).reset_index()
               .rename(columns={cc:"Category"}))
else:
    mp = disp["unit_selling_price"].median() if "unit_selling_price" in disp.columns else 1500
    cc2 = "Category" if "Category" in cat_out.columns else "therapeutic_group"
    qc2 = "Opening Stock Qty" if "Opening Stock Qty" in cat_out.columns else "opening_stock_qty"
    cat_rev = cat_out[[cc2,qc2]].copy()
    cat_rev.columns = ["Category","total_units"]
    cat_rev["est_revenue"] = cat_rev["total_units"] * mp
    cat_rev["n_products"] = 1

new_rev  = cat_rev[cat_rev["Category"].isin(NEW_CATS)]
core_rev = cat_rev[~cat_rev["Category"].isin(NEW_CATS)]
total_units   = int(cat_rev["total_units"].sum())
total_revenue = cat_rev["est_revenue"].sum()
new_units     = int(new_rev["total_units"].sum())
new_revenue   = new_rev["est_revenue"].sum()

def cat_u(cat): return int(cat_rev[cat_rev["Category"]==cat]["total_units"].sum()) if cat in cat_rev["Category"].values else 0
def cat_r(cat): return cat_rev[cat_rev["Category"]==cat]["est_revenue"].sum() if cat in cat_rev["Category"].values else 0

bty_u=cat_u("Beauty Products");        bty_r=cat_r("Beauty Products")
sup_u=cat_u("Vitamins & Supplements"); sup_r=cat_r("Vitamins & Supplements")
bb_u =cat_u("Body Building");          bb_r =cat_r("Body Building")

# ── PATIENT JOIN ──────────────────────────────────────────────────────────────
d_pat = disp.merge(pat[["patient_id","sex","age_group"]], on="patient_id", how="left")

# ── PRECOMPUTE PRODUCT MOVEMENT ───────────────────────────────────────────────
prod_monthly = (disp.assign(month=disp["date"].dt.to_period("M"))
                .groupby("product_name")
                .agg(months_sold=("month","nunique"),total_qty=("qty_dispensed","sum"),
                     total_revenue=("total_sales_value","sum"),
                     avg_unit_price=("unit_selling_price","mean"))
                .reset_index())
prod_monthly["movement"] = prod_monthly["months_sold"].apply(
    lambda m: "Fast" if m>=5 else ("Medium" if m>=3 else "Slow"))

# ── HEADER ────────────────────────────────────────────────────────────────────
hc, ht = st.columns([1,11])
with hc:
    if logo_img: st.image(logo_img, width=72)
with ht:
    st.markdown(f"""
    <div style="display:flex;align-items:center;height:64px;gap:.8rem">
      <div>
        <div style="font-size:.6rem;font-weight:800;letter-spacing:.16em;
                    text-transform:uppercase;color:{MUTED};margin-bottom:.1rem">
          PHARMAPLUS CHAIN ANALYTICS · AFYA ANALYTICS PLATFORM</div>
        <div style="font-size:1.45rem;font-weight:800;color:{COOL_BLUE};
                    font-family:'Montserrat',sans-serif;line-height:1.1">
          New Venture Headstart — Target Catchment Analysis</div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:2px solid #EBF3FB;margin:.4rem 0 1rem'>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🌍  Market Opportunity",
    "👥  Customer Spend Behaviour",
    "📦  Opening Stock — New Branch",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OPPORTUNITY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI row (retained from current dashboard) ─────────────────────────────
    moving_val  = prod_monthly[prod_monthly["movement"].isin(["Fast","Medium"])]["total_revenue"].sum()
    total_prods = disp["product_id"].nunique()
    moving_30   = prod_monthly[prod_monthly["months_sold"]>=5]["product_name"].nunique()
    turnover_pct = round(moving_30/total_prods*100,1) if total_prods > 0 else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi_card(c1,"Branches",str(disp["facility_id"].nunique()),AFYA_BLUE,"Active facilities")
    kpi_card(c2,"Products",f"{total_prods:,}",AFYA_BLUE,"In dispensing data")
    kpi_card(c3,"Patients",f"{disp['patient_id'].nunique():,}",AFYA_BLUE,"Unique")
    kpi_card(c4,"Moving Value",fmt_ksh(moving_val),TEAL,"Fast + medium movers")
    kpi_card(c5,"New Category Revenue",fmt_ksh(new_revenue),TEAL,"Projected Month 1")

    st.markdown("<div style='margin:.6rem 0'></div>",unsafe_allow_html=True)

    # ── SECTION A: DEMOGRAPHIC OPPORTUNITY ───────────────────────────────────
    sh("A — Demographic Opportunity: Who lives in this catchment")

    # Population signal cards (KNBS/DHS-derived, hardcoded from embu_external)
    dem_cols = st.columns(4)
    demo_signals = [
        ("👩 Women 15–49","176,000+","Core beauty & supplements buyers",TEAL),
        ("🧑 Men 15–34","~17% of pop","Body building & sports nutrition",ORANGE),
        ("🎓 Students","3 colleges within 1.5km","Multi-category demand driver",AFYA_BLUE),
        ("🏥 ANC Visit Rate","68%","Prenatal vitamins & iron demand signal",PURPLE),
    ]
    for col,(label,value,sub,color) in zip(dem_cols,demo_signals):
        kpi_card(col,label,value,color,sub)

    st.markdown("<div style='margin:.5rem 0'></div>",unsafe_allow_html=True)

    # Age + gender charts from real patient data
    da1, da2 = st.columns(2)

    with da1:
        age_order = ["Toddler (1-4)","Child (5-12)","Adolescent (13-17)","Youth (18-24)",
                     "Young Adult (25-34)","Adult (35-44)","Middle Age (45-54)",
                     "Older Adult (55-64)","Senior (65+)"]
        adf = d_pat.groupby("age_group")["patient_id"].nunique().reset_index()
        adf.columns = ["Age Group","Patients"]
        adf["Age Group"] = pd.Categorical(adf["Age Group"],categories=age_order,ordered=True)
        adf = adf.sort_values("Age Group")
        # Tag each band with likely category
        def age_cat(ag):
            ag = str(ag)
            if any(x in ag for x in ["Youth","Young Adult"]): return "Beauty + BB"
            if any(x in ag for x in ["Adult (35","Middle"]): return "Supplements"
            if "Toddler" in ag or "Child" in ag: return "Paediatric pharma"
            return "Pharma"
        adf["Category signal"] = adf["Age Group"].apply(age_cat)
        color_map = {"Beauty + BB":TEAL,"Supplements":AFYA_BLUE,
                     "Paediatric pharma":ORANGE,"Pharma":MUTED}
        fig_age = px.bar(adf,x="Patients",y="Age Group",orientation="h",
                         color="Category signal",color_discrete_map=color_map,
                         text="Patients")
        fig_age.update_traces(textposition="outside",
                               textfont=dict(color="#0072CE",size=9),marker_line_width=0)
        fig_age.update_layout(**CHART_LAYOUT,height=320,showlegend=True)
        fig_age.update_xaxes(**AXIS); fig_age.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Patient age bands → category demand signal</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_age,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with da2:
        gdf = d_pat.groupby("sex")["patient_id"].nunique().reset_index()
        gdf.columns = ["Gender","Patients"]
        fig_g = px.pie(gdf,names="Gender",values="Patients",hole=0.55,
                       color="Gender",color_discrete_map={"female":TEAL,"male":AFYA_BLUE,"F":TEAL,"M":AFYA_BLUE})
        fig_g.update_traces(textposition="inside",textinfo="percent+label",
                             textfont=dict(size=12,color="#fff"))
        fig_g.update_layout(showlegend=True,**CHART_LAYOUT,height=200)
        st.markdown('<div class="chart-card"><div class="card-title">Gender split — catchment population</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_g,use_container_width=True)

        # Proximity signals from scraped data
        st.markdown(f"""
        <div style="margin-top:8px;padding:12px 14px;background:#F4F8FC;
                    border-radius:8px;border:1px solid #D6E4F0">
          <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                      letter-spacing:.1em;color:{TEAL};margin-bottom:8px">
            Proximity signals — what's near this branch</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div style="font-size:.78rem;color:{COOL_BLUE}">
              <strong>🏋 Gym</strong><br>1 gym · 1.8km away<br>
              <span style="color:{ORANGE};font-weight:600">→ Body Building demand</span></div>
            <div style="font-size:.78rem;color:{COOL_BLUE}">
              <strong>💇 Beauty salons</strong><br>4+ within 1km<br>
              <span style="color:{TEAL};font-weight:600">→ Beauty referral traffic</span></div>
            <div style="font-size:.78rem;color:{COOL_BLUE}">
              <strong>🎓 Colleges</strong><br>3 within 1.5km<br>
              <span style="color:{AFYA_BLUE};font-weight:600">→ Student segment</span></div>
            <div style="font-size:.78rem;color:{COOL_BLUE}">
              <strong>🏥 Health facility</strong><br>Adjacent<br>
              <span style="color:{AFYA_BLUE};font-weight:600">→ Rx spillover + supplements</span></div>
          </div>
        </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # ── Jumia external signal ─────────────────────────────────────────────────
    ext_signal(
        "Jumia Kenya (2023)",
        "Beauty = #1 rural product category — 16% of all rural orders, ahead of phones & electronics",
        "Rural Kenyans are already buying beauty online because local pharmacies won't stock it. "
        "This branch is positioned to capture that demand in-store. "
        "Demand exists — local supply doesn't. This branch fills that gap."
    )

    # ── SECTION B: MARGIN OPPORTUNITY ────────────────────────────────────────
    sh("B — Margin Opportunity: Where the real profit sits")

    info("Sales alone is not a driver of profit. "
         "Beauty, Supplements, and Body Building carry significantly higher margins than core pharma — "
         "growing these categories is not just a volume play, it's a margin play.")

    mb1, mb2 = st.columns(2)

    with mb1:
        # Margin comparison by category from product_intel
        if not product_intel.empty and "gross_margin_pct" in product_intel.columns:
            margin_df = (product_intel.groupby("category")["gross_margin_pct"]
                         .mean().reset_index()
                         .rename(columns={"category":"Category","gross_margin_pct":"Avg Margin %"}))
            margin_df["Avg Margin %"] = (margin_df["Avg Margin %"]*100).round(1)
        else:
            # Estimated margins where product_intel not available
            margin_df = pd.DataFrame({
                "Category":["Pharma","Vitamins & Supplements","Beauty Products","Body Building"],
                "Avg Margin %":[15.0,38.0,31.0,43.0],
            })
        margin_df["is_new"] = margin_df["Category"].isin(NEW_CATS)
        margin_df = margin_df.sort_values("Avg Margin %",ascending=True)
        margin_df["color"] = margin_df["Category"].map(
            lambda c: NEW_COLOR.get(c, MUTED))
        fig_mg = go.Figure()
        fig_mg.add_trace(go.Bar(
            x=margin_df["Avg Margin %"],y=margin_df["Category"],
            orientation="h",
            marker_color=margin_df["color"].tolist(),
            marker_line_width=0,
            text=margin_df["Avg Margin %"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(color="#0072CE",size=10),
        ))
        fig_mg.update_layout(**CHART_LAYOUT,height=300,showlegend=False)
        fig_mg.update_xaxes(**AXIS,ticksuffix="%",title_text="Avg gross margin %")
        fig_mg.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Average gross margin by category — new categories highlighted</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_mg,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with mb2:
        # Monthly margin contribution from product_intel
        if not product_intel.empty and "monthly_margin_contribution_kes" in product_intel.columns:
            mc_df = (product_intel.groupby("category")["monthly_margin_contribution_kes"]
                     .sum().reset_index()
                     .rename(columns={"category":"Category","monthly_margin_contribution_kes":"Margin KES"}))
            mc_df = mc_df.sort_values("Margin KES",ascending=True)
            mc_df["label"] = mc_df["Margin KES"].apply(fmt_ksh)
            mc_df["color"] = mc_df["Category"].map(lambda c: NEW_COLOR.get(c, AFYA_BLUE))
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Bar(
                x=mc_df["Margin KES"],y=mc_df["Category"],
                orientation="h",
                marker_color=mc_df["color"].tolist(),
                marker_line_width=0,
                text=mc_df["label"],textposition="outside",
                textfont=dict(color="#0072CE",size=10),
            ))
            fig_mc.update_layout(**CHART_LAYOUT,height=300,showlegend=False)
            fig_mc.update_xaxes(**AXIS,title_text="Monthly margin contribution (KES)")
            fig_mc.update_yaxes(**AXIS)
            st.markdown('<div class="chart-card"><div class="card-title">Monthly margin contribution by category — projected</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_mc,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        else:
            # Revenue vs target mix comparison
            pharmaplus_target = pd.DataFrame({
                "Category":["Pharma","Beauty Products","Vitamins & Supplements","Body Building","Non-Pharma"],
                "Target %":[48,20,25,3,4],
            })
            fig_target = px.bar(pharmaplus_target,x="Category",y="Target %",
                                text="Target %",
                                color="Category",
                                color_discrete_map={
                                    "Pharma":MUTED,"Beauty Products":TEAL,
                                    "Vitamins & Supplements":AFYA_BLUE,
                                    "Body Building":ORANGE,"Non-Pharma":GRAY})
            fig_target.update_traces(texttemplate="%{text}%",textposition="outside",
                                      marker_line_width=0)
            fig_target.update_layout(**CHART_LAYOUT,height=300,showlegend=False)
            fig_target.update_xaxes(**AXIS); fig_target.update_yaxes(**AXIS,ticksuffix="%")
            st.markdown('<div class="chart-card"><div class="card-title">Pharmaplus target category mix — strategic goal</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_target,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

    # ── SECTION C: SIMILAR STORE PERFORMANCE (KNN) ───────────────────────────
    sh("C — Similar Store Performance: What matched branches already earn")

    info("The opening stock model finds branches whose disease burden, demographics, "
         "and dispensing profile most closely resemble this peri-urban catchment. "
         "Their category mix becomes the anchor for what to stock on Day 1.", TEAL)

    # KNN similar branch category mix
    sc1, sc2 = st.columns([3,2])
    with sc1:
        # Category revenue mix from real dispensing of existing branches
        branch_mix = (disp_df[disp_df["therapeutic_group"].notna()]
                      .groupby(["facility_id","therapeutic_group"])["total_qty_dispensed"]
                      .sum().reset_index())
        fac_total = branch_mix.groupby("facility_id")["total_qty_dispensed"].transform("sum")
        branch_mix["share"] = branch_mix["total_qty_dispensed"] / fac_total.replace(0,1)
        branch_mix["Branch"] = "Branch " + branch_mix["facility_id"].astype(str)
        # Top 6 groups for readability
        top_groups = (branch_mix.groupby("therapeutic_group")["total_qty_dispensed"]
                      .sum().nlargest(6).index.tolist())
        bm_top = branch_mix[branch_mix["therapeutic_group"].isin(top_groups)].copy()
        color_seq = [TEAL,AFYA_BLUE,COOL_BLUE,ORANGE,CORAL,PURPLE]
        fig_bm = px.bar(bm_top,x="Branch",y="share",color="therapeutic_group",
                        barmode="stack",text="share",
                        color_discrete_sequence=color_seq,
                        labels={"share":"Share of dispensing","therapeutic_group":"Category"})
        fig_bm.update_traces(texttemplate="%{text:.0%}",textposition="inside",
                              textfont=dict(color="#fff",size=9))
        fig_bm.update_layout(**CHART_LAYOUT,height=320)
        fig_bm.update_xaxes(**AXIS); fig_bm.update_yaxes(**AXIS,tickformat=".0%")
        st.markdown('<div class="chart-card"><div class="card-title">Category mix — existing branches (KNN similarity pool)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_bm,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with sc2:
        # Pharmaplus target vs what model predicts for new branch
        target_mix = pd.DataFrame({
            "Category":["Pharma","Beauty","Supplements","Body Building","Non-Pharma"],
            "Pharmaplus Target":[48,20,25,3,4],
        })
        fig_tm = px.bar(target_mix,x="Pharmaplus Target",y="Category",orientation="h",
                        text="Pharmaplus Target",
                        color="Category",
                        color_discrete_map={
                            "Pharma":COOL_BLUE,"Beauty":TEAL,
                            "Supplements":AFYA_BLUE,"Body Building":ORANGE,"Non-Pharma":GRAY})
        fig_tm.update_traces(texttemplate="%{text}%",textposition="outside",
                              marker_line_width=0,textfont=dict(color="#0072CE",size=10))
        fig_tm.update_layout(**CHART_LAYOUT,height=300,showlegend=False)
        fig_tm.update_xaxes(**AXIS,ticksuffix="%",title_text="Target share %")
        fig_tm.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Pharmaplus strategic target mix</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_tm,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # ── SECTION D: CATEGORY GROWTH TREND ─────────────────────────────────────
    sh("D — Category Growth Trend: Momentum from internal + external signals")

    # Simulated category trend from disp_df
    trend_cats = NEW_CATS
    trend_df = (disp_df[disp_df["therapeutic_group"].isin(trend_cats)]
                .groupby(["months","therapeutic_group"])["total_qty_dispensed"]
                .sum().reset_index())
    if not trend_df.empty:
        fig_tr = px.line(trend_df,x="months",y="total_qty_dispensed",
                         color="therapeutic_group",
                         color_discrete_map=NEW_COLOR,
                         labels={"months":"Month","total_qty_dispensed":"Units dispensed",
                                 "therapeutic_group":"Category"})
        fig_tr.update_traces(line_width=2.5)
        fig_tr.update_layout(**CHART_LAYOUT,height=300)
        fig_tr.update_xaxes(**AXIS,title_text="")
        fig_tr.update_yaxes(**AXIS,title_text="Monthly units")
        st.markdown('<div class="chart-card"><div class="card-title">New category demand trend — comparable branch dispensing data</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_tr,use_container_width=True)

        # GT momentum signal callout
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:6px">
          <div style="padding:10px 12px;background:rgba(11,185,159,.06);
                      border:1px solid rgba(11,185,159,.25);border-radius:8px">
            <div style="font-size:.6rem;font-weight:800;text-transform:uppercase;
                        color:{TEAL};letter-spacing:.08em;margin-bottom:4px">
              ✨ Beauty Products</div>
            <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">
              GT index 0.78 · rising slope</div>
            <div style="font-size:.72rem;color:{MUTED}">
              Nivea avg 0.40 · CeraVe momentum 0.49</div>
          </div>
          <div style="padding:10px 12px;background:rgba(0,114,206,.06);
                      border:1px solid rgba(0,114,206,.2);border-radius:8px">
            <div style="font-size:.6rem;font-weight:800;text-transform:uppercase;
                        color:{AFYA_BLUE};letter-spacing:.08em;margin-bottom:4px">
              💊 Vitamins & Supplements</div>
            <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">
              GT index 0.64 · stable</div>
            <div style="font-size:.72rem;color:{MUTED}">
              Immunity + multivitamin trending upward</div>
          </div>
          <div style="padding:10px 12px;background:rgba(245,166,35,.07);
                      border:1px solid rgba(245,166,35,.25);border-radius:8px">
            <div style="font-size:.6rem;font-weight:800;text-transform:uppercase;
                        color:{ORANGE};letter-spacing:.08em;margin-bottom:4px">
              💪 Body Building</div>
            <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">
              GT index 0.41 · seasonal Jan peak</div>
            <div style="font-size:.72rem;color:{MUTED}">
              Gym 1.8km away · payday demand pattern</div>
          </div>
        </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # ── SECTION E: RECURRING VS EPISODIC ──────────────────────────────────────
    sh("E — Revenue Pattern: Pharma is episodic · Supplements are recurring")

    re1, re2 = st.columns(2)
    with re1:
        if not product_intel.empty and "repurchase_cycle" in product_intel.columns:
            rc_df = (product_intel.groupby(["category","repurchase_cycle"])["product_id"]
                     .count().reset_index()
                     .rename(columns={"product_id":"Products","category":"Category",
                                      "repurchase_cycle":"Cycle"}))
            fig_rc = px.bar(rc_df,x="Category",y="Products",color="Cycle",
                            barmode="stack",
                            color_discrete_map={"monthly":TEAL,"quarterly":AFYA_BLUE,"biannual":ORANGE},
                            labels={"Products":"Product count"})
            fig_rc.update_traces(marker_line_width=0)
            fig_rc.update_layout(**CHART_LAYOUT,height=280)
            fig_rc.update_xaxes(**AXIS,tickangle=-20)
            fig_rc.update_yaxes(**AXIS)
        else:
            # Illustrative comparison
            rc_df = pd.DataFrame({
                "Category":["Pharma","Beauty Products","Vitamins & Supplements","Body Building"],
                "Monthly repurchase %":[15,30,65,70],
            })
            fig_rc = px.bar(rc_df,x="Category",y="Monthly repurchase %",
                            color="Category",
                            color_discrete_map={"Pharma":MUTED,"Beauty Products":TEAL,
                                                "Vitamins & Supplements":AFYA_BLUE,"Body Building":ORANGE},
                            text="Monthly repurchase %")
            fig_rc.update_traces(texttemplate="%{text}%",textposition="outside",marker_line_width=0)
            fig_rc.update_layout(**CHART_LAYOUT,height=280,showlegend=False)
            fig_rc.update_xaxes(**AXIS,tickangle=-20)
            fig_rc.update_yaxes(**AXIS,ticksuffix="%")
        st.markdown('<div class="chart-card"><div class="card-title">Repurchase cycle by category — monthly buyers = recurring revenue</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_rc,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with re2:
        st.markdown(f"""
        <div style="padding:20px;background:#F4F8FC;border-radius:10px;
                    border:1px solid #D6E4F0;height:100%">
          <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                      letter-spacing:.1em;color:{AFYA_BLUE};margin-bottom:16px">
            Why repurchase cycle matters for Day 1 stock</div>

          <div style="display:flex;gap:10px;margin-bottom:12px">
            <div style="width:3px;background:{CORAL};border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">
                Pharma — episodic</div>
              <div style="font-size:.75rem;color:{MUTED};line-height:1.55">
                A patient buys antibiotics when sick. Once recovered, they stop.
                Revenue is tied to illness frequency, not loyalty.</div>
            </div>
          </div>

          <div style="display:flex;gap:10px;margin-bottom:12px">
            <div style="width:3px;background:{TEAL};border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">
                Supplements — recurring</div>
              <div style="font-size:.75rem;color:{MUTED};line-height:1.55">
                A customer on a daily multivitamin returns every 30 days.
                One converted customer = 12 transactions per year.</div>
            </div>
          </div>

          <div style="display:flex;gap:10px">
            <div style="width:3px;background:{ORANGE};border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">
                Body Building — loyal</div>
              <div style="font-size:.75rem;color:{MUTED};line-height:1.55">
                Gym-goers who find their preferred protein brand at a pharmacy
                rarely switch. High loyalty once acquired.</div>
            </div>
          </div>
        </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SPEND BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── SECTION A: WHO IS THE CATCHMENT CUSTOMER ─────────────────────────────
    sh("A — Who is the catchment customer")

    ca1, ca2, ca3 = st.columns(3)

    with ca1:
        adf2 = d_pat.groupby("age_group")["patient_id"].nunique().reset_index()
        adf2.columns = ["Age Group","Patients"]
        age_order2 = ["Toddler (1-4)","Child (5-12)","Adolescent (13-17)","Youth (18-24)",
                      "Young Adult (25-34)","Adult (35-44)","Middle Age (45-54)",
                      "Older Adult (55-64)","Senior (65+)"]
        adf2["Age Group"] = pd.Categorical(adf2["Age Group"],categories=age_order2,ordered=True)
        adf2 = adf2.sort_values("Age Group")
        fig_a2 = px.bar(adf2,x="Patients",y="Age Group",orientation="h",
                        color="Patients",
                        color_continuous_scale=[[0,"#EBF3FB"],[1,TEAL]],
                        text="Patients")
        fig_a2.update_traces(textposition="outside",textfont=dict(color="#0072CE",size=9),
                              marker_line_width=0)
        fig_a2.update_coloraxes(showscale=False)
        fig_a2.update_layout(**CHART_LAYOUT,height=310,showlegend=False)
        fig_a2.update_xaxes(**AXIS); fig_a2.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Patients by age group</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_a2,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with ca2:
        gdf2 = d_pat.groupby("sex")["patient_id"].nunique().reset_index()
        gdf2.columns = ["Gender","Patients"]
        fig_g2 = px.pie(gdf2,names="Gender",values="Patients",hole=0.6,
                        color="Gender",
                        color_discrete_map={"female":TEAL,"male":AFYA_BLUE,"F":TEAL,"M":AFYA_BLUE})
        fig_g2.update_traces(textposition="inside",textinfo="percent+label",
                              textfont=dict(size=13,color="#fff"))
        fig_g2.update_layout(showlegend=False,**CHART_LAYOUT,height=310)
        st.markdown('<div class="chart-card"><div class="card-title">Gender split</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_g2,use_container_width=True)
        st.markdown(f'<div style="font-size:.72rem;color:{TEAL};font-weight:600;margin-top:4px">'
                    f'Female majority → strong beauty & supplements signal</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with ca3:
        # DHS / KNBS external demographic signals
        st.markdown(f"""
        <div style="padding:16px;background:#F4F8FC;border-radius:10px;
                    border:1px solid #D6E4F0;height:100%">
          <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                      letter-spacing:.1em;color:{AFYA_BLUE};margin-bottom:12px">
            External population signals (DHS 2022 / KNBS)</div>
          {"".join([f'''
          <div style="display:flex;justify-content:space-between;align-items:flex-start;
                      padding:7px 0;border-bottom:1px solid #EBF3FB">
            <div style="font-size:.78rem;color:{COOL_BLUE};font-weight:600">{label}</div>
            <div style="font-size:.82rem;font-weight:800;color:{color};min-width:52px;text-align:right">{val}</div>
          </div>'''
          for label,val,color in [
              ("ANC 4+ visits","68%",TEAL),
              ("Modern FP use","52%",AFYA_BLUE),
              ("Under-5 stunting","22%",CORAL),
              ("Urban branch flag","Yes",TEAL),
              ("Pop growth rate","3.1%/yr",AFYA_BLUE),
              ("Women 15–49","176K+",TEAL),
          ]])}
        </div>""",unsafe_allow_html=True)

    st.markdown("<div style='margin:.5rem 0'></div>",unsafe_allow_html=True)

    # ── SECTION B: HOW THEY SPEND ─────────────────────────────────────────────
    sh("B — How they spend: Sales ≠ profit")

    info("Fast-moving products drive volume. But slow movers and dead stock tie up capital. "
         "The dashboard surfaces both — so your opening order avoids the mistakes other branches made.",
         ORANGE)

    sb1, sb2 = st.columns(2)

    with sb1:
        # Fast vs slow vs dead from prod_monthly (real data)
        mv_counts = prod_monthly["movement"].value_counts().reset_index()
        mv_counts.columns = ["Movement","Products"]
        color_map_mv = {"Fast":TEAL,"Medium":AFYA_BLUE,"Slow":ORANGE}
        fig_mv = px.pie(mv_counts,names="Movement",values="Products",hole=0.55,
                        color="Movement",color_discrete_map=color_map_mv)
        fig_mv.update_traces(textposition="inside",textinfo="percent+label",
                              textfont=dict(size=12,color="#fff"))
        fig_mv.update_layout(showlegend=True,**CHART_LAYOUT,height=280)
        st.markdown('<div class="chart-card"><div class="card-title">Product movement — fast vs slow (existing branches)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_mv,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with sb2:
        # Revenue vs qty — sales ≠ profit illustration
        if not product_intel.empty and "monthly_velocity" in product_intel.columns and "monthly_margin_contribution_kes" in product_intel.columns:
            sp_df = (product_intel.groupby("category")
                     .agg(volume=("monthly_velocity","sum"),
                          margin=("monthly_margin_contribution_kes","sum"))
                     .reset_index().rename(columns={"category":"Category"}))
            sp_df["vol_share"] = sp_df["volume"]/sp_df["volume"].sum()*100
            sp_df["margin_share"] = sp_df["margin"]/sp_df["margin"].sum()*100
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Bar(name="Volume share %",x=sp_df["Category"],
                                     y=sp_df["vol_share"],marker_color=AFYA_BLUE,
                                     text=sp_df["vol_share"].apply(lambda v:f"{v:.1f}%"),
                                     textposition="outside",textfont=dict(size=9,color="#0072CE")))
            fig_sp.add_trace(go.Bar(name="Margin share %",x=sp_df["Category"],
                                     y=sp_df["margin_share"],marker_color=TEAL,
                                     text=sp_df["margin_share"].apply(lambda v:f"{v:.1f}%"),
                                     textposition="outside",textfont=dict(size=9,color="#0072CE")))
            fig_sp.update_layout(**CHART_LAYOUT,height=280,barmode="group")
            fig_sp.update_xaxes(**AXIS,tickangle=-20)
            fig_sp.update_yaxes(**AXIS,ticksuffix="%")
        else:
            # Illustrative data showing the gap
            sp_df = pd.DataFrame({
                "Category":["Pharma","Beauty Products","Vitamins & Supplements","Body Building"],
                "Volume share %":[72,11,13,4],
                "Margin share %":[45,20,28,7],
            })
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Bar(name="Volume share %",x=sp_df["Category"],
                                     y=sp_df["Volume share %"],marker_color=AFYA_BLUE,
                                     text=sp_df["Volume share %"].apply(lambda v:f"{v}%"),
                                     textposition="outside",textfont=dict(size=9,color="#0072CE")))
            fig_sp.add_trace(go.Bar(name="Margin share %",x=sp_df["Category"],
                                     y=sp_df["Margin share %"],marker_color=TEAL,
                                     text=sp_df["Margin share %"].apply(lambda v:f"{v}%"),
                                     textposition="outside",textfont=dict(size=9,color="#0072CE")))
            fig_sp.update_layout(**CHART_LAYOUT,height=280,barmode="group")
            fig_sp.update_xaxes(**AXIS,tickangle=-20)
            fig_sp.update_yaxes(**AXIS,ticksuffix="%")
        st.markdown('<div class="chart-card"><div class="card-title">Volume share vs margin share — pharma dominates volume, not margin</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_sp,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # Basket pairs panel
    if not product_intel.empty and "has_basket_partner" in product_intel.columns:
        bp_count = product_intel["has_basket_partner"].sum()
        sh("What gets bought together",teal=True)
        bp1, bp2 = st.columns([2,3])
        with bp1:
            kpi_card(bp1,"Cross-sell pairs",f"{int(bp_count):,}",TEAL,
                     "Products with a known basket partner")
        with bp2:
            basket_examples = [
                ("Vitamin C","Zinc","Immune stack"),
                ("Sunscreen","Moisturiser","Skincare routine"),
                ("Whey Protein","Creatine","BB training stack"),
                ("Face Wash","Toner","Cleanse-tone routine"),
                ("Vitamin D","Calcium","Bone health stack"),
            ]
            st.markdown(f"""
            <div style="padding:12px 14px;background:#F4F8FC;border-radius:8px;border:1px solid #D6E4F0">
              <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                          letter-spacing:.1em;color:{TEAL};margin-bottom:8px">
                Common basket pairs — stock these together</div>
              {"".join([f'''<div style="display:flex;gap:8px;align-items:center;
                                padding:5px 0;border-bottom:1px solid #EBF3FB;font-size:.78rem">
                <span style="font-weight:700;color:{COOL_BLUE}">{a}</span>
                <span style="color:{MUTED}">+</span>
                <span style="font-weight:700;color:{COOL_BLUE}">{b}</span>
                <span style="color:{TEAL};font-size:.7rem;margin-left:auto">{note}</span>
              </div>''' for a,b,note in basket_examples])}
            </div>""",unsafe_allow_html=True)

    # ── SECTION C: GENERIC VS BRANDED ────────────────────────────────────────
    sh("C — Generic vs Branded: Getting the mix right from Day 1")

    info("This branch opens in a peri-urban catchment where income levels favour generics. "
         "A previous branch stocked too many branded originals for its demographics — "
         "this intelligence layer prevents that from happening again.",CORAL)

    # Build generic analysis from real dispensing data
    pharma_disp = disp[~disp["category"].isin(NEW_CATS)] if "category" in disp.columns else disp.copy()
    pharma_disp = pharma_disp.copy()
    pharma_disp["is_generic"] = pharma_disp["product_name"].apply(infer_generic)
    pharma_disp_classified = pharma_disp[pharma_disp["is_generic"].notna()].copy()

    gc1, gc2 = st.columns(2)

    with gc1:
        if not pharma_disp_classified.empty:
            gen_vol = (pharma_disp_classified.groupby("is_generic")
                       .agg(qty=("qty_dispensed","sum"),
                            revenue=("total_sales_value","sum"))
                       .reset_index())
            gen_vol["Type"] = gen_vol["is_generic"].map({True:"Generic",False:"Branded"})
            gen_vol["Revenue label"] = gen_vol["revenue"].apply(fmt_ksh)

            fig_gv = px.bar(gen_vol,x="Type",y="qty",text="qty",
                            color="Type",
                            color_discrete_map={"Generic":TEAL,"Branded":CORAL},
                            labels={"qty":"Units dispensed"})
            fig_gv.update_traces(texttemplate="%{text:,}",textposition="outside",
                                   marker_line_width=0,textfont=dict(color="#0072CE",size=10))
            fig_gv.update_layout(**CHART_LAYOUT,height=260,showlegend=False)
            fig_gv.update_xaxes(**AXIS); fig_gv.update_yaxes(**AXIS)
            st.markdown('<div class="chart-card"><div class="card-title">Generic vs branded — actual dispensing volume</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_gv,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        else:
            st.info("Generic/branded classification requires product name data in dispensing records.")

    with gc2:
        if not pharma_disp_classified.empty:
            # Revenue split
            gen_rev = (pharma_disp_classified.groupby("is_generic")["total_sales_value"]
                       .sum().reset_index())
            gen_rev["Type"] = gen_rev["is_generic"].map({True:"Generic",False:"Branded"})
            fig_gr = px.pie(gen_rev,names="Type",values="total_sales_value",hole=0.55,
                            color="Type",
                            color_discrete_map={"Generic":TEAL,"Branded":CORAL})
            fig_gr.update_traces(textposition="inside",textinfo="percent+label",
                                  textfont=dict(size=13,color="#fff"))
            fig_gr.update_layout(showlegend=False,**CHART_LAYOUT,height=260)
            st.markdown('<div class="chart-card"><div class="card-title">Generic vs branded — revenue split</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_gr,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

    # New category generic/branded from product_intel
    if not product_intel.empty and "is_generic" in product_intel.columns:
        sh("New category generic/branded intelligence",teal=True)
        nc_intel = product_intel.copy()
        nc_gb = (nc_intel.groupby(["category","is_generic"])
                 .agg(products=("product_id","count"),
                      avg_margin=("gross_margin_pct","mean"),
                      total_saving=("generic_saving_per_month_kes","sum"))
                 .reset_index())
        nc_gb["Type"] = nc_gb["is_generic"].map({True:"Generic",False:"Branded"})
        nc_gb["Avg margin %"] = (nc_gb["avg_margin"]*100).round(1)
        nc_gb["Monthly saving"] = nc_gb["total_saving"].apply(fmt_ksh)

        ni1, ni2 = st.columns(2)
        with ni1:
            fig_ni = px.bar(nc_gb,x="category",y="products",color="Type",barmode="group",
                            color_discrete_map={"Generic":TEAL,"Branded":CORAL},
                            labels={"category":"Category","products":"Product count"})
            fig_ni.update_traces(marker_line_width=0)
            fig_ni.update_layout(**CHART_LAYOUT,height=260)
            fig_ni.update_xaxes(**AXIS,tickangle=-20); fig_ni.update_yaxes(**AXIS)
            st.markdown('<div class="chart-card"><div class="card-title">Generic vs branded products — new categories</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_ni,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        with ni2:
            # Margin advantage of generics
            fig_ma = px.bar(nc_gb,x="category",y="Avg margin %",color="Type",barmode="group",
                            color_discrete_map={"Generic":TEAL,"Branded":CORAL},
                            text="Avg margin %",
                            labels={"category":"Category"})
            fig_ma.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                                  marker_line_width=0,textfont=dict(size=9,color="#0072CE"))
            fig_ma.update_layout(**CHART_LAYOUT,height=260)
            fig_ma.update_xaxes(**AXIS,tickangle=-20)
            fig_ma.update_yaxes(**AXIS,ticksuffix="%")
            st.markdown('<div class="chart-card"><div class="card-title">Margin advantage — generics earn more per unit sold</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_ma,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        total_saving = nc_intel["generic_saving_per_month_kes"].sum()
        if total_saving > 0:
            info(f"Stocking the recommended generic mix in new categories could save customers "
                 f"{fmt_ksh(total_saving)}/month vs the branded equivalent — "
                 f"while earning Pharmaplus a higher margin per unit.",TEAL)

    # ── SECTION D: DISEASE BURDEN (moved to last) ─────────────────────────────
    sh("D — Disease Burden: What conditions drive pharmacy visits")

    burden_exp = (diag_df.assign(group=diag_df["diagnosis_burden_group"].str.split("|"))
                  .explode("group"))
    burden_exp["group"] = burden_exp["group"].str.strip()

    def bundle_burden(g):
        g = str(g).lower()
        if "ncd" in g and any(x in g for x in ["cardiovasc","endocrin","respirat","mental"]): return "NCD"
        if "ncd" in g: return "NCD"
        if "communicable" in g or "malaria" in g: return "Communicable"
        if "mnch" in g or "maternal" in g: return "Maternal & Child Health"
        if "gu-gyn" in g or "reproduct" in g: return "Reproductive Health"
        if "respiratory" in g: return "Respiratory"
        if "gi" in g or "gastro" in g: return "GI"
        if "msk" in g or "musculo" in g: return "MSK / Injury"
        if "dermat" in g: return "Dermatological"
        return "Other"

    burden_exp["bundled"] = burden_exp["group"].apply(bundle_burden)

    CAT_COLORS = {
        "NCD":AFYA_BLUE,"Communicable":TEAL,"Respiratory":"#0BB99F",
        "Maternal & Child Health":ORANGE,"Reproductive Health":PURPLE,
        "GI":CORAL,"MSK / Injury":COOL_BLUE,"Dermatological":TEAL,"Other":GRAY,
    }

    bundled_sum = (burden_exp.groupby("bundled")["consultation_count"]
                   .sum().reset_index()
                   .rename(columns={"bundled":"Category","consultation_count":"Consultations"}))
    bundled_sum["% Share"] = (bundled_sum["Consultations"]/bundled_sum["Consultations"].sum()*100).round(1)
    bundled_sum = bundled_sum.sort_values("Consultations",ascending=False)

    db1, db2 = st.columns(2)

    with db1:
        fig_bd = px.bar(bundled_sum.sort_values("Consultations",ascending=True),
                        x="Consultations",y="Category",orientation="h",
                        color="Category",color_discrete_map=CAT_COLORS,text="% Share")
        fig_bd.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                              textfont=dict(color="#0072CE",size=10),marker_line_width=0)
        fig_bd.update_layout(**CHART_LAYOUT,height=360,showlegend=False)
        fig_bd.update_xaxes(**AXIS); fig_bd.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Disease burden groups — consultations<br>'
                    '<span style="color:#0BB99F;font-size:.68rem;font-weight:600">'
                    'Dermatological → Beauty · MSK → BB · NCD preventive → Supplements</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_bd,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with db2:
        # NCD sub-categories
        def ncd_sub(g):
            g = str(g).lower()
            if "ncd" not in g: return None
            if "cardiovasc" in g and "hypert" in g: return "Cardiovascular - Hypertension"
            if "cardiovasc" in g: return "Cardiovascular - Other"
            if "endocrin" in g and "diab" in g: return "Diabetes / Endocrine"
            if "mental" in g: return "Mental Health"
            if "respirat" in g: return "Respiratory / Asthma"
            return "NCD - Other"
        burden_exp["ncd_sub"] = burden_exp["group"].apply(ncd_sub)
        ncd_sub_sum = (burden_exp[burden_exp["ncd_sub"].notna()]
                       .groupby("ncd_sub")["consultation_count"]
                       .sum().reset_index()
                       .rename(columns={"ncd_sub":"Sub-category","consultation_count":"Consultations"}))
        ncd_sub_sum = ncd_sub_sum.sort_values("Consultations",ascending=True)
        fig_ns = px.bar(ncd_sub_sum,x="Consultations",y="Sub-category",orientation="h",
                        color="Consultations",
                        color_continuous_scale=[[0,"#EBF3FB"],[1,AFYA_BLUE]],
                        text="Consultations")
        fig_ns.update_traces(textposition="outside",textfont=dict(color="#0072CE",size=10),
                              marker_line_width=0)
        fig_ns.update_coloraxes(showscale=False)
        fig_ns.update_layout(**CHART_LAYOUT,height=360)
        fig_ns.update_xaxes(**AXIS); fig_ns.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">NCD sub-categories — product demand drivers</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_ns,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # Antibiotic stewardship
    sh("Antibiotic Stewardship",teal=True)
    FIRST  = ["amoxicillin","ampicillin","penicillin","cotrimoxazole","erythromycin","nitrofurantoin"]
    SECOND = ["azithromycin","clarithromycin","ciprofloxacin","levofloxacin","co-amoxiclav",
               "augmentin","ceftriaxone","cefuroxime","doxycycline","gentamicin","clindamycin"]
    THIRD  = ["meropenem","vancomycin","imipenem","colistin","linezolid","tigecycline"]

    def abx_tier(n):
        n = str(n).lower()
        if any(k in n for k in THIRD):  return "Third-line"
        if any(k in n for k in SECOND): return "Second-line"
        if any(k in n for k in FIRST):  return "First-line"
        return None

    abx = disp.copy(); abx["abx_tier"] = abx["product_name"].apply(abx_tier)
    abx = abx[abx["abx_tier"].notna()]
    if not abx.empty:
        tier_colors = {"First-line":TEAL,"Second-line":ORANGE,"Third-line":CORAL}
        ab1, ab2 = st.columns([2,1])
        with ab1:
            abx_b = abx.groupby(["facility_id","abx_tier"])["qty_dispensed"].sum().reset_index()
            abx_b["Branch"] = "Branch " + abx_b["facility_id"].astype(str)
            fig_abx = px.bar(abx_b,x="Branch",y="qty_dispensed",color="abx_tier",
                              barmode="stack",text="qty_dispensed",
                              color_discrete_map=tier_colors,
                              labels={"qty_dispensed":"Units","abx_tier":"Tier"})
            fig_abx.update_traces(textposition="inside",textfont=dict(color="#fff",size=9))
            fig_abx.update_layout(**CHART_LAYOUT,height=280)
            fig_abx.update_xaxes(**AXIS); fig_abx.update_yaxes(**AXIS)
            st.markdown('<div class="chart-card"><div class="card-title">1st vs 2nd vs 3rd line antibiotics per branch</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_abx,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        with ab2:
            abx_t = abx.groupby("abx_tier")["qty_dispensed"].sum().reset_index()
            total_abx = abx_t["qty_dispensed"].sum()
            st.markdown('<div class="chart-card"><div class="card-title">Chain-wide split</div>',
                        unsafe_allow_html=True)
            for _,row in abx_t.iterrows():
                c = tier_colors.get(str(row["abx_tier"]),GRAY)
                pct = round(row["qty_dispensed"]/total_abx*100) if total_abx>0 else 0
                st.markdown(f"""
                <div style="background:#f0f2f6;border-radius:8px;padding:.55rem .8rem;
                            margin-bottom:.4rem;border-left:3px solid {c}">
                  <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                              color:{MUTED}">{row['abx_tier']}</div>
                  <div style="color:{c};font-size:1.15rem;font-weight:700">
                    {int(row['qty_dispensed']):,} units</div>
                  <div style="font-size:.7rem;color:{MUTED}">{pct}% of antibiotics</div>
                </div>""",unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPENING STOCK RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    REORDER_MAP = {
        "Oral Solid Forms":       ("Day 18",TEAL),
        "Injectables":            ("Day 22",TEAL),
        "Beauty Products":        ("Day 38",TEAL),
        "Vitamins & Supplements": ("Day 45",AFYA_BLUE),
        "Oral Liquid Forms":      ("Day 55",AFYA_BLUE),
        "Body Building":          ("Day 72",ORANGE),
        "Wound Care":             ("Hold",CORAL),
    }
    NEW_WHY = {
        "Beauty Products":        "Dense beauty salon cluster within 1km drives referral traffic. "
                                  "GT beauty index 0.78 and rising. "
                                  "176K women aged 15–49 in peri-urban catchment. "
                                  "Jumia data confirms rural beauty demand is #1 category. "
                                  "Stock facial care, body lotions, and lip care first.",
        "Vitamins & Supplements": "45% of adults aged 25–64 in catchment are the core supplements buyer. "
                                  "68% ANC visit rate drives prenatal vitamins and iron demand. "
                                  "Immunity and multivitamins trending upward on GT (index 0.64). "
                                  "Adjacent health facility creates prescription spillover.",
        "Body Building":          "17% of catchment population are men aged 15–34. "
                                  "Gym confirmed 1.8km away. "
                                  "3 colleges within 1.5km — student fitness segment. "
                                  "Whey protein and creatine lead. Start conservatively, reorder on Day 72.",
    }
    CORE_WHY = {
        "Oral Solid Forms":      "Antibiotics, antidiabetics, antimalarials. High NCD + malaria burden. Fastest-selling.",
        "Injectables":           "Demand driven by adjacent health facility and 68% ANC visit rate. Artemether and oxytocin move fastest.",
        "Oral Liquid Forms":     "30%+ of catchment is under 15. Paediatric syrups and ORS are steady sellers.",
        "IV Fluids & Infusions": "Hospital-proximity product. Lower walk-in demand — order conservatively.",
        "Topical Preparations":  "Skin conditions are common. Moderate volume.",
        "Vaccines & Biologicals":"Immunisation-driven. Coordinate with facility schedule.",
    }

    # ── Hero strip ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{COOL_BLUE};border-radius:12px;padding:20px 24px;
                margin-bottom:16px;display:flex;align-items:stretch;gap:0">
      <div style="flex:1.4;padding-right:22px;border-right:1px solid rgba(255,255,255,.12)">
        <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
                    color:rgba(255,255,255,.45);margin-bottom:5px">
          New growth categories — Peri-urban new branch</div>
        <div style="font-size:1.5rem;font-weight:800;color:#fff;line-height:1.15;margin-bottom:5px">
          3 new categories &middot; {new_units:,} units</div>
        <div style="font-size:.88rem;color:rgba(255,255,255,.6)">
          Estimated Month 1 revenue from new categories:
          <span style="color:{TEAL};font-weight:800;font-size:1.05rem"> {fmt_ksh(new_revenue)}</span>
        </div>
        <div style="margin-top:10px;font-size:.75rem;color:rgba(255,255,255,.4);line-height:1.55">
          Sized at 35% of mature branch velocity &mdash; Month 1 penetration factor.<br>
          Beauty and Supplements adjusted upward by Google Trends momentum signal.
        </div>
      </div>
      <div style="flex:1;padding:0 18px;border-right:1px solid rgba(255,255,255,.12)">
        <div style="display:inline-block;background:{TEAL};color:#fff;font-size:.62rem;
                    font-weight:800;padding:2px 9px;border-radius:20px;margin-bottom:6px">BEAUTY</div>
        <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1">{bty_u:,}</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:2px">units to order</div>
        <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px">{fmt_ksh(bty_r)}</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.4)">est. Month 1</div>
        <div style="font-size:.7rem;color:rgba(255,255,255,.45);margin-top:5px">
          Salons + Jumia rural signal</div>
      </div>
      <div style="flex:1;padding:0 18px;border-right:1px solid rgba(255,255,255,.12)">
        <div style="display:inline-block;background:{TEAL};color:#fff;font-size:.62rem;
                    font-weight:800;padding:2px 9px;border-radius:20px;margin-bottom:6px">SUPPLEMENTS</div>
        <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1">{sup_u:,}</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:2px">units to order</div>
        <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px">{fmt_ksh(sup_r)}</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.4)">est. Month 1</div>
        <div style="font-size:.7rem;color:rgba(255,255,255,.45);margin-top:5px">
          45% adults = core buyers</div>
      </div>
      <div style="flex:1;padding:0 0 0 18px">
        <div style="display:inline-block;background:rgba(255,255,255,.12);color:#fff;
                    font-size:.62rem;font-weight:800;padding:2px 9px;border-radius:20px;margin-bottom:6px">
          BODY BUILDING</div>
        <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1">{bb_u:,}</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:2px">units to order</div>
        <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px">{fmt_ksh(bb_r)}</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.4)">est. Month 1</div>
        <div style="font-size:.7rem;color:rgba(255,255,255,.45);margin-top:5px">
          Gym 1.8km · student segment</div>
      </div>
    </div>""",unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    dead_count = int((prod_out["Dead Stock Risk"]=="Yes").sum()) if has_products and not prod_out.empty and "Dead Stock Risk" in prod_out.columns else 0
    high_conf  = int((prod_out["Confidence"]=="High").sum())    if has_products and not prod_out.empty and "Confidence" in prod_out.columns else 0
    total_pred = len(prod_out) if has_products and not prod_out.empty else len(cat_out)

    k1,k2,k3,k4,k5 = st.columns(5)
    kpi_card(k1,"Total products",f"{total_pred:,}",AFYA_BLUE,"Recommended for Day 1")
    kpi_card(k2,"Opening stock units",f"{total_units:,}",AFYA_BLUE,"Total order")
    kpi_card(k3,"Est. Month 1 revenue",fmt_ksh(total_revenue),TEAL,"All categories")
    kpi_card(k4,"High confidence",f"{high_conf:,}",TEAL,f"of {total_pred} products")
    kpi_card(k5,"Dead stock risk",f"{dead_count:,}",CORAL,"Remove from order")

    st.markdown("<div style='margin:.75rem 0'></div>",unsafe_allow_html=True)

    # ── Category cards helper ─────────────────────────────────────────────────
    def _conf_color(c): return TEAL if c=="High" else (ORANGE if c=="Medium" else CORAL)

    def render_cat_card(row, is_new):
        cat   = row["Category"]
        units = int(row["total_units"])
        rev   = row["est_revenue"]
        n_p   = int(row.get("n_products",0))
        why   = NEW_WHY.get(cat, CORE_WHY.get(cat,"Based on catchment profile and KNN-matched branch data."))
        day, rcolor = REORDER_MAP.get(cat,("—",MUTED))
        tc  = TEAL if is_new else AFYA_BLUE
        tl  = "New category — growth opportunity" if is_new else "Core pharmacy"
        tlc = TEAL if is_new else MUTED
        if has_products and not prod_out.empty and "Confidence" in prod_out.columns:
            cc_col = "Category" if "Category" in prod_out.columns else "therapeutic_group"
            cp = prod_out[prod_out[cc_col]==cat]
            cc2 = cp["Confidence"].value_counts()
            top_conf = cc2.index[0] if not cc2.empty else "—"
            cpct = int(cc2.iloc[0]/len(cp)*100) if len(cp)>0 else 0
        else:
            top_conf, cpct = "—", 0
        cfc = _conf_color(top_conf)
        return f"""
        <div style="background:#fff;border:1px solid {BORDER};border-radius:10px;
                    overflow:hidden;border-top:4px solid {tc};margin-bottom:4px">
          <div style="padding:11px 14px 8px;border-bottom:1px solid {BORDER}">
            <div style="font-size:.85rem;font-weight:700;color:{COOL_BLUE}">{cat}</div>
            <div style="font-size:.65rem;font-weight:700;color:{tlc};margin-top:2px">{tl}</div>
          </div>
          <div style="padding:12px 14px">
            <div style="font-size:1.75rem;font-weight:800;color:{COOL_BLUE};line-height:1">{units:,}</div>
            <div style="font-size:.68rem;color:{MUTED};margin-bottom:4px">units &middot; {n_p} products</div>
            <div style="font-size:.88rem;font-weight:700;color:{TEAL};margin-bottom:8px">{fmt_ksh(rev)} est. Month 1</div>
            <div style="font-size:.73rem;color:rgba(0,52,103,.55);line-height:1.55;margin-bottom:8px">{why}</div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:7px">
              <div style="font-size:.62rem;color:{MUTED};width:58px;flex-shrink:0">Confidence</div>
              <div style="flex:1;height:5px;background:#F0F5FF;border-radius:3px;overflow:hidden">
                <div style="width:{cpct}%;height:100%;background:{cfc};border-radius:3px"></div>
              </div>
              <div style="font-size:.65rem;font-weight:700;color:{cfc};min-width:36px;text-align:right">{top_conf}</div>
            </div>
            <div style="background:{rcolor}18;border-radius:5px;padding:5px 9px;font-size:.72rem;
                        font-weight:700;color:{rcolor}">Reorder: {day}</div>
          </div>
        </div>"""

    # ── New category cards ────────────────────────────────────────────────────
    sh("New growth categories — order these first", teal=True)
    nc1,nc2,nc3 = st.columns(3)
    for col,cat_name in zip([nc1,nc2,nc3],NEW_CATS):
        row = cat_rev[cat_rev["Category"]==cat_name]
        if row.empty:
            row = pd.DataFrame([{"Category":cat_name,"total_units":0,"est_revenue":0,"n_products":0}])
        with col:
            st.markdown(render_cat_card(row.iloc[0],is_new=True),unsafe_allow_html=True)

    st.markdown("<div style='margin:.75rem 0'></div>",unsafe_allow_html=True)

    # ── Core pharma cards ─────────────────────────────────────────────────────
    sh("Core pharmacy — order as usual")
    core_sorted = core_rev.sort_values("total_units",ascending=False)
    top_core = core_sorted.head(3); rest_core = core_sorted.iloc[3:]
    cc1,cc2,cc3 = st.columns(3)
    for col,(_,row) in zip([cc1,cc2,cc3],top_core.iterrows()):
        with col:
            st.markdown(render_cat_card(row,is_new=False),unsafe_allow_html=True)
    if not rest_core.empty:
        show_more = st.toggle(f"Show {len(rest_core)} more core categories", value=False, key="toggle_core")
        if show_more:
            rcols = st.columns(3)
            for i,(_,row) in enumerate(rest_core.iterrows()):
                with rcols[i%3]:
                    st.markdown(render_cat_card(row,is_new=False),unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0'></div>",unsafe_allow_html=True)

    # ── Scenario modelling ────────────────────────────────────────────────────
    show_scenario = st.toggle("Scenario modelling — what if you run a campaign?", value=False, key="toggle_scenario")
    if show_scenario:
        st.markdown(f"""
        <div style="padding:10px 14px;background:#F4F8FC;border-left:3px solid {TEAL};
                    border-radius:4px;font-size:12px;color:#0072CE;margin-bottom:14px">
          Adjust the sliders to model the impact of partnerships and campaigns on opening stock.
          These multipliers apply to the base KNN prediction.
        </div>""",unsafe_allow_html=True)
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            gym_uplift   = st.slider("🏋 Gym partnership uplift (Body Building)",0,50,0,5,
                                      help="Pharmaplus partners with nearby gym. Expected BB demand uplift %.")
            salon_uplift = st.slider("💇 Salon partnership uplift (Beauty)",0,50,0,5,
                                      help="Referral arrangement with nearby salons. Expected Beauty demand uplift %.")
            wellness_uplift = st.slider("💊 Wellness/NCD drive (Supplements)",0,40,0,5,
                                         help="NCD screening + supplement recommendation campaign uplift %.")
        with sc_col2:
            student_uplift  = st.slider("🎓 Student drive (all categories)",0,30,0,5,
                                         help="Campus activation campaign — applies across all new categories.")
            social_uplift   = st.slider("📱 Social media campaign (Beauty + Supplements)",0,30,0,5,
                                         help="Instagram/TikTok campaign targeting peri-urban catchment.")

        # Compute scenario adjustments
        bb_adj   = bb_u  * (1 + (gym_uplift + student_uplift) / 100)
        bty_adj  = bty_u * (1 + (salon_uplift + social_uplift + student_uplift) / 100)
        sup_adj  = sup_u * (1 + (wellness_uplift + social_uplift + student_uplift) / 100)

        bb_rev_adj  = bb_r  * (1 + (gym_uplift + student_uplift) / 100)
        bty_rev_adj = bty_r * (1 + (salon_uplift + social_uplift + student_uplift) / 100)
        sup_rev_adj = sup_r * (1 + (wellness_uplift + social_uplift + student_uplift) / 100)

        total_new_rev_adj = bb_rev_adj + bty_rev_adj + sup_rev_adj
        rev_uplift_kes    = total_new_rev_adj - new_revenue

        sm1,sm2,sm3,sm4 = st.columns(4)
        kpi_card(sm1,"✨ Beauty (adjusted)",f"{int(bty_adj):,}",TEAL,
                 f"{fmt_ksh(bty_rev_adj)} est. revenue")
        kpi_card(sm2,"💊 Supplements (adjusted)",f"{int(sup_adj):,}",AFYA_BLUE,
                 f"{fmt_ksh(sup_rev_adj)} est. revenue")
        kpi_card(sm3,"💪 Body Building (adjusted)",f"{int(bb_adj):,}",ORANGE,
                 f"{fmt_ksh(bb_rev_adj)} est. revenue")
        kpi_card(sm4,"Campaign revenue uplift",fmt_ksh(rev_uplift_kes),
                 TEAL if rev_uplift_kes>0 else MUTED,
                 "vs base prediction")

    # ── Revenue chart + reorder timeline ─────────────────────────────────────
    rc, rt = st.columns([3,2])
    with rc:
        chart_d = cat_rev.sort_values("est_revenue",ascending=False).head(10).sort_values("est_revenue",ascending=True).copy()
        chart_d["is_new"] = chart_d["Category"].isin(NEW_CATS)
        chart_d["label"]  = chart_d["est_revenue"].apply(fmt_ksh)
        fig_rc = go.Figure()
        for label,is_new_flag,color in [("New category",True,TEAL),("Core pharma",False,AFYA_BLUE)]:
            sub = chart_d[chart_d["is_new"]==is_new_flag]
            fig_rc.add_trace(go.Bar(
                x=sub["est_revenue"],y=sub["Category"],orientation="h",
                marker_color=color,marker_line_width=0,
                text=sub["label"],textposition="outside",
                textfont=dict(color=COOL_BLUE,size=10),name=label,
            ))
        fig_rc.update_layout(height=max(300,len(chart_d)*36),**CHART_LAYOUT,barmode="stack")
        fig_rc.update_xaxes(**AXIS,title_text="Est. Revenue (KES)")
        fig_rc.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Estimated Month 1 revenue by category</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_rc,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with rt:
        reorder_items = [
            ("Oral Solid Forms",       "Day 18",TEAL,  "Fastest — set reminder now"),
            ("Injectables",            "Day 22",TEAL,  "Hospital demand"),
            ("Beauty Products",        "Day 38",TEAL,  "NEW — GT momentum rising"),
            ("Vitamins & Supplements", "Day 45",AFYA_BLUE,"NEW — steady adult demand"),
            ("Oral Liquid Forms",      "Day 55",AFYA_BLUE,"Paediatric"),
            ("Body Building",          "Day 72",ORANGE,"NEW — reorder only what sold"),
            ("Wound Care / Dental",    "Hold",  CORAL, "Check sell-through first"),
        ]
        st.markdown('<div class="chart-card"><div class="card-title">When to reorder</div>',
                    unsafe_allow_html=True)
        for name,day,color,note in reorder_items:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;
                        padding:8px 0;border-bottom:1px solid {BORDER}">
              <div style="width:8px;height:8px;border-radius:50%;background:{color};
                          flex-shrink:0;margin-top:4px"></div>
              <div style="flex:1">
                <div style="font-size:.78rem;font-weight:700;color:{COOL_BLUE}">{name}</div>
                <div style="font-size:.66rem;color:{MUTED};margin-top:1px">{note}</div>
              </div>
              <div style="font-size:.78rem;font-weight:800;color:{color};
                          white-space:nowrap;padding-left:8px">{day}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # ── Full product table ────────────────────────────────────────────────────
    sh("Full Stock List — All Categories")

    if has_products and not prod_out.empty:
        cc_name = "Category" if "Category" in prod_out.columns else "therapeutic_group"
        qc_name = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
        pn_col  = "Product" if "Product" in prod_out.columns else "product_name"

        f1,f2,f3,_ = st.columns([2,2,2,4])
        with f1:
            cats_all = ["All categories"] + sorted(prod_out[cc_name].dropna().unique().tolist())
            sel_c = st.selectbox("Category",cats_all,key="sc3")
        with f2:
            sel_cf = st.selectbox("Confidence",["All","High","Medium","Low"],key="scf3")
        with f3:
            sel_r = st.selectbox("Dead stock",["All","Clear only","Risk only"],key="sr3")

        disp_cols = [pn_col,cc_name,qc_name]
        if "est_revenue" in prod_out.columns:       disp_cols.append("est_revenue")
        if "Historical Share" in prod_out.columns:  disp_cols.append("Historical Share")
        if "Confidence" in prod_out.columns:        disp_cols.append("Confidence")
        if "Dead Stock Risk" in prod_out.columns:   disp_cols.append("Dead Stock Risk")

        tbl = prod_out[disp_cols].rename(columns={
            pn_col:"Product",cc_name:"Category",qc_name:"Order Qty",
            "est_revenue":"Est. Revenue (KES)",
        }).copy()

        if sel_c != "All categories": tbl = tbl[tbl["Category"]==sel_c]
        if sel_cf != "All" and "Confidence" in tbl.columns:
            tbl = tbl[tbl["Confidence"]==sel_cf]
        if sel_r == "Risk only" and "Dead Stock Risk" in tbl.columns:
            tbl = tbl[tbl["Dead Stock Risk"]=="Yes"]
        elif sel_r == "Clear only" and "Dead Stock Risk" in tbl.columns:
            tbl = tbl[tbl["Dead Stock Risk"]=="No"]

        tbl = tbl.sort_values("Order Qty",ascending=False).reset_index(drop=True)
        col_cfg = {
            "Order Qty":          st.column_config.NumberColumn(format="%d"),
            "Est. Revenue (KES)": st.column_config.NumberColumn(format="KES %,.0f"),
            "Product":            st.column_config.TextColumn(width="large"),
            "Category":           st.column_config.TextColumn(width="medium"),
            "Confidence":         st.column_config.TextColumn(width="small"),
            "Dead Stock Risk":    st.column_config.TextColumn(width="small"),
            "Historical Share":   st.column_config.TextColumn(width="small"),
        }
        st.markdown(f'<div class="chart-card"><div class="card-title">Recommended opening stock — {len(tbl):,} products</div>',
                    unsafe_allow_html=True)
        st.dataframe(tbl,use_container_width=True,hide_index=True,
                     column_config={k:v for k,v in col_cfg.items() if k in tbl.columns},
                     height=420)
        st.download_button("⬇  Download full stock list (CSV)",
                           tbl.to_csv(index=False).encode("utf-8"),
                           "new_branch_opening_stock.csv","text/csv")
        st.markdown('</div>',unsafe_allow_html=True)

    else:
        out = cat_out.copy()
        keep_c = ["Category" if "Category" in cat_out.columns else "therapeutic_group",
                  "Opening Stock Qty" if "Opening Stock Qty" in cat_out.columns else "opening_stock_qty",
                  "Confidence","Dead Stock Risk"]
        out = out[[c for c in keep_c if c in out.columns]]
        st.markdown('<div class="chart-card"><div class="card-title">Opening stock — category level</div>',
                    unsafe_allow_html=True)
        st.dataframe(out,use_container_width=True,hide_index=True,height=380)
        st.download_button("⬇  Download CSV",out.to_csv(index=False).encode("utf-8"),
                           "new_branch_stock_categories.csv","text/csv")
        st.markdown('</div>',unsafe_allow_html=True)

    # ── Do not order ──────────────────────────────────────────────────────────
    if has_products and not prod_out.empty and "Dead Stock Risk" in prod_out.columns:
        dead_list = prod_out[prod_out["Dead Stock Risk"]=="Yes"].copy()
        if not dead_list.empty:
            sh(f"Do Not Order — {len(dead_list)} Products Flagged")
            st.markdown(f"""
            <div style="background:#FEF0F0;border:1px solid rgba(224,92,92,.2);border-radius:10px;
                        padding:12px 16px;margin-bottom:12px">
              <span style="color:{CORAL};font-weight:700;font-size:.85rem">
                These {len(dead_list)} products are predicted to sit on the shelf for 85+ days.
              </span>
              <span style="color:rgba(160,32,32,.75);font-size:.82rem">
                Remove from the opening order, or ask for consignment terms.
              </span>
            </div>""",unsafe_allow_html=True)
            pnd  = "Product" if "Product" in dead_list.columns else "product_name"
            catd = "Category" if "Category" in dead_list.columns else "therapeutic_group"
            dcols = [c for c in [pnd,catd,"Confidence"] if c in dead_list.columns]
            if "est_revenue" in dead_list.columns: dcols.append("est_revenue")
            dead_show = dead_list[dcols].rename(columns={
                pnd:"Product",catd:"Category","est_revenue":"Predicted Revenue (KES)"
            }).reset_index(drop=True)
            st.markdown('<div class="chart-card"><div class="card-title">Dead stock list — remove before ordering</div>',
                        unsafe_allow_html=True)
            st.dataframe(dead_show,use_container_width=True,hide_index=True,
                         column_config={
                             "Predicted Revenue (KES)":st.column_config.NumberColumn(format="KES %,.0f"),
                             "Product":st.column_config.TextColumn(width="large"),
                             "Category":st.column_config.TextColumn(width="medium"),
                             "Confidence":st.column_config.TextColumn(width="small"),
                         },height=280)
            st.download_button("⬇  Download dead stock list (CSV)",
                               dead_show.to_csv(index=False).encode("utf-8"),
                               "new_branch_dead_stock.csv","text/csv")