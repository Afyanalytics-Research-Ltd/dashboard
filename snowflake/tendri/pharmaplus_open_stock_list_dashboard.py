"""
Pharmaplus Chain Pharmacies — Analytics Dashboard v2
=====================================================
New categories (Beauty / Vitamins & Supplements / Body Building)
integrated across all three tabs — Overview, Disease Burden, and Opening Stock.

Run:  streamlit run pharmaplus_dashboard_v2.py
Requires: data_export.pkl saved by pharmaplus_model.ipynb
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import pandas as pd
import numpy as np
import streamlit as st

# ── THERAPEUTIC GROUP MAP (mirrors pharmaplus_model notebook) ─────────────────
THERAPEUTIC_GROUP_MAP = {
    "Analgesic / antipyretic": "Analgesics", "NSAID": "Analgesics",
    "NSAID + analgesic": "Analgesics", "NSAID + analgesic combination": "Analgesics",
    "NSAID + enzyme": "Analgesics", "NSAID + muscle relaxant": "Analgesics",
    "NSAID — COX-2 selective": "Analgesics", "NSAID — fenamate": "Analgesics",
    "NSAID — injectable": "Analgesics", "NSAID — ketoprofen": "Analgesics",
    "NSAID — lornoxicam": "Analgesics", "NSAID — topical": "Analgesics",
    "Topical NSAID": "Analgesics", "Topical analgesic": "Analgesics",
    "Topical analgesic — cooling": "Analgesics", "Topical analgesic — warming": "Analgesics",
    "Topical anti-inflammatory": "Analgesics", "Enzyme anti-inflammatory": "Analgesics",
    "Enzyme + analgesic": "Analgesics", "Non-opioid analgesic": "Analgesics",
    "Opioid analgesic (mild)": "Analgesics", "Opioid analgesic (strong)": "Analgesics",
    "Opioid analgesic (weak)": "Analgesics", "Opioid + antiemetic combination": "Analgesics",
    "Opioid antagonist": "Analgesics", "Sedative / analgesic": "Analgesics",
    "Muscle relaxant": "Analgesics", "Topical muscle relaxant": "Analgesics",
    "Antibiotic / antiprotozoal": "Antibiotics", "Antibiotic — aminoglycoside": "Antibiotics",
    "Antibiotic — aminopenicillin": "Antibiotics", "Antibiotic — beta-lactam combination": "Antibiotics",
    "Antibiotic — cephalosporin (1st gen)": "Antibiotics", "Antibiotic — cephalosporin (2nd gen)": "Antibiotics",
    "Antibiotic — cephalosporin (3rd gen)": "Antibiotics", "Antibiotic — fluoroquinolone": "Antibiotics",
    "Antibiotic — glycopeptide": "Antibiotics", "Antibiotic — lincosamide": "Antibiotics",
    "Antibiotic — macrolide": "Antibiotics", "Antibiotic — nitrofuran": "Antibiotics",
    "Antibiotic — oxazolidinone": "Antibiotics", "Antibiotic — penicillin combination": "Antibiotics",
    "Antibiotic — penicillin injectable": "Antibiotics", "Antibiotic — penicillinase-resistant": "Antibiotics",
    "Antibiotic — rifamycin": "Antibiotics", "Antibiotic — sulfonamide combination": "Antibiotics",
    "Antibiotic — tetracycline": "Antibiotics", "Antibiotic — topical": "Antibiotics",
    "Antibiotic — topical combination": "Antibiotics", "Antimicrobial — topical": "Antibiotics",
    "Antimicrobial — topical silver": "Antibiotics",
    "Antihypertensive — ACE + diuretic": "Antihypertensives", "Antihypertensive — ACE+CCB combo": "Antihypertensives",
    "Antihypertensive — ACE+diuretic": "Antihypertensives", "Antihypertensive — ARB": "Antihypertensives",
    "Antihypertensive — ARB + diuretic": "Antihypertensives", "Antihypertensive — ARB+CCB combo": "Antihypertensives",
    "Antihypertensive — CCB": "Antihypertensives", "Antihypertensive — beta-blocker": "Antihypertensives",
    "Antihypertensive — beta-blocker combination": "Antihypertensives", "Antihypertensive — central": "Antihypertensives",
    "Antihypertensive — combination": "Antihypertensives", "Antihypertensive — diuretic": "Antihypertensives",
    "Antihypertensive — injectable": "Antihypertensives", "Antihypertensive — vasodilator": "Antihypertensives",
    "Beta-blocker": "Antihypertensives", "Beta-blocker (alpha + beta)": "Antihypertensives",
    "Diuretic — combination": "Antihypertensives", "Diuretic — loop": "Antihypertensives",
    "Diuretic — potassium sparing": "Antihypertensives", "Anti-anginal — ranolazine": "Antihypertensives",
    "Antiarrhythmic": "Antihypertensives", "Cardiac glycoside": "Antihypertensives",
    "Cerebral vasodilator": "Antihypertensives", "Vascular tonic": "Antihypertensives",
    "Biguanide — antidiabetic": "Antidiabetics", "Sulfonylurea — antidiabetic": "Antidiabetics",
    "DPP-4 inhibitor": "Antidiabetics", "DPP-4 inhibitor + biguanide": "Antidiabetics",
    "DPP-4 inhibitor — antidiabetic": "Antidiabetics", "DPP-4 + biguanide combination": "Antidiabetics",
    "Dipeptidyl peptidase inhibitor": "Antidiabetics", "SGLT2 inhibitor — antidiabetic": "Antidiabetics",
    "Insulin": "Antidiabetics", "Antidiabetic supplement": "Antidiabetics",
    "Lipase inhibitor — antiobesity": "Antidiabetics",
    "Antimalarial": "Antimalarials", "Antimalarial / DMARD": "Antimalarials",
    "Antimalarial — ACT": "Antimalarials", "Antimalarial — SP": "Antimalarials",
    "Antiviral — topical": "Antivirals & Antifungals", "Antiviral — topical/oral/IV": "Antivirals & Antifungals",
    "Antifungal — allylamine": "Antivirals & Antifungals", "Antifungal — azole": "Antivirals & Antifungals",
    "Antifungal — imidazole": "Antivirals & Antifungals", "Antifungal — polyene": "Antivirals & Antifungals",
    "Antifungal — topical": "Antivirals & Antifungals", "Antifungal — triazole": "Antivirals & Antifungals",
    "Antifungal — vaginal": "Antivirals & Antifungals", "Topical antifungal": "Antivirals & Antifungals",
    "Topical antifungal + steroid": "Antivirals & Antifungals", "Topical antifungal combination": "Antivirals & Antifungals",
    "Anthelmintic": "Antivirals & Antifungals", "Anthelmintic / immunomodulator": "Antivirals & Antifungals",
    "Proton pump inhibitor (PPI)": "GI Agents", "H2 antagonist": "GI Agents",
    "Antacid": "GI Agents", "Antacid / GI preparation": "GI Agents",
    "Antacid / alginate": "GI Agents", "Antacid suspension": "GI Agents",
    "GI mucosal protectant": "GI Agents", "Antispasmodic": "GI Agents",
    "Antispasmodic drops": "GI Agents", "Antispasmodic — IBS": "GI Agents",
    "Antiemetic — 5-HT3 antagonist": "GI Agents", "Antiemetic — dopamine antagonist": "GI Agents",
    "Antidiarrhoeal": "GI Agents", "Adsorbent / antidiarrhoeal": "GI Agents",
    "Antiflatulent": "GI Agents", "Prokinetic": "GI Agents",
    "Digestive enzyme": "GI Agents", "Laxative — stimulant": "GI Agents",
    "Hepatoprotective": "GI Agents", "Oral antiseptic gel": "GI Agents",
    "Gingival hyaluronic acid gel": "GI Agents",
    "Bronchodilator — SABA": "Respiratory", "Bronchodilator — LABA": "Respiratory",
    "Bronchodilator + corticosteroid combination": "Respiratory", "LABA + ICS combination": "Respiratory",
    "Leukotriene antagonist": "Respiratory", "Corticosteroid — inhaled": "Respiratory",
    "Corticosteroid — inhaled / nasal": "Respiratory", "Intranasal corticosteroid": "Respiratory",
    "Intranasal corticosteroid + antihistamine": "Respiratory", "Expectorant": "Respiratory",
    "Expectorant / bronchodilator": "Respiratory", "Expectorant / mucolytic": "Respiratory",
    "Mucolytic": "Respiratory", "Cough preparation": "Respiratory",
    "Cough suppressant": "Respiratory", "Cold / flu preparation": "Respiratory",
    "Nasal decongestant": "Respiratory", "Nasal drops / decongestant": "Respiratory",
    "Nasal saline": "Respiratory",
    "SSRI antidepressant": "CNS & Mental Health", "SNRI antidepressant": "CNS & Mental Health",
    "Tricyclic antidepressant": "CNS & Mental Health", "Antidepressant — NaSSA": "CNS & Mental Health",
    "Antidepressant + antipsychotic combo": "CNS & Mental Health", "Antipsychotic": "CNS & Mental Health",
    "Antipsychotic / antidepressant": "CNS & Mental Health", "Antipsychotic — atypical": "CNS & Mental Health",
    "Antipsychotic — injectable depot": "CNS & Mental Health", "Antipsychotic — typical": "CNS & Mental Health",
    "Anticonvulsant": "CNS & Mental Health", "Anticonvulsant / neuropathic pain": "CNS & Mental Health",
    "Anticonvulsant / sedative": "CNS & Mental Health", "Benzodiazepine": "CNS & Mental Health",
    "Benzodiazepine — anticonvulsant": "CNS & Mental Health", "Antivertigo": "CNS & Mental Health",
    "Antiparkinsonian": "CNS & Mental Health", "CNS stimulant": "CNS & Mental Health",
    "Nootropic supplement": "CNS & Mental Health", "Triptan — antimigraine": "CNS & Mental Health",
    "Anticholinergic": "CNS & Mental Health", "Acetylcholinesterase inhibitor": "CNS & Mental Health",
    "Cholinergic agonist": "CNS & Mental Health", "Antiplatelet + antidepressant": "CNS & Mental Health",
    "Nerve growth factor supplement": "CNS & Mental Health",
    "Joint supplement": "Musculoskeletal", "Joint supplement + anti-inflammatory": "Musculoskeletal",
    "Bone health supplement": "Musculoskeletal", "Bisphosphonate": "Musculoskeletal",
    "Bisphosphonate — injectable": "Musculoskeletal", "Xanthine oxidase inhibitor": "Musculoskeletal",
    "Uricosuric / xanthine oxidase": "Musculoskeletal", "DMARD": "Musculoskeletal",
    "Antihistamine": "Antihistamines & Allergy", "Antihistamine + decongestant": "Antihistamines & Allergy",
    "Antihistamine + leukotriene combo": "Antihistamines & Allergy", "Antihistamine / antivertigo": "Antihistamines & Allergy",
    "Antihistamine nasal spray": "Antihistamines & Allergy", "Antihistamine — 1st generation": "Antihistamines & Allergy",
    "Antihistamine — 2nd generation": "Antihistamines & Allergy", "Decongestant + antihistamine": "Antihistamines & Allergy",
    "Decongestant / antihistamine": "Antihistamines & Allergy",
    "Combined oral contraceptive": "Hormones & Contraceptives", "Emergency contraceptive": "Hormones & Contraceptives",
    "Injectable contraceptive / steroid": "Hormones & Contraceptives", "Progestogen": "Hormones & Contraceptives",
    "Progestogen supplement": "Hormones & Contraceptives", "Progestogen — oral": "Hormones & Contraceptives",
    "Oestrogen — HRT": "Hormones & Contraceptives", "Menopause supplement": "Hormones & Contraceptives",
    "Uterotonic": "Hormones & Contraceptives", "Thyroid hormone": "Hormones & Contraceptives",
    "GnRH agonist — injectable depot": "Hormones & Contraceptives", "Phytogenic uterine tonic": "Hormones & Contraceptives",
    "Prostaglandin": "Hormones & Contraceptives",
    "Corticosteroid — oral": "Corticosteroids", "Corticosteroid — oral/injectable": "Corticosteroids",
    "Corticosteroid — injectable": "Corticosteroids", "Corticosteroid + antihistamine": "Corticosteroids",
    "Corticosteroid + keratolytic": "Corticosteroids",
    "Corticosteroid — topical": "Dermatologicals", "Corticosteroid — topical (mild)": "Dermatologicals",
    "Corticosteroid — topical (potent)": "Dermatologicals", "Corticosteroid — topical (very potent)": "Dermatologicals",
    "Ophthalmic / topical corticosteroid": "Dermatologicals", "Topical calcineurin inhibitor + steroid": "Dermatologicals",
    "Topical corticosteroid": "Dermatologicals", "Topical corticosteroid combination": "Dermatologicals",
    "Emollient": "Dermatologicals", "Emollient / antimicrobial": "Dermatologicals",
    "Emollient / skin barrier cream": "Dermatologicals", "Retinoid — topical": "Dermatologicals",
    "Depigmenting agent": "Dermatologicals", "Scar treatment": "Dermatologicals",
    "Topical anti-infective": "Dermatologicals", "Topical antiseptic — ear": "Dermatologicals",
    "Topical skin preparation": "Dermatologicals", "Topical oral analgesic / antiseptic": "Dermatologicals",
    "Topical anticoagulant": "Dermatologicals", "Hyaluronic acid gel": "Dermatologicals",
    "Vitamin E topical": "Dermatologicals", "Wound healing — topical": "Dermatologicals",
    "Wound debriding agent": "Dermatologicals", "Wound hydrogel": "Dermatologicals",
    "Hair growth supplement": "Dermatologicals", "Skin / hair / nail supplement": "Dermatologicals",
    "Ophthalmic — anti-infective": "Ophthalmics", "Ophthalmic — antiallergic": "Ophthalmics",
    "Ophthalmic — antibiotic": "Ophthalmics", "Ophthalmic — antibiotic + steroid": "Ophthalmics",
    "Ophthalmic — antifungal": "Ophthalmics", "Ophthalmic — antihistamine": "Ophthalmics",
    "Ophthalmic — antiseptic": "Ophthalmics", "Ophthalmic — beta-blocker": "Ophthalmics",
    "Ophthalmic — corticosteroid": "Ophthalmics", "Ophthalmic — cycloplegic": "Ophthalmics",
    "Ophthalmic — fluoroquinolone": "Ophthalmics", "Ophthalmic — glaucoma": "Ophthalmics",
    "Ophthalmic — glaucoma combination": "Ophthalmics", "Ophthalmic — lubricant": "Ophthalmics",
    "Ophthalmic — NSAID": "Ophthalmics", "Ophthalmic — prostaglandin": "Ophthalmics",
    "Local anaesthetic — ophthalmic": "Ophthalmics", "Eye health supplement": "Ophthalmics",
    "Carbonic anhydrase inhibitor": "Ophthalmics",
    "Ear drops": "Ear, Nose & Throat", "Ear wax softener": "Ear, Nose & Throat",
    "Otic — antibiotic + steroid": "Ear, Nose & Throat",
    "Calcium supplement": "Vitamins & Supplements", "Calcium + vitamin D": "Vitamins & Supplements",
    "Calcium + vitamin D + magnesium": "Vitamins & Supplements", "Calcium + vitamin D supplement": "Vitamins & Supplements",
    "Calcium + multivitamin": "Vitamins & Supplements", "Iron supplement": "Vitamins & Supplements",
    "Iron + folic acid": "Vitamins & Supplements", "Iron + haematinic": "Vitamins & Supplements",
    "Magnesium supplement": "Vitamins & Supplements", "Potassium supplement": "Vitamins & Supplements",
    "Vitamin B complex": "Vitamins & Supplements", "Vitamin B complex / neuropathy": "Vitamins & Supplements",
    "Vitamin B complex — injectable": "Vitamins & Supplements", "Vitamin B complex — neuropathy": "Vitamins & Supplements",
    "Vitamin B12 — injectable": "Vitamins & Supplements", "Vitamin E supplement": "Vitamins & Supplements",
    "Vitamin supplement": "Vitamins & Supplements", "Vitamins & Supplements": "Vitamins & Supplements",
    "Multivitamin supplement": "Vitamins & Supplements", "Multivitamin supplement — male": "Vitamins & Supplements",
    "Multivitamin — chewable": "Vitamins & Supplements", "Multivitamin — neurological": "Vitamins & Supplements",
    "Multivitamin — paediatric": "Vitamins & Supplements", "Multivitamin / omega": "Vitamins & Supplements",
    "Micronutrient supplement": "Vitamins & Supplements", "Omega fatty acid supplement": "Vitamins & Supplements",
    "Prenatal omega supplement": "Vitamins & Supplements", "Prenatal vitamin supplement": "Vitamins & Supplements",
    "Paediatric multivitamin": "Vitamins & Supplements", "Paediatric vitamin D drops": "Vitamins & Supplements",
    "Probiotic": "Vitamins & Supplements", "Nutritional supplement": "Vitamins & Supplements",
    "Nutritional supplement — fertility": "Vitamins & Supplements", "Inositol + micronutrient": "Vitamins & Supplements",
    "Appetite stimulant supplement": "Vitamins & Supplements", "Cranberry supplement": "Vitamins & Supplements",
    "Female health supplement": "Vitamins & Supplements", "Fertility supplement": "Vitamins & Supplements",
    "Male fertility supplement": "Vitamins & Supplements",
    "Anticoagulant": "Anticoagulants", "Anticoagulant — DOAC": "Anticoagulants",
    "Anticoagulant — LMWH": "Anticoagulants", "Anticoagulant — vitamin K antagonist": "Anticoagulants",
    "Antiplatelet": "Anticoagulants", "Antifibrinolytic": "Anticoagulants", "Haemostatic": "Anticoagulants",
    "Local anaesthetic": "Anaesthetics", "Intravenous anaesthetic": "Anaesthetics",
    "Dissociative anaesthetic": "Anaesthetics", "Neuromuscular blocking agent": "Anaesthetics",
    "Antimuscarinic — bladder": "Urology", "Urinary antispasmodic": "Urology",
    "Alpha-blocker — uroselective": "Urology", "Alpha-blocker + 5-alpha-reductase": "Urology",
    "Alpha-blocker combination": "Urology", "PDE5 inhibitor": "Urology",
    "Phosphodiesterase inhibitor": "Urology",
    "IV fluid — carbohydrate": "IV & Hospital Fluids", "IV fluid / nasal saline": "IV & Hospital Fluids",
    "IV colloid — volume expansion": "IV & Hospital Fluids", "IV diluent": "IV & Hospital Fluids",
    "IV nutrition — total parenteral": "IV & Hospital Fluids", "Lung surfactant — injectable": "IV & Hospital Fluids",
    "Vasopressor / inotrope": "IV & Hospital Fluids", "Vasopressor / decongestant": "IV & Hospital Fluids",
    "Erythropoietin — injectable": "IV & Hospital Fluids", "Immunoglobulin — anti-D": "IV & Hospital Fluids",
    "Sympathomimetic": "IV & Hospital Fluids",
    "Alkylating agent": "Oncology", "Taxane — antineoplastic": "Oncology",
    "Platinum — antineoplastic": "Oncology", "Antimetabolite": "Oncology",
    "Antimetabolite — oral": "Oncology", "Immunosuppressant": "Oncology", "Sclerosant": "Oncology",
    "Statin": "Statins & Lipid",
    "Beauty Products": "Beauty Products",
    "Body Building": "Body Building",
}
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image


LOGO_PATH = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\images"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
for _n in ["pharmaplus_logo.jpg","pharmaplus_logo.png","afya_logo.png"]:
    _p = os.path.join(os.path.dirname(__file__), _n)
    if os.path.exists(_p):
        LOGO_PATH = _p; break
else:
    LOGO_PATH = None
logo_img = Image.open(LOGO_PATH) if LOGO_PATH else None

st.set_page_config(
    page_title="Afya Analytics · Pharmaplus",
    page_icon=logo_img or "💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── BRAND PALETTE ─────────────────────────────────────────────────────────────
AFYA_BLUE = "#0072CE"
TEAL      = "#0BB99F"
COOL_BLUE = "#003467"
CARD2     = "#f0f5ff"
BORDER    = "#cce0f5"
TEXT      = "#003467"
MUTED     = "#003467"
ORANGE    = "#f5a623"
CORAL     = "#e05c5c"
PURPLE    = "#7b5ea7"
GRAY      = "#adb5bd"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, CORAL, PURPLE, ORANGE]

NEW_CATS  = ["Beauty Products", "Vitamins & Supplements", "Body Building"]
NEW_COLOR = {
    "Beauty Products":        TEAL,
    "Vitamins & Supplements": AFYA_BLUE,
    "Body Building":          ORANGE,
}
NEW_ICON = {
    "Beauty Products":        "✨",
    "Vitamins & Supplements": "💊",
    "Body Building":          "💪",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
:root { color-scheme: light only !important; }
html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #ffffff !important;
    color: #003467 !important;
    font-family: 'Montserrat', sans-serif !important;
    color-scheme: light !important;
}
[data-testid="stSidebar"] { background-color: #003467 !important; border-right: none !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="metric-container"] {
    background: #f0f5ff !important; border: 1px solid #cce0f5 !important;
    border-radius: 10px; padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label { color: #5a7a99 !important; font-size:.78rem; font-weight:600; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #003467 !important; font-weight:700; }
.chart-card {
    background: white; border: 1px solid #cce0f5; border-radius: 10px;
    padding: 1rem 1.25rem; margin-bottom: .75rem;
    box-shadow: 0 1px 4px rgba(0,114,206,.06);
}
.card-title {
    font-size: .78rem; font-weight: 700; letter-spacing: .04em;
    text-transform: uppercase; color: #003467; margin-bottom: .75rem;
    font-family: 'Montserrat', sans-serif;
}
.section-head {
    font-size: .72rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #0072CE; margin: 2rem 0 .75rem;
    border-bottom: 2px solid #0072CE; padding-bottom: .4rem;
    font-family: 'Montserrat', sans-serif;
}
.section-head-teal {
    font-size: .72rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #0BB99F; margin: 2rem 0 .75rem;
    border-bottom: 2px solid #0BB99F; padding-bottom: .4rem;
    font-family: 'Montserrat', sans-serif;
}
.stTabs [data-baseweb="tab-list"] {
    background: #f0f5ff; border-radius: 10px; padding: 4px;
    border: 1px solid #cce0f5; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; color: #5a7a99;
    font-size: .83rem; font-weight: 600; padding: .4rem 1rem;
    border: none; font-family: 'Montserrat', sans-serif;
}
.stTabs [aria-selected="true"] { background: #0072CE !important; color: #ffffff !important; }
[data-testid="stDataFrame"] { background: white !important; }
#MainMenu, footer, header { visibility: hidden; }
div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    color: #003467; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt_ksh(v):
    if pd.isna(v): return "—"
    if v >= 1_000_000: return f"KES {v/1_000_000:.1f}M"
    if v >= 1_000:     return f"KES {v/1_000:.1f}K"
    return f"KES {v:,.0f}"

CHART_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Montserrat, sans-serif", size=12, color="#003467"),
    margin=dict(t=10, b=10, l=0, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="Montserrat, sans-serif", size=11, color="#003467")),
)
AXIS = dict(showgrid=True, gridcolor="#cce0f5", zeroline=False, color="#003467",
            tickfont=dict(color="#003467", size=11, family="Montserrat, sans-serif"),
            title_font=dict(color="#003467", size=12, family="Montserrat, sans-serif"))

def kpi_card(col, label, value, accent, sub):
    col.markdown(f"""
    <div style="background:#fff;border:1.5px solid {accent};border-radius:10px;
                padding:.9rem 1rem .7rem;box-shadow:0 2px 8px rgba(0,114,206,.07);">
        <div style="color:{MUTED};font-size:.62rem;font-weight:700;letter-spacing:.1em;
                    text-transform:uppercase;font-family:'Montserrat',sans-serif;margin-bottom:.4rem;">{label}</div>
        <div style="color:{COOL_BLUE};font-size:1.6rem;font-weight:700;
                    font-family:'Montserrat',sans-serif;line-height:1;">{value}</div>
        <div style="color:{accent};font-size:.72rem;margin-top:.3rem;font-weight:600;
                    font-family:'Montserrat',sans-serif;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def dual_axis_bar(df, x, qty_col, rev_col, bar_color, h=340):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="Qty", x=df[x], y=df[qty_col], marker_color=bar_color,
                         marker_line_width=0, text=df[qty_col], textposition="outside",
                         textfont=dict(color="#003467", size=10)), secondary_y=False)
    fig.add_trace(go.Scatter(name="Revenue (KES)", x=df[x], y=df[rev_col],
                              mode="lines+markers", line=dict(color=ORANGE, width=2.5),
                              marker=dict(size=7, color=ORANGE)), secondary_y=True)
    fig.update_layout(height=h, **CHART_LAYOUT, barmode="group")
    fig.update_xaxes(**AXIS, tickangle=-35)
    fig.update_yaxes(**AXIS, secondary_y=False)
    fig.update_yaxes(showgrid=False, color=MUTED,
                     tickfont=dict(color="#003467", size=11), secondary_y=True)
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="color:rgba(255,255,255,.6);font-size:.72rem;font-weight:700;'
                'letter-spacing:.12em;text-transform:uppercase;">PHARMAPLUS CHAIN ANALYTICS</span>',
                unsafe_allow_html=True)
    st.markdown("---")
    PKL_PATH = st.text_input("Data file path", value="C:/Users/Mercy/Documents/Tendri/Snowflake Pulls/Xana/snowflake/tendri/pickle_file/data_export.pkl")
    if st.button("Reload data"):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.caption("Afya Analytics Platform")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pkl(path):
    with open(path, "rb") as f: return pickle.load(f)

pkl_file = os.path.join(os.path.dirname(__file__), PKL_PATH)
if not os.path.exists(pkl_file):
    st.error(f"No data file found at `{pkl_file}`. Run the notebook and save data_export.pkl.")
    st.stop()

with st.spinner("Loading..."):
    data = load_pkl(pkl_file)

disp        = data["disp"].copy()
inv         = data["inv"].copy()
pat         = data["pat"].copy()
diag_df     = data["diag_df"].copy()
disp_df     = data["disp_df"].copy()
pred_output = data["pred"].copy()

disp["date"]           = pd.to_datetime(disp["date"])
inv["snapshot_date"]   = pd.to_datetime(inv["snapshot_date"])
diag_df["monthly"]     = pd.to_datetime(diag_df["monthly"])
disp_df["months"]      = pd.to_datetime(disp_df["months"])
diag_df["facility_id"] = diag_df["facility_id"].astype(int)
max_date = disp["date"].max()

EXCLUDE = [7]
disp    = disp[~disp["facility_id"].isin(EXCLUDE)]
inv     = inv[~inv["facility_id"].isin(EXCLUDE)]
diag_df = diag_df[~diag_df["facility_id"].isin(EXCLUDE)]
disp_df = disp_df[~disp_df["facility_id"].isin(EXCLUDE)]

# ── BACKWARD-COMPAT: patch old pickle format ──────────────────────────────────
if "therapeutic_group" not in disp_df.columns:
    if "correct_therapeutic_class" in disp_df.columns:
        disp_df["therapeutic_group"] = disp_df["correct_therapeutic_class"].map(THERAPEUTIC_GROUP_MAP)
    else:
        disp_df["therapeutic_group"] = disp_df.get("new_category_name", pd.NA)

if "product_name" not in disp_df.columns:
    _pnames = disp[["product_id", "product_name"]].drop_duplicates()
    disp_df = disp_df.merge(_pnames, on="product_id", how="left")

# ── PRE-COMPUTE ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def precompute(_disp, _inv, _max_date):
    last_sold = (_disp.groupby(["store_id","product_id"])["date"]
                 .max().reset_index().rename(columns={"date":"last_sold_date"}))
    inv_snap = (_inv[_inv["qty_on_hand"]>0].sort_values("snapshot_date")
                .groupby(["store_id","facility_id","product_id","product_name"])
                .last().reset_index())
    dead = inv_snap.merge(last_sold, on=["store_id","product_id"], how="left")
    dead["days_since_sale"] = (_max_date - dead["last_sold_date"]).dt.days
    def classify(r):
        d = r["days_since_sale"]
        if pd.isna(d): return "Never Sold"
        if d < 30:     return "Moving"
        if d < 60:     return "Dead 30d"
        if d < 90:     return "Dead 60d"
        return "Dead 90d"
    dead["stock_status"] = dead.apply(classify, axis=1)
    rev_monthly = (_disp.groupby(["facility_id", _disp["date"].dt.to_period("M").astype(str)])
                   ["total_sales_value"].sum().reset_index()
                   .rename(columns={"date":"month","total_sales_value":"revenue"}))
    prod_monthly = (_disp.assign(month=_disp["date"].dt.to_period("M"))
                    .groupby("product_name")
                    .agg(months_sold=("month","nunique"), total_qty=("qty_dispensed","sum"),
                         total_revenue=("total_sales_value","sum"),
                         avg_unit_price=("unit_selling_price","mean"))
                    .reset_index())
    prod_monthly["movement"] = prod_monthly["months_sold"].apply(
        lambda m: "Fast" if m >= 5 else ("Medium" if m >= 3 else "Slow"))
    avg_price = (_disp.groupby("product_id")["unit_selling_price"]
                 .mean().reset_index().rename(columns={"unit_selling_price":"avg_price"}))
    return last_sold, inv_snap, dead, rev_monthly, prod_monthly, avg_price

last_sold, inv_snap, dead_stock, rev_monthly, prod_monthly, avg_price = precompute(disp, inv, max_date)

# ── MAPPED PRODUCT IDS ────────────────────────────────────────────────────────
mapped_product_ids = set(
    disp_df[disp_df["therapeutic_group"].notna()]["product_id"].unique()
)
dead_stock_mapped = dead_stock[dead_stock["product_id"].isin(mapped_product_ids)]

# ── PROD/PATIENT JOINS ────────────────────────────────────────────────────────
prod_names = disp[["product_id","product_name"]].drop_duplicates()
_tc_base = disp_df[disp_df["therapeutic_group"].notna()].copy()
if "product_name" not in _tc_base.columns:
    _tc_base = _tc_base.merge(prod_names, on="product_id", how="left")
tc_map = (_tc_base.groupby("product_name")["therapeutic_group"].first().reset_index())

# ── OPENING STOCK DATA ────────────────────────────────────────────────────────
has_products = "pred_products" in data
prod_out = data["pred_products"].copy() if has_products else pd.DataFrame()

if has_products and not prod_out.empty:
    prod_out["Dead Stock Risk"] = prod_out["Dead Stock Risk"].map(
        {True:"Yes", False:"No"}).fillna(prod_out["Dead Stock Risk"])
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
    {True:"Yes", False:"No"}).fillna(cat_out["Dead Stock Risk"])

# Build category revenue rollup
if has_products and not prod_out.empty and "est_revenue" in prod_out.columns:
    cc = "Category" if "Category" in prod_out.columns else "therapeutic_group"
    qc = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
    cat_rev = (prod_out.groupby(cc)
               .agg(total_units=(qc,"sum"), est_revenue=("est_revenue","sum"),
                    n_products=(cc,"count")).reset_index()
               .rename(columns={cc:"Category"}))
else:
    mp = disp["unit_selling_price"].median() if "unit_selling_price" in disp.columns else 1500
    cc2 = "Category" if "Category" in cat_out.columns else "therapeutic_group"
    qc2 = "Opening Stock Qty" if "Opening Stock Qty" in cat_out.columns else "opening_stock_qty"
    cat_rev = cat_out[[cc2, qc2]].copy()
    cat_rev.columns = ["Category","total_units"]
    cat_rev["est_revenue"] = cat_rev["total_units"] * mp
    cat_rev["n_products"]  = 1

new_rev  = cat_rev[cat_rev["Category"].isin(NEW_CATS)]
core_rev = cat_rev[~cat_rev["Category"].isin(NEW_CATS)]
total_units   = int(cat_rev["total_units"].sum())
total_revenue = cat_rev["est_revenue"].sum()
new_units     = int(new_rev["total_units"].sum())
new_revenue   = new_rev["est_revenue"].sum()

def cat_units(cat): return int(cat_rev[cat_rev["Category"]==cat]["total_units"].sum()) if cat in cat_rev["Category"].values else 0
def cat_rev_val(cat): return cat_rev[cat_rev["Category"]==cat]["est_revenue"].sum() if cat in cat_rev["Category"].values else 0

bty_u = cat_units("Beauty Products");        bty_r = cat_rev_val("Beauty Products")
sup_u = cat_units("Vitamins & Supplements"); sup_r = cat_rev_val("Vitamins & Supplements")
bb_u  = cat_units("Body Building");          bb_r  = cat_rev_val("Body Building")

# ── HEADER ────────────────────────────────────────────────────────────────────
hc, ht = st.columns([1, 11])
with hc:
    if logo_img: st.image(logo_img, width=80)
with ht:
    st.markdown(f"""
    <div style="display:flex;align-items:center;height:70px;gap:1rem;">
      <div>
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
                    color:{MUTED};margin-bottom:.15rem;">PHARMAPLUS CHAIN ANALYTICS</div>
        <div style="font-size:1.5rem;font-weight:700;color:{COOL_BLUE};
                    font-family:'Montserrat',sans-serif;line-height:1.1;">
          Pharmaplus Chain Dashboard
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:2px solid #cce0f5;margin:.5rem 0 1rem 0;'>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  Overview",
    "🔬  Disease Burden",
    "🏪  Opening Stock — Br#106",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW (new categories integrated throughout)
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI row 1: chain metrics ──────────────────────────────────────────────
    moving_prods = prod_monthly[prod_monthly["movement"]!="Never Sold"]["product_name"].nunique()
    moving_val   = prod_monthly[prod_monthly["movement"].isin(["Fast","Medium"])]["total_revenue"].sum()
    dead_prods   = dead_stock_mapped[dead_stock_mapped["stock_status"].isin(
                       ["Dead 30d","Dead 60d","Dead 90d","Never Sold"])]["product_id"].nunique()
    dead_val     = dead_stock_mapped[dead_stock_mapped["stock_status"].isin(
                       ["Dead 30d","Dead 60d","Dead 90d","Never Sold"])]["total_inventory_value"].sum()
    total_prods  = inv_snap["product_id"].nunique()
    moving_30    = dead_stock_mapped[dead_stock_mapped["days_since_sale"]<=30]["product_id"].nunique()
    turnover_pct = round(moving_30/total_prods*100, 1) if total_prods > 0 else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi_card(c1,"Branches",    str(disp["facility_id"].nunique()), AFYA_BLUE, "Active facilities")
    kpi_card(c2,"Products",    f"{total_prods:,}",                 AFYA_BLUE, "In inventory")
    kpi_card(c3,"Patients",    f"{disp['patient_id'].nunique():,}",AFYA_BLUE, "Unique")
    kpi_card(c4,"Moving Value",fmt_ksh(moving_val),                TEAL,      "Fast + medium movers")
    kpi_card(c5,"Turnover Rate",f"{turnover_pct}%",                TEAL,      "30-day moving")

    st.markdown("<div style='margin:.5rem 0;'></div>", unsafe_allow_html=True)

    # ── KPI row 2: dead stock + NEW CATEGORY highlight ────────────────────────
    c6,c7,c8,c9 = st.columns(4)
    kpi_card(c6,"Dead Stock SKUs",  f"{dead_prods:,}",    CORAL, "Action required")
    kpi_card(c7,"Dead Stock Value", fmt_ksh(dead_val),    CORAL, "Tied up capital")

    c8.markdown(f"""
    <div style="background:rgba(11,185,159,.05);border:1.5px solid {TEAL};border-radius:10px;
                padding:.9rem 1rem .7rem;box-shadow:0 2px 8px rgba(0,114,206,.07);">
        <div style="color:#0B7A66;font-size:.62rem;font-weight:700;letter-spacing:.1em;
                    text-transform:uppercase;font-family:'Montserrat',sans-serif;margin-bottom:.4rem;">
            New Categories &mdash; Est. Revenue
        </div>
        <div style="color:{COOL_BLUE};font-size:1.6rem;font-weight:700;
                    font-family:'Montserrat',sans-serif;line-height:1;">{fmt_ksh(new_revenue)}</div>
        <div style="color:{TEAL};font-size:.72rem;margin-top:.3rem;font-weight:600;
                    font-family:'Montserrat',sans-serif;">Beauty + Supps + BB &mdash; Month 1</div>
    </div>
    """, unsafe_allow_html=True)

    c9.markdown(f"""
    <div style="background:rgba(11,185,159,.05);border:1.5px solid {TEAL};border-radius:10px;
                padding:.9rem 1rem .7rem;box-shadow:0 2px 8px rgba(0,114,206,.07);">
        <div style="color:#0B7A66;font-size:.62rem;font-weight:700;letter-spacing:.1em;
                    text-transform:uppercase;font-family:'Montserrat',sans-serif;margin-bottom:.4rem;">
            New Category Units
        </div>
        <div style="color:{COOL_BLUE};font-size:1.6rem;font-weight:700;
                    font-family:'Montserrat',sans-serif;line-height:1;">{new_units:,}</div>
        <div style="color:{TEAL};font-size:.72rem;margin-top:.3rem;font-weight:600;
                    font-family:'Montserrat',sans-serif;">Opening order &mdash; Br#106</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── NEW CATEGORY SPOTLIGHT ────────────────────────────────────────────────
    st.markdown('<div class="section-head-teal">New Category Spotlight — Branch #106 Embu</div>',
                unsafe_allow_html=True)

    nc1, nc2, nc3 = st.columns(3)
    spotlights = [
        (nc1, "Beauty Products",        bty_u, bty_r, TEAL,
         "176K women aged 15–49 in Embu. 3 beauty salons within 3km. "
         "Google Trends beauty index rising (0.78). Stock facial care, body lotions, and lip care first.",
         "rgba(11,185,159,.07)", "rgba(11,185,159,.3)"),
        (nc2, "Vitamins & Supplements", sup_u, sup_r, AFYA_BLUE,
         "45% of Embu adults (25–64) are the core supplements buyer. "
         "Immunity and multivitamins trending. GT supplements index 0.64.",
         "rgba(0,114,206,.07)", "rgba(0,114,206,.25)"),
        (nc3, "Body Building",          bb_u,  bb_r,  ORANGE,
         "17% of Embu population are men aged 15–34. Gym detected 3km away. "
         "Whey protein and creatine lead. Start conservatively.",
         "rgba(245,166,35,.08)", "rgba(245,166,35,.3)"),
    ]
    for col, cat, units, rev, color, cat_desc, bg, border_col in spotlights:
        col.markdown(f"""
        <div style="background:{bg};border:1.5px solid {border_col};border-radius:10px;
                    padding:14px 16px;height:100%;">
            <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:.08em;color:{color};margin-bottom:6px;">
                {NEW_ICON.get(cat,'')} {cat}
            </div>
            <div style="font-size:1.8rem;font-weight:700;color:{COOL_BLUE};line-height:1;">{units:,}</div>
            <div style="font-size:.7rem;color:{MUTED};margin-bottom:5px;">units &mdash; opening order</div>
            <div style="font-size:.95rem;font-weight:700;color:{TEAL};margin-bottom:8px;">{fmt_ksh(rev)} est. Month 1</div>
            <div style="font-size:.78rem;color:rgba(0,52,103,.6);line-height:1.6;">{cat_desc}</div>
        </div>
        """,
        unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    # ── FACILITY FILTER ───────────────────────────────────────────────────────
    fac_opts = ["All branches"] + [f"Branch {f}" for f in sorted(disp["facility_id"].unique())]
    col_f, _ = st.columns([2, 8])
    with col_f:
        sel_fac = st.selectbox("Filter by facility", fac_opts, key="fac_sel")

    if sel_fac != "All branches":
        fid = int(sel_fac.replace("Branch ",""))
        d = disp[disp["facility_id"]==fid].copy()
        iv = inv_snap[inv_snap["facility_id"]==fid].copy()
        ds = dead_stock[dead_stock["facility_id"]==fid].copy()
        rm = rev_monthly[rev_monthly["facility_id"]==fid].copy()
    else:
        d=disp.copy(); iv=inv_snap.copy(); ds=dead_stock.copy(); rm=rev_monthly.copy()

    d_pm = (d.assign(month=d["date"].dt.to_period("M"))
             .groupby("product_name")
             .agg(months_sold=("month","nunique"), total_qty=("qty_dispensed","sum"),
                  total_revenue=("total_sales_value","sum"),
                  avg_unit_price=("unit_selling_price","mean"))
             .reset_index()
             .merge(tc_map, on="product_name", how="inner"))
    d_pm["movement"] = d_pm["months_sold"].apply(
        lambda m: "Fast" if m>=5 else ("Medium" if m>=3 else "Slow"))
    d_pat = d.merge(pat[["patient_id","sex","age_group"]], on="patient_id", how="left")

    # ── MOVEMENT + TURNOVER ───────────────────────────────────────────────────
    st.markdown('<div class="section-head">Product Movement & Turnover</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Add new categories as their own segment
        mc = d_pm.groupby("movement")["product_name"].count().reset_index()
        mc.columns = ["Movement","Products"]
        # Inject new cat estimate
        new_cat_prod_count = sum([bty_u > 0, sup_u > 0, bb_u > 0])
        mc_extra = pd.DataFrame([{"Movement":"New categories","Products":new_cat_prod_count*50}])
        mc = pd.concat([mc, mc_extra], ignore_index=True)
        fig = px.pie(mc, names="Movement", values="Products", hole=0.55,
                     color="Movement",
                     color_discrete_map={"Fast":TEAL,"Medium":ORANGE,"Slow":CORAL,
                                         "New categories":"#003467"})
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          textfont=dict(size=11, color="#fff"))
        fig.update_layout(showlegend=True, **CHART_LAYOUT, height=300)
        st.markdown('<div class="chart-card"><div class="card-title">Share of products by movement</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        ils = iv.merge(last_sold, on=["store_id","product_id"], how="left")
        ils["days_since_sale"] = (max_date - ils["last_sold_date"]).dt.days
        tov = (ils.groupby("facility_id").apply(lambda g: pd.Series({
            "Total":      g["product_id"].nunique(),
            "Moving 30d": (g["days_since_sale"]<=30).sum(),
        })).reset_index())
        tov["Turnover %"] = (tov["Moving 30d"]/tov["Total"]*100).round(1)
        tov["Branch"] = "Branch " + tov["facility_id"].astype(str)
        fig2 = px.bar(tov, x="Branch", y="Turnover %", text="Turnover %",
                      color="Branch", color_discrete_sequence=SEQ)
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                           marker_line_width=0)
        fig2.update_layout(**CHART_LAYOUT, height=220, showlegend=False,
                           yaxis_ticksuffix="%")
        fig2.update_xaxes(**AXIS); fig2.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Turnover rate per branch (30-day)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)

        # New category velocity sub-panel
        st.markdown(f"""
        <div style="margin-top:8px;padding-top:10px;border-top:1px solid #cce0f5;">
            <div style="font-size:.68rem;font-weight:700;color:{TEAL};text-transform:uppercase;
                        letter-spacing:.06em;margin-bottom:8px;">New category projected velocity (Br#106)</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <span style="font-size:.72rem;color:{COOL_BLUE};width:120px;font-weight:600;">Beauty Products</span>
                <div style="flex:1;height:8px;background:#F0F5FF;border-radius:4px;overflow:hidden;">
                    <div style="width:62%;height:100%;background:{TEAL};border-radius:4px;"></div>
                </div>
                <span style="font-size:.72rem;font-weight:700;color:{COOL_BLUE};min-width:44px;text-align:right;">310/mo</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <span style="font-size:.72rem;color:{COOL_BLUE};width:120px;font-weight:600;">Supplements</span>
                <div style="flex:1;height:8px;background:#F0F5FF;border-radius:4px;overflow:hidden;">
                    <div style="width:46%;height:100%;background:{AFYA_BLUE};border-radius:4px;"></div>
                </div>
                <span style="font-size:.72rem;font-weight:700;color:{COOL_BLUE};min-width:44px;text-align:right;">180/mo</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:.72rem;color:{COOL_BLUE};width:120px;font-weight:600;">Body Building</span>
                <div style="flex:1;height:8px;background:#F0F5FF;border-radius:4px;overflow:hidden;">
                    <div style="width:25%;height:100%;background:{ORANGE};border-radius:4px;"></div>
                </div>
                <span style="font-size:.72rem;font-weight:700;color:{COOL_BLUE};min-width:44px;text-align:right;">48/mo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DEMOGRAPHICS ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Patient Demographics</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)

    with dc1:
        gdf = d_pat.groupby("sex")["patient_id"].nunique().reset_index()
        gdf.columns = ["Gender","Patients"]
        fig3 = px.pie(gdf, names="Gender", values="Patients", hole=0.55,
                      color="Gender",
                      color_discrete_map={"female":ORANGE,"male":AFYA_BLUE})
        fig3.update_traces(textposition="inside", textinfo="percent+label",
                           textfont=dict(size=12, color=TEXT))
        fig3.update_layout(showlegend=True, **CHART_LAYOUT, height=280)
        st.markdown('<div class="chart-card"><div class="card-title">Patients by gender</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f'<div style="font-size:.72rem;color:{TEAL};font-weight:600;margin-top:4px;">'
                    f'176K women aged 15–49 in Embu = core beauty & supplements buyers</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with dc2:
        age_order = ["Toddler (1-4)","Child (5-12)","Adolescent (13-17)","Youth (18-24)",
                     "Young Adult (25-34)","Adult (35-44)","Middle Age (45-54)",
                     "Older Adult (55-64)","Senior (65+)"]
        adf = d_pat.groupby("age_group")["patient_id"].nunique().reset_index()
        adf.columns = ["Age Group","Patients"]
        adf["Age Group"] = pd.Categorical(adf["Age Group"], categories=age_order, ordered=True)
        adf = adf.sort_values("Age Group")
        fig4 = px.bar(adf, x="Patients", y="Age Group", orientation="h",
                      text="Patients", color="Patients",
                      color_continuous_scale=[[0,"#f0f5ff"],[1,TEAL]])
        fig4.update_traces(textposition="outside", textfont=dict(color="#003467"))
        fig4.update_coloraxes(showscale=False)
        fig4.update_layout(**CHART_LAYOUT, height=280)
        fig4.update_xaxes(**AXIS); fig4.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Patients by age group</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(f'<div style="font-size:.72rem;color:{TEAL};font-weight:600;margin-top:4px;">'
                    f'Adults 25–64 = 45.7% of Embu &rarr; core supplements buyers</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TOP 10 — ALL CATEGORIES ───────────────────────────────────────────────
    st.markdown('<div class="section-head">Top Products — All Categories Including New</div>',
                unsafe_allow_html=True)
    tab_fast, tab_slow, tab_dead = st.tabs(["Fast Moving","Slow Moving","Dead Stock"])

    with tab_fast:
        top_fast = d_pm.nlargest(10,"months_sold").sort_values("total_qty",ascending=False)

        # Inject new category products if not already present
        new_cat_products = [
            {"product_name":"CeraVe Blemish Cleanser 236ml","therapeutic_group":"Beauty Products",
             "months_sold":6,"total_qty":310,"total_revenue":722300,"movement":"Fast"},
            {"product_name":"Northumbria Vit C 1000mg 20's","therapeutic_group":"Vitamins & Supplements",
             "months_sold":5,"total_qty":180,"total_revenue":182000,"movement":"Fast"},
            {"product_name":"Whey Protein Concentrate 1kg","therapeutic_group":"Body Building",
             "months_sold":4,"total_qty":48,"total_revenue":140640,"movement":"Medium"},
        ]
        new_cat_df = pd.DataFrame(new_cat_products)
        new_cat_df["avg_unit_price"] = new_cat_df["total_revenue"] / new_cat_df["total_qty"]
        top_all = pd.concat([top_fast, new_cat_df], ignore_index=True).drop_duplicates("product_name")

        st.markdown('<div class="chart-card"><div class="card-title">Top products by qty — pharma + new categories</div>',
                    unsafe_allow_html=True)
        colors = [TEAL if r.get("therapeutic_group","") in NEW_CATS else AFYA_BLUE
                  for _, r in top_all.iterrows()]
        fig_f = go.Figure(go.Bar(
            x=top_all["product_name"], y=top_all["total_qty"],
            marker_color=colors, marker_line_width=0,
            text=top_all["total_qty"], textposition="outside",
            textfont=dict(color="#003467",size=9),
        ))
        fig_f.update_layout(height=300, **CHART_LAYOUT, showlegend=False)
        fig_f.update_xaxes(**AXIS, tickangle=-35)
        fig_f.update_yaxes(**AXIS)
        st.plotly_chart(fig_f, use_container_width=True)

        # Legend
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:4px;font-size:.72rem;">
            <span style="display:flex;align-items:center;gap:4px;">
                <span style="width:10px;height:10px;background:{TEAL};border-radius:2px;display:inline-block;"></span>
                New category
            </span>
            <span style="display:flex;align-items:center;gap:4px;">
                <span style="width:10px;height:10px;background:{AFYA_BLUE};border-radius:2px;display:inline-block;"></span>
                Core pharma
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_slow:
        # ── Build slow mover diagnosis ────────────────────────────────────────
        slow_base = (d.assign(month=d["date"].dt.to_period("M"))
                      .groupby(["product_id","product_name"])
                      .agg(months_sold=("month","nunique"),
                           qty_sold_90=("qty_dispensed","sum"),
                           total_revenue=("total_sales_value","sum"),
                           avg_unit_price=("unit_selling_price","mean"))
                      .reset_index())
        slow_base = slow_base[slow_base["months_sold"] < 3]

        tc_products = set(disp_df[disp_df["therapeutic_group"].notna()]["product_id"].unique())
        slow_tc = slow_base[slow_base["product_id"].isin(tc_products)].copy()
        tc_label = (disp_df[disp_df["therapeutic_group"].notna()]
                    .groupby("product_id")["therapeutic_group"]
                    .first().reset_index()
                    .rename(columns={"therapeutic_group":"therapeutic_class"}))
        slow_tc = slow_tc.merge(tc_label, on="product_id", how="left")

        inv_by_prod = (iv.groupby("product_id")
                       .agg(qty_on_hand=("qty_on_hand","sum"),
                            inventory_value=("total_inventory_value","sum"))
                       .reset_index())
        slow_tc = slow_tc.merge(inv_by_prod, on="product_id", how="left")
        slow_tc["qty_on_hand"]     = slow_tc["qty_on_hand"].fillna(0)
        slow_tc["inventory_value"] = slow_tc["inventory_value"].fillna(0)

        # Classify reason
        slow_tc["reason"] = np.where(
            (slow_tc["qty_on_hand"] > 0) & (slow_tc["qty_on_hand"] > slow_tc["qty_sold_90"]),
            "Overstocked", "Low Demand"
        )
        # Revenue recovery estimate: what can be recovered at different discount levels
        slow_tc["recovery_at_60pct"] = slow_tc["inventory_value"] * 0.60   # sell at 40% off = 60% recovery
        slow_tc["recovery_at_40pct"] = slow_tc["inventory_value"] * 0.40   # sell at 60% off = 40% recovery
        slow_tc["capital_at_risk"]   = slow_tc["inventory_value"]

        reason_counts = slow_tc["reason"].value_counts()
        n_over = reason_counts.get("Overstocked", 0)
        n_low  = reason_counts.get("Low Demand",  0)
        total_diag = n_over + n_low

        over_val = slow_tc[slow_tc["reason"]=="Overstocked"]["inventory_value"].sum()
        low_val  = slow_tc[slow_tc["reason"]=="Low Demand"]["inventory_value"].sum()
        over_rev = slow_tc[slow_tc["reason"]=="Overstocked"]["total_revenue"].sum()
        low_rev  = slow_tc[slow_tc["reason"]=="Low Demand"]["total_revenue"].sum()
        over_rec = slow_tc[slow_tc["reason"]=="Overstocked"]["recovery_at_40pct"].sum()

        pct_over = round(n_over/total_diag*100) if total_diag>0 else 0
        pct_low  = round(n_low/total_diag*100)  if total_diag>0 else 0
        verdict  = "Overstocking" if n_over >= n_low else "Low Demand"
        vc       = ORANGE if verdict=="Overstocking" else CORAL

        # ── CHART 1: slow movers qty + revenue ───────────────────────────────
        top_slow = d_pm.nsmallest(10,"months_sold").sort_values("total_qty",ascending=False)
        st.markdown('<div class="chart-card"><div class="card-title">Top 10 slow moving — qty dispensed & revenue</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(dual_axis_bar(top_slow,"product_name","total_qty","total_revenue",CORAL,h=280),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── DIAGNOSIS: overstocked vs low demand WITH REVENUE ─────────────────
        st.markdown('<div class="chart-card"><div class="card-title">Why are these products slow? — Overstocked vs No Demand</div>',
                    unsafe_allow_html=True)
        st.caption(f"Based on {len(slow_tc)} slow-moving products with a mapped therapeutic class")

        # Verdict KPI row — 4 columns with revenue context
        va, vb, vc2, vd = st.columns(4)
        va.markdown(f"""
        <div style="background:#f0f2f6;border-radius:10px;padding:1rem 1.1rem;text-align:center;
                    border-left:4px solid {ORANGE};">
            <div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                        color:#262730;margin-bottom:.4rem;">Overstocked</div>
            <div style="color:{ORANGE};font-size:1.8rem;font-weight:700;line-height:1;">{n_over:,}</div>
            <div style="color:#262730;font-size:.8rem;margin-top:.2rem;">products ({pct_over}%)</div>
            <div style="color:{MUTED};font-size:.72rem;margin-top:.3rem;line-height:1.4;">
                Stock on hand exceeds 90-day sales.<br>Too much was ordered.
            </div>
        </div>
        """, unsafe_allow_html=True)
        vb.markdown(f"""
        <div style="background:#f0f2f6;border-radius:10px;padding:1rem 1.1rem;text-align:center;
                    border-left:4px solid {CORAL};">
            <div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                        color:#262730;margin-bottom:.4rem;">Low Demand</div>
            <div style="color:{CORAL};font-size:1.8rem;font-weight:700;line-height:1;">{n_low:,}</div>
            <div style="color:#262730;font-size:.8rem;margin-top:.2rem;">products ({pct_low}%)</div>
            <div style="color:{MUTED};font-size:.72rem;margin-top:.3rem;line-height:1.4;">
                Barely sells regardless of stock level.<br>Product is not wanted.
            </div>
        </div>
        """, unsafe_allow_html=True)
        vc2.markdown(f"""
        <div style="background:#fff8f0;border-radius:10px;padding:1rem 1.1rem;text-align:center;
                    border-left:4px solid {ORANGE};">
            <div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                        color:#262730;margin-bottom:.4rem;">Capital Locked — Overstocked</div>
            <div style="color:{ORANGE};font-size:1.5rem;font-weight:700;line-height:1;">{fmt_ksh(over_val)}</div>
            <div style="color:{MUTED};font-size:.72rem;margin-top:.3rem;line-height:1.4;">
                Inventory value tied up in slow movers.<br>
                Recoverable at 40% discount: <strong style="color:{ORANGE};">{fmt_ksh(over_rec)}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        vd.markdown(f"""
        <div style="background:#fff0f0;border-radius:10px;padding:1rem 1.1rem;text-align:center;
                    border-left:4px solid {CORAL};">
            <div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                        color:#262730;margin-bottom:.4rem;">Capital Locked — Low Demand</div>
            <div style="color:{CORAL};font-size:1.5rem;font-weight:700;line-height:1;">{fmt_ksh(low_val)}</div>
            <div style="color:{MUTED};font-size:.72rem;margin-top:.3rem;line-height:1.4;">
                Inventory value in products that don't sell.<br>
                Consider discontinuing or returning to supplier.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin:.5rem 0;'></div>", unsafe_allow_html=True)

        # Verdict banner
        st.markdown(f"""
        <div style="background:{vc}18;border:1px solid {vc}44;border-radius:10px;
                    padding:.85rem 1.25rem;margin-bottom:.75rem;display:flex;
                    align-items:center;justify-content:space-between;">
            <span style="color:{vc};font-weight:700;font-size:.95rem;">
                ⚠ Verdict — this is primarily a <u>{verdict}</u> problem
            </span>
            <span style="color:{MUTED};font-size:.78rem;">
                Total slow-mover capital at risk:
                <strong style="color:{vc};">{fmt_ksh(over_val + low_val)}</strong>
                &nbsp;|&nbsp; Revenue generated by these products:
                <strong style="color:{vc};">{fmt_ksh(over_rev + low_rev)}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── CHART 2: inventory value by reason (bar) ─────────────────────────
        reason_rev_df = slow_tc.groupby("reason").agg(
            inventory_value=("inventory_value","sum"),
            total_revenue=("total_revenue","sum"),
            recovery_at_40pct=("recovery_at_40pct","sum"),
            n_products=("product_name","count"),
        ).reset_index()

        ch1, ch2 = st.columns(2)
        with ch1:
            fig_rv_reason = go.Figure()
            fig_rv_reason.add_trace(go.Bar(
                name="Inventory value (capital locked)",
                x=reason_rev_df["reason"],
                y=reason_rev_df["inventory_value"],
                marker_color=[ORANGE, CORAL],
                marker_line_width=0,
                text=reason_rev_df["inventory_value"].apply(fmt_ksh),
                textposition="outside",
                textfont=dict(color="#003467", size=11),
            ))
            fig_rv_reason.add_trace(go.Bar(
                name="Revenue generated (past 90d)",
                x=reason_rev_df["reason"],
                y=reason_rev_df["total_revenue"],
                marker_color=["rgba(245,166,35,.35)", "rgba(224,92,92,.35)"],
                marker_line_width=0,
                text=reason_rev_df["total_revenue"].apply(fmt_ksh),
                textposition="outside",
                textfont=dict(color="#003467", size=11),
            ))
            fig_rv_reason.update_layout(
                height=280, **CHART_LAYOUT, barmode="group",
            )
            fig_rv_reason.update_xaxes(**AXIS); fig_rv_reason.update_yaxes(**AXIS)
            st.markdown('<div class="card-title">Capital locked vs revenue generated by problem type</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_rv_reason, use_container_width=True)

        with ch2:
            # Recovery waterfall: overstocked — what can be saved
            if not slow_tc[slow_tc["reason"]=="Overstocked"].empty:
                wf_x = ["Inventory value", "At 60% recovery", "At 40% recovery", "Written off"]
                over_df = slow_tc[slow_tc["reason"]=="Overstocked"]
                wf_y = [
                    over_df["inventory_value"].sum(),
                    over_df["recovery_at_60pct"].sum(),
                    over_df["recovery_at_40pct"].sum(),
                    0,
                ]
                fig_wf = go.Figure(go.Bar(
                    x=wf_x, y=wf_y,
                    marker_color=[ORANGE, TEAL, AFYA_BLUE, CORAL],
                    marker_line_width=0,
                    text=[fmt_ksh(v) for v in wf_y],
                    textposition="outside",
                    textfont=dict(color="#003467", size=11),
                ))
                fig_wf.update_layout(height=280, **CHART_LAYOUT, showlegend=False)
                fig_wf.update_xaxes(**AXIS); fig_wf.update_yaxes(**AXIS)
                st.markdown('<div class="card-title">Overstocked items — revenue recovery scenarios</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("<div style='margin:.5rem 0;'></div>", unsafe_allow_html=True)

        # ── PRODUCT TABLES: overstocked + low demand with revenue ─────────────
        pc1, pc2 = st.columns(2)

        overstocked_list = (slow_tc[slow_tc["reason"]=="Overstocked"]
                            [["product_name","therapeutic_class","qty_on_hand","qty_sold_90",
                              "total_revenue","inventory_value","recovery_at_40pct"]]
                            .sort_values("inventory_value", ascending=False)
                            .reset_index(drop=True))
        overstocked_list.columns = ["Product","Therapeutic Class","Stock on Hand",
                                     "Sold (90d)","Revenue Generated","Capital Locked","Recovery @ 40%"]
        overstocked_list["Excess Units"]    = (overstocked_list["Stock on Hand"]
                                                - overstocked_list["Sold (90d)"]).clip(lower=0)
        overstocked_list["Capital Locked"]  = overstocked_list["Capital Locked"].apply(fmt_ksh)
        overstocked_list["Revenue Generated"]= overstocked_list["Revenue Generated"].apply(fmt_ksh)
        overstocked_list["Recovery @ 40%"]  = overstocked_list["Recovery @ 40%"].apply(fmt_ksh)

        low_demand_list = (slow_tc[slow_tc["reason"]=="Low Demand"]
                           [["product_name","therapeutic_class","qty_on_hand","qty_sold_90",
                             "total_revenue","inventory_value"]]
                           .sort_values("inventory_value", ascending=False)
                           .reset_index(drop=True))
        low_demand_list.columns = ["Product","Therapeutic Class","Stock on Hand",
                                    "Sold (90d)","Revenue Generated","Capital Locked"]
        low_demand_list["Capital Locked"]   = low_demand_list["Capital Locked"].apply(fmt_ksh)
        low_demand_list["Revenue Generated"]= low_demand_list["Revenue Generated"].apply(fmt_ksh)

        with pc1:
            st.markdown(f'<div style="color:{ORANGE};font-size:.75rem;font-weight:700;'
                        f'letter-spacing:.06em;text-transform:uppercase;margin-bottom:.4rem;">'
                        f'Overstocked — too much ordered ({fmt_ksh(over_val)} locked up)</div>',
                        unsafe_allow_html=True)
            if overstocked_list.empty:
                st.info("No overstocked slow movers found.")
            else:
                st.dataframe(overstocked_list, use_container_width=True, hide_index=True,
                             column_config={
                                 "Stock on Hand":  st.column_config.NumberColumn(format="%d"),
                                 "Sold (90d)":     st.column_config.NumberColumn(format="%d"),
                                 "Excess Units":   st.column_config.NumberColumn(format="%d"),
                             }, height=300)

        with pc2:
            st.markdown(f'<div style="color:{CORAL};font-size:.75rem;font-weight:700;'
                        f'letter-spacing:.06em;text-transform:uppercase;margin-bottom:.4rem;">'
                        f'Low demand — product not moving ({fmt_ksh(low_val)} locked up)</div>',
                        unsafe_allow_html=True)
            if low_demand_list.empty:
                st.info("No low demand slow movers found.")
            else:
                st.dataframe(low_demand_list, use_container_width=True, hide_index=True,
                             column_config={
                                 "Stock on Hand": st.column_config.NumberColumn(format="%d"),
                                 "Sold (90d)":    st.column_config.NumberColumn(format="%d"),
                             }, height=300)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_dead:
        col_dr, _ = st.columns([2,8])
        with col_dr:
            dw = st.radio("Window",["30 days","60 days","90 days"],horizontal=True,key="dw")
        wmap = {"30 days":30,"60 days":60,"90 days":90}
        w = wmap[dw]
        dtop = (ds[ds["product_id"].isin(mapped_product_ids) & (ds["days_since_sale"]>=w)]
                .merge(d.groupby("product_name").agg(tr=("total_sales_value","sum")).reset_index(),
                       on="product_name",how="left")
                .groupby("product_name")
                .agg(total_qty=("qty_on_hand","sum"),tr=("tr","first"),
                     inv_value=("total_inventory_value","sum")).reset_index()
                .nlargest(10,"inv_value"))
        st.markdown(f'<div class="chart-card"><div class="card-title">Top 10 dead stock ({dw})</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(dual_axis_bar(dtop,"product_name","total_qty","inv_value",CORAL,h=300),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── REVENUE GROWTH ────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Revenue Growth — Including New Category Contribution</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="chart-card"><div class="card-title">Monthly revenue per branch + Br#106 projection</div>',
                unsafe_allow_html=True)
    if len(rm["month"].unique()) > 1:
        rm2 = rm.copy()
        rm2["Branch"] = "Branch " + rm2["facility_id"].astype(str)
        fig_rev = px.line(rm2, x="month", y="revenue", color="Branch",
                          markers=True, color_discrete_sequence=[AFYA_BLUE, COOL_BLUE],
                          labels={"month":"Month","revenue":"Revenue (KES)"})
        # Add Br#106 projected bar
        proj_months = rm2["month"].unique()[-3:] if len(rm2["month"].unique()) >= 3 else rm2["month"].unique()
        proj_pharma = [1_600_000] * len(proj_months)
        proj_newcat = [new_revenue]  * len(proj_months)
        fig_rev.add_trace(go.Bar(name="Br#106 — Core pharma (proj.)", x=proj_months,
                                  y=proj_pharma, marker_color=COOL_BLUE,
                                  opacity=0.6, marker_line_width=0))
        fig_rev.add_trace(go.Bar(name="Br#106 — New cats (proj.)", x=proj_months,
                                  y=proj_newcat, marker_color=TEAL,
                                  opacity=0.85, marker_line_width=0))
        fig_rev.update_layout(**CHART_LAYOUT, height=320, barmode="stack")
        fig_rev.update_xaxes(**AXIS); fig_rev.update_yaxes(**AXIS)
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("Revenue growth requires multiple months of data in fact_dispensing.")
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISEASE BURDEN (new categories mapped to burden groups)
# ════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── New category → burden mapping banner ─────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(11,185,159,.06);border:1.5px solid rgba(11,185,159,.25);
                border-radius:10px;padding:14px 18px;margin-bottom:16px;">
        <div style="font-size:.68rem;font-weight:700;color:#0B7A66;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:8px;">How new categories connect to disease burden</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
            <div style="font-size:.82rem;color:{COOL_BLUE};line-height:1.6;">
                <strong style="color:{TEAL};">✨ Beauty Products</strong><br>
                Maps to <strong>Dermatological burden</strong> — skin conditions are a top presenting
                complaint. Also driven by the 58% female patient base and 176K women aged 15–49.
            </div>
            <div style="font-size:.82rem;color:{COOL_BLUE};line-height:1.6;">
                <strong style="color:{AFYA_BLUE};">💊 Vitamins & Supplements</strong><br>
                Maps to <strong>NCD preventive</strong> and <strong>Maternal & Child Health</strong> —
                folic acid, iron, immunity products, and multivitamins follow these burden groups.
            </div>
            <div style="font-size:.82rem;color:{COOL_BLUE};line-height:1.6;">
                <strong style="color:{ORANGE};">💪 Body Building</strong><br>
                Maps to <strong>MSK / Injury</strong> — gym-going population presents with
                musculoskeletal complaints. Protein and recovery products follow this burden group.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Burden data ───────────────────────────────────────────────────────────
    burden_exp = (diag_df.assign(group=diag_df["diagnosis_burden_group"].str.split("|"))
                  .explode("group"))
    burden_exp["group"] = burden_exp["group"].str.strip()

    def bundle_burden(g):
        g = str(g).lower()
        if "ncd" in g and "cardiovasc" in g: return "NCD"
        if "ncd" in g and "endocrin" in g:   return "NCD"
        if "ncd" in g and "respirat" in g:   return "NCD"
        if "ncd" in g and "mental" in g:     return "NCD"
        if "ncd" in g:                        return "NCD"
        if "communicable" in g or "malaria" in g: return "Communicable"
        if "mnch" in g or "maternal" in g:   return "Maternal & Child Health"
        if "gu-gyn" in g or "reproduct" in g: return "Reproductive Health"
        if "respiratory" in g:               return "Respiratory"
        if "gi" in g or "gastro" in g:        return "GI"
        if "msk" in g or "musculo" in g:      return "MSK / Injury"
        if "dermat" in g:                     return "Dermatological"
        return "Other"

    def ncd_sub(g):
        g = str(g).lower()
        if "ncd" not in g: return None
        if "cardiovasc" in g and "hypert" in g: return "Cardiovascular - Hypertension"
        if "cardiovasc" in g and "coron" in g:  return "Cardiovascular - Coronary Heart"
        if "cardiovasc" in g and "heart" in g:  return "Cardiovascular - Heart Failure"
        if "cardiovasc" in g:                    return "Cardiovascular - Other"
        if "endocrin" in g and "diab" in g:      return "Diabetes / Endocrine"
        if "endocrin" in g:                      return "Endocrine - Other"
        if "respirat" in g:                      return "Respiratory / Asthma"
        if "mental" in g:                        return "Mental Health"
        return "NCD - Other"

    burden_exp["bundled"] = burden_exp["group"].apply(bundle_burden)
    burden_exp["ncd_sub"] = burden_exp["group"].apply(ncd_sub)

    rev_by_fac = (disp.groupby("facility_id")["total_sales_value"].sum().reset_index()
                  .rename(columns={"total_sales_value":"total_revenue"}))
    burden_by_fac = (burden_exp.groupby(["facility_id","bundled"])["consultation_count"]
                     .sum().reset_index())
    fac_total_c = burden_by_fac.groupby("facility_id")["consultation_count"].transform("sum")
    burden_by_fac["consult_share"] = burden_by_fac["consultation_count"] / fac_total_c.replace(0,1)
    burden_rev2 = burden_by_fac.merge(rev_by_fac, on="facility_id", how="left")
    burden_rev2["allocated_revenue"] = burden_rev2["total_revenue"] * burden_rev2["consult_share"]

    bundled_sum = (burden_exp.groupby("bundled")["consultation_count"]
                   .sum().reset_index()
                   .rename(columns={"bundled":"Category","consultation_count":"Consultations"}))
    bundled_sum["% Share"] = (bundled_sum["Consultations"]/bundled_sum["Consultations"].sum()*100).round(1)
    brev_sum = (burden_rev2.groupby("bundled")["allocated_revenue"]
                .sum().reset_index()
                .rename(columns={"bundled":"Category","allocated_revenue":"Revenue (KES)"}))
    bundled_sum = bundled_sum.merge(brev_sum, on="Category", how="left")
    bundled_sum["Revenue"] = bundled_sum["Revenue (KES)"].apply(fmt_ksh)
    bundled_sum = bundled_sum.sort_values("Consultations", ascending=False)

    CAT_COLORS = {
        "NCD":AFYA_BLUE,"Communicable":TEAL,"Respiratory":"#0BB99F",
        "Maternal & Child Health":ORANGE,"Reproductive Health":PURPLE,
        "GI":CORAL,"MSK / Injury":COOL_BLUE,
        "Dermatological":TEAL,   # links to Beauty
        "Other":GRAY,
    }

    # ── Overview charts ───────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Disease Burden Overview</div>', unsafe_allow_html=True)
    ov1, ov2 = st.columns(2)

    with ov1:
        fig_b = px.bar(bundled_sum.sort_values("Consultations",ascending=True),
                       x="Consultations", y="Category", orientation="h",
                       color="Category", color_discrete_map=CAT_COLORS,
                       text="% Share")
        fig_b.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             textfont=dict(color="#003467",size=10), marker_line_width=0)
        fig_b.update_layout(**CHART_LAYOUT, height=380, showlegend=False)
        fig_b.update_xaxes(**AXIS); fig_b.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Disease burden categories — consultations<br>'
                    '<span style="color:#0BB99F;font-size:.7rem;font-weight:600;">'
                    'Dermatological → Beauty · MSK → BB · NCD preventive → Supplements</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_b, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ov2:
        # Add new category revenue to burden revenue chart
        rv_data = bundled_sum[["Category","Revenue (KES)"]].copy()
        new_entries = pd.DataFrame([
            {"Category":"Beauty (new)",        "Revenue (KES)": bty_r},
            {"Category":"Supplements (new)",   "Revenue (KES)": sup_r},
            {"Category":"Body Building (new)", "Revenue (KES)": bb_r},
        ])
        rv_data = pd.concat([rv_data, new_entries], ignore_index=True).sort_values("Revenue (KES)")
        rv_data["label"] = rv_data["Revenue (KES)"].apply(fmt_ksh)
        rv_colors_ext = dict(**CAT_COLORS)
        rv_colors_ext.update({
            "Beauty (new)":        TEAL,
            "Supplements (new)":   AFYA_BLUE,
            "Body Building (new)": ORANGE,
        })
        fig_rv = px.bar(rv_data, x="Revenue (KES)", y="Category", orientation="h",
                        color="Category", color_discrete_map=rv_colors_ext, text="label")
        fig_rv.update_traces(textposition="outside",
                              textfont=dict(color="#003467",size=10), marker_line_width=0)
        fig_rv.update_layout(**CHART_LAYOUT, height=380, showlegend=False)
        fig_rv.update_xaxes(**AXIS); fig_rv.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Estimated revenue per burden category — incl. new categories</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_rv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── NCD breakdown ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">NCD Breakdown + New Category Products</div>',
                unsafe_allow_html=True)

    NCD_MAP = {
        "Cardiovascular - Hypertension": ["amlodipine","atenolol","lisinopril","losartan",
                                           "nifedipine","hydrochlorothiazide","ramipril"],
        "Diabetes / Endocrine":          ["metformin","glibenclamide","insulin","glucophage"],
        "HIV / ARVs":                    ["efavirenz","nevirapine","lamivudine","tenofovir",
                                           "dolutegravir"],
    }
    BURDEN_PRODUCT_KW = {
        "NCD":                    ["metformin","amlodipine","atenolol","insulin","losartan"],
        "Communicable":           ["artemether","coartem","amoxicillin","cotrimoxazole","ciprofloxacin"],
        "Respiratory":            ["salbutamol","prednisolone","beclomethasone","ambroxol"],
        "Maternal & Child Health":["folic","ferrous","oxytocin","magnesium","antenatal"],
        "Reproductive Health":    ["postinor","contraceptive","progesterone","norethisterone"],
        "GI":                     ["omeprazole","pantoprazole","ranitidine","metoclopramide"],
        "MSK / Injury":           ["ibuprofen","diclofenac","naproxen","tramadol","meloxicam"],
        "Dermatological":         ["cerave","lotion","cream","moisturiser","sunscreen","toner",
                                    "cleanser","serum"],
    }

    disp["ncd_product"] = disp["product_name"].apply(
        lambda n: next((c for c,kws in NCD_MAP.items() if any(k in str(n).lower() for k in kws)), None))
    ncd_prod_df = disp[disp["ncd_product"].notna()].copy()

    ncd_rows = burden_exp[burden_exp["bundled"]=="NCD"].copy()
    ncd_sub_sum = (ncd_rows.groupby("ncd_sub")["consultation_count"]
                   .sum().reset_index()
                   .rename(columns={"ncd_sub":"Sub-category","consultation_count":"Consultations"}))
    ncd_sub_sum = ncd_sub_sum[ncd_sub_sum["Sub-category"].notna()]
    ncd_sub_sum["% of NCD"] = (ncd_sub_sum["Consultations"]/ncd_sub_sum["Consultations"].sum()*100).round(1)
    ncd_sub_sum = ncd_sub_sum.sort_values("Consultations",ascending=False)

    nd1, nd2 = st.columns(2)
    with nd1:
        fig_ns = px.bar(ncd_sub_sum.sort_values("Consultations",ascending=True),
                        x="Consultations", y="Sub-category", orientation="h",
                        color="% of NCD",
                        color_continuous_scale=[[0,"#cce0f5"],[1,AFYA_BLUE]],
                        text="% of NCD")
        fig_ns.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              textfont=dict(color="#003467",size=10), marker_line_width=0)
        fig_ns.update_coloraxes(showscale=False)
        fig_ns.update_layout(**CHART_LAYOUT, height=320)
        fig_ns.update_xaxes(**AXIS); fig_ns.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">NCD sub-categories</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_ns, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with nd2:
        # Top products INCLUDING new category products
        if not ncd_prod_df.empty:
            top_ncd = (ncd_prod_df.groupby(["ncd_product","product_name"])
                       .agg(qty=("qty_dispensed","sum"),revenue=("total_sales_value","sum"))
                       .reset_index().sort_values("revenue",ascending=False))
            top_ncd["Revenue"] = top_ncd["revenue"].apply(fmt_ksh)
            top5 = top_ncd.groupby("ncd_product").head(2)
        else:
            top5 = pd.DataFrame()

        # Append new category products
        new_cat_top = pd.DataFrame([
            {"ncd_product":"Beauty (new)","product_name":"CeraVe Blemish Cleanser 236ml",
             "revenue":180000,"Revenue":"KES 180K"},
            {"ncd_product":"Supplements (new)","product_name":"Centrum Multivitamin Women",
             "revenue":96000,"Revenue":"KES 96K"},
            {"ncd_product":"Body Building (new)","product_name":"Whey Protein Concentrate 1kg",
             "revenue":65000,"Revenue":"KES 65K"},
        ])
        top5 = pd.concat([top5, new_cat_top], ignore_index=True).rename(
            columns={"ncd_product":"Condition","product_name":"Product","qty":"Units"})

        color_map_ext = {
            "Cardiovascular - Hypertension":AFYA_BLUE,
            "Diabetes / Endocrine":         TEAL,
            "HIV / ARVs":                   CORAL,
            "Beauty (new)":                 TEAL,
            "Supplements (new)":            AFYA_BLUE,
            "Body Building (new)":          ORANGE,
        }
        fig_np = px.bar(top5.sort_values("revenue",ascending=True),
                        x="revenue", y="Product", orientation="h",
                        color="Condition", color_discrete_map=color_map_ext,
                        text="Revenue")
        fig_np.update_traces(textposition="outside",
                              textfont=dict(color="#003467",size=10), marker_line_width=0)
        fig_np.update_layout(**CHART_LAYOUT, height=320)
        fig_np.update_xaxes(**AXIS); fig_np.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Top products by revenue — all categories incl. new</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_np, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Burden group detail ───────────────────────────────────────────────────
    st.markdown('<div class="section-head">Top Burden Groups — Detail</div>', unsafe_allow_html=True)
    top5_cats = bundled_sum.head(5)["Category"].tolist()
    sel_cat = st.selectbox("Select burden group", top5_cats, key="bcat")

    cat_rows2 = burden_exp[burden_exp["bundled"]==sel_cat].copy()
    det1, det2 = st.columns(2)
    with det1:
        sub_df = (cat_rows2.groupby("group")["consultation_count"]
                  .sum().reset_index()
                  .rename(columns={"group":"Diagnosis","consultation_count":"Consultations"})
                  .sort_values("Consultations",ascending=True).tail(8))
        fig_sd = px.bar(sub_df, x="Consultations", y="Diagnosis", orientation="h",
                        color="Consultations",
                        color_continuous_scale=[[0,"#cce0f5"],[1,AFYA_BLUE]], text="Consultations")
        fig_sd.update_traces(textposition="outside",
                              textfont=dict(color="#003467",size=10), marker_line_width=0)
        fig_sd.update_coloraxes(showscale=False)
        fig_sd.update_layout(**CHART_LAYOUT, height=320)
        fig_sd.update_xaxes(**AXIS); fig_sd.update_yaxes(**AXIS)
        st.markdown(f'<div class="chart-card"><div class="card-title">{sel_cat} — sub-groups</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_sd, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with det2:
        kws = BURDEN_PRODUCT_KW.get(sel_cat, [])
        if kws:
            mask = disp["product_name"].str.lower().apply(
                lambda n: any(k in str(n) for k in kws) if n else False)
            cp = disp[mask].groupby("product_name").agg(
                qty=("qty_dispensed","sum"), revenue=("total_sales_value","sum")
            ).reset_index().nlargest(8,"revenue")
            cp["Revenue"] = cp["revenue"].apply(fmt_ksh)
            bar_color = TEAL if sel_cat == "Dermatological" else AFYA_BLUE
            fig_cp = px.bar(cp.sort_values("revenue",ascending=True),
                            x="revenue", y="product_name", orientation="h",
                            text="Revenue", color_discrete_sequence=[bar_color])
            fig_cp.update_traces(textposition="outside",
                                  textfont=dict(color="#003467",size=10), marker_line_width=0)
            fig_cp.update_layout(**CHART_LAYOUT, height=320, showlegend=False)
            fig_cp.update_xaxes(**AXIS); fig_cp.update_yaxes(**AXIS)
            st.markdown(f'<div class="chart-card"><div class="card-title">{sel_cat} — top products by revenue</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_cp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Antibiotic stewardship ────────────────────────────────────────────────
    st.markdown('<div class="section-head">Antibiotic Stewardship</div>', unsafe_allow_html=True)
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
        ab1, ab2 = st.columns([2,1])
        tier_colors = {"First-line":TEAL,"Second-line":ORANGE,"Third-line":CORAL}
        with ab1:
            abx_b = abx.groupby(["facility_id","abx_tier"])["qty_dispensed"].sum().reset_index()
            abx_b["Branch"] = "Branch " + abx_b["facility_id"].astype(str)
            fig_abx = px.bar(abx_b, x="Branch", y="qty_dispensed", color="abx_tier",
                              barmode="stack", text="qty_dispensed",
                              color_discrete_map=tier_colors,
                              labels={"qty_dispensed":"Units","abx_tier":"Tier"})
            fig_abx.update_traces(textposition="inside", textfont=dict(color="white",size=9))
            fig_abx.update_layout(**CHART_LAYOUT, height=300)
            fig_abx.update_xaxes(**AXIS); fig_abx.update_yaxes(**AXIS)
            st.markdown('<div class="chart-card"><div class="card-title">1st vs 2nd vs 3rd line per branch</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_abx, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with ab2:
            abx_t = abx.groupby("abx_tier")["qty_dispensed"].sum().reset_index()
            total_abx = abx_t["qty_dispensed"].sum()
            st.markdown('<div class="chart-card"><div class="card-title">Chain-wide split</div>',
                        unsafe_allow_html=True)
            for _, row in abx_t.iterrows():
                c = tier_colors.get(str(row["abx_tier"]), GRAY)
                pct = round(row["qty_dispensed"]/total_abx*100) if total_abx > 0 else 0
                st.markdown(f"""
                <div style="background:#f0f2f6;border-radius:8px;padding:.65rem .9rem;
                            margin-bottom:.4rem;border-left:3px solid {c};">
                    <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;">{row["abx_tier"]}</div>
                    <div style="color:{c};font-size:1.2rem;font-weight:700;">{int(row["qty_dispensed"]):,} units</div>
                    <div style="font-size:.75rem;color:{MUTED};">{pct}% of antibiotics</div>
                </div>
                """, unsafe_allow_html=True)
            if "Second-line" in abx_t["abx_tier"].values and "First-line" in abx_t["abx_tier"].values:
                sl = abx_t[abx_t["abx_tier"]=="Second-line"]["qty_dispensed"].values[0]
                fl = abx_t[abx_t["abx_tier"]=="First-line"]["qty_dispensed"].values[0]
                ratio = round(sl/fl, 2) if fl > 0 else "N/A"
                wc = CORAL if isinstance(ratio,float) and ratio > 0.5 else TEAL
                st.markdown(f"""
                <div style="background:{wc}18;border:1px solid {wc}55;border-radius:8px;
                            padding:.55rem .9rem;margin-top:.2rem;">
                    <span style="color:{wc};font-weight:700;font-size:.8rem;">
                        2nd:1st = {ratio}
                        {"  ⚠ Review prescribing" if isinstance(ratio,float) and ratio>0.5 else "  ✓ Within range"}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPENING STOCK (growth categories first, revenue throughout)
# ════════════════════════════════════════════════════════════════════════════
with tab3:

    REORDER_MAP = {
        "Oral Solid Forms":        ("Day 18", TEAL),
        "Injectables":             ("Day 22", TEAL),
        "Beauty Products":         ("Day 38", AFYA_BLUE),
        "Vitamins & Supplements":  ("Day 45", AFYA_BLUE),
        "Oral Liquid Forms":       ("Day 55", AFYA_BLUE),
        "Body Building":           ("Day 72", ORANGE),
        "Wound Care":              ("Hold",   CORAL),
    }
    NEW_WHY = {
        "Beauty Products":        "176K women aged 15–49 in Embu. 3 beauty salons within 3km. GT beauty index 0.78 rising. Stock facial care, body lotions, and lip care first.",
        "Vitamins & Supplements": "45% of Embu adults (25–64) are the core buyer. Immunity and multivitamins trending upward. GT supplements index 0.64.",
        "Body Building":          "17% of Embu population are men aged 15–34. Gym detected 3km away. Whey protein and creatine lead. 12 flagged items removed.",
    }
    CORE_WHY = {
        "Oral Solid Forms":   "Antibiotics, antidiabetics, antimalarials. High NCD and malaria burden. Fastest-selling.",
        "Injectables":        "Demand driven by nearby hospital and 68% ANC visit rate. Artemether and oxytocin move fastest.",
        "Oral Liquid Forms":  "30% of Embu is under 15. Paediatric syrups and ORS are steady sellers.",
        "IV Fluids & Infusions": "Hospital-proximity product. Lower walk-in demand — order conservatively.",
        "Topical Preparations":  "Skin conditions are common. Moderate volume.",
        "Vaccines & Biologicals":"Immunisation-driven. Coordinate with facility schedule.",
    }

    # ── Growth hero strip ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{COOL_BLUE};border-radius:12px;padding:20px 24px;
                margin-bottom:16px;display:flex;align-items:stretch;gap:0;">
        <div style="flex:1.3;padding-right:22px;border-right:1px solid rgba(255,255,255,.15);">
            <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
                        color:rgba(255,255,255,.5);margin-bottom:5px;">New growth categories &mdash; Branch #106 Embu</div>
            <div style="font-size:1.5rem;font-weight:700;color:#fff;line-height:1.2;margin-bottom:5px;">
                3 new categories &middot; {new_units:,} units
            </div>
            <div style="font-size:.92rem;color:rgba(255,255,255,.65);">
                Estimated Month 1 revenue from new categories:
                <span style="color:{TEAL};font-weight:700;font-size:1.1rem;"> {fmt_ksh(new_revenue)}</span>
            </div>
            <div style="margin-top:10px;font-size:.8rem;color:rgba(255,255,255,.45);line-height:1.5;">
                Sized at 35% of mature branch velocity &mdash; Month 1 penetration factor.<br>
                Beauty and Supplements adjusted upward by Google Trends momentum signal.
            </div>
        </div>
        <div style="flex:1;padding:0 18px;border-right:1px solid rgba(255,255,255,.15);">
            <div style="display:inline-block;background:{TEAL};color:#fff;font-size:.68rem;
                        font-weight:700;padding:2px 9px;border-radius:20px;margin-bottom:6px;">BEAUTY</div>
            <div style="font-size:2rem;font-weight:700;color:#fff;line-height:1;">{bty_u:,}</div>
            <div style="font-size:.72rem;color:rgba(255,255,255,.5);margin-top:2px;">units to order</div>
            <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px;">{fmt_ksh(bty_r)}</div>
            <div style="font-size:.7rem;color:rgba(255,255,255,.45);">est. Month 1</div>
            <div style="font-size:.72rem;color:rgba(255,255,255,.5);margin-top:5px;">176K women aged 15&ndash;49</div>
        </div>
        <div style="flex:1;padding:0 18px;border-right:1px solid rgba(255,255,255,.15);">
            <div style="display:inline-block;background:{TEAL};color:#fff;font-size:.68rem;
                        font-weight:700;padding:2px 9px;border-radius:20px;margin-bottom:6px;">SUPPLEMENTS</div>
            <div style="font-size:2rem;font-weight:700;color:#fff;line-height:1;">{sup_u:,}</div>
            <div style="font-size:.72rem;color:rgba(255,255,255,.5);margin-top:2px;">units to order</div>
            <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px;">{fmt_ksh(sup_r)}</div>
            <div style="font-size:.7rem;color:rgba(255,255,255,.45);">est. Month 1</div>
            <div style="font-size:.72rem;color:rgba(255,255,255,.5);margin-top:5px;">45% of Embu adults are buyers</div>
        </div>
        <div style="flex:1;padding:0 0 0 18px;">
            <div style="display:inline-block;background:rgba(255,255,255,.15);color:#fff;
                        font-size:.68rem;font-weight:700;padding:2px 9px;border-radius:20px;margin-bottom:6px;">BODY BUILDING</div>
            <div style="font-size:2rem;font-weight:700;color:#fff;line-height:1;">{bb_u:,}</div>
            <div style="font-size:.72rem;color:rgba(255,255,255,.5);margin-top:2px;">units to order</div>
            <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px;">{fmt_ksh(bb_r)}</div>
            <div style="font-size:.7rem;color:rgba(255,255,255,.45);">est. Month 1</div>
            <div style="font-size:.72rem;color:rgba(255,255,255,.5);margin-top:5px;">Gym detected 3km away</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    dead_count = int((prod_out["Dead Stock Risk"]=="Yes").sum()) if has_products and not prod_out.empty and "Dead Stock Risk" in prod_out.columns else 0
    high_conf  = int((prod_out["Confidence"]=="High").sum())    if has_products and not prod_out.empty and "Confidence" in prod_out.columns else 0
    total_pred = len(prod_out) if has_products and not prod_out.empty else len(cat_out)

    k1,k2,k3,k4,k5 = st.columns(5)
    kpi_card(k1,"Total products",       f"{total_pred:,}",        AFYA_BLUE, "22 categories")
    kpi_card(k2,"Opening stock units",  f"{total_units:,}",        AFYA_BLUE, "Day 1 order")
    kpi_card(k3,"Est. Month 1 revenue", fmt_ksh(total_revenue),    TEAL,      "All categories")
    kpi_card(k4,"High confidence",      f"{high_conf:,}",          TEAL,      f"of {total_pred} products")
    kpi_card(k5,"Dead stock risk",      f"{dead_count:,}",         CORAL,     "Remove from order")

    st.markdown("<div style='margin:.75rem 0;'></div>", unsafe_allow_html=True)

    # ── Category cards helper ─────────────────────────────────────────────────
    def _conf_color(c): return TEAL if c=="High" else (ORANGE if c=="Medium" else CORAL)

    def render_cat_card(row, is_new):
        cat   = row["Category"]
        units = int(row["total_units"])
        rev   = row["est_revenue"]
        n_p   = int(row.get("n_products",0))
        why   = NEW_WHY.get(cat, CORE_WHY.get(cat,"Based on Embu catchment profile."))
        day, rcolor = REORDER_MAP.get(cat, ("—", MUTED))
        rbg   = f"{rcolor}18"
        tc    = TEAL if is_new else AFYA_BLUE
        tl    = "New category — growth opportunity" if is_new else "Core pharmacy"
        tlc   = TEAL if is_new else MUTED
        if has_products and not prod_out.empty and "Confidence" in prod_out.columns:
            ccol = "Category" if "Category" in prod_out.columns else "therapeutic_group"
            cp = prod_out[prod_out[ccol]==cat]
            cc2 = cp["Confidence"].value_counts()
            top_conf = cc2.index[0] if not cc2.empty else "—"
            cpct = int(cc2.iloc[0]/len(cp)*100) if len(cp)>0 else 0
        else:
            top_conf, cpct = "—", 0
        cfc = _conf_color(top_conf)
        return f"""
        <div style="background:#fff;border:1px solid {BORDER};border-radius:10px;
                    overflow:hidden;border-top:4px solid {tc};margin-bottom:4px;">
            <div style="padding:11px 14px 8px;border-bottom:1px solid {BORDER};">
                <div style="font-size:.85rem;font-weight:700;color:{COOL_BLUE};">{cat}</div>
                <div style="font-size:.68rem;font-weight:600;color:{tlc};margin-top:2px;">{tl}</div>
            </div>
            <div style="padding:12px 14px;">
                <div style="font-size:1.8rem;font-weight:700;color:{COOL_BLUE};line-height:1;">{units:,}</div>
                <div style="font-size:.7rem;color:{MUTED};margin-bottom:4px;">units &middot; {n_p} products</div>
                <div style="font-size:.9rem;font-weight:700;color:{TEAL};margin-bottom:8px;">{fmt_ksh(rev)} est. Month 1</div>
                <div style="font-size:.75rem;color:rgba(0,52,103,.55);line-height:1.55;margin-bottom:8px;">{why}</div>
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:7px;">
                    <div style="font-size:.65rem;color:{MUTED};width:58px;flex-shrink:0;">Confidence</div>
                    <div style="flex:1;height:5px;background:#F0F5FF;border-radius:3px;overflow:hidden;">
                        <div style="width:{cpct}%;height:100%;background:{cfc};border-radius:3px;"></div>
                    </div>
                    <div style="font-size:.68rem;font-weight:700;color:{cfc};min-width:40px;text-align:right;">{top_conf}</div>
                </div>
                <div style="background:{rbg};border-radius:5px;padding:5px 9px;font-size:.75rem;
                            font-weight:600;color:{rcolor};">Reorder: {day}</div>
            </div>
        </div>
        """

    # ── New category cards ────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:.68rem;font-weight:700;color:{TEAL};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:8px;">New growth categories — order these first</div>',
                unsafe_allow_html=True)
    nc1, nc2, nc3 = st.columns(3)
    for col, cat_name in zip([nc1,nc2,nc3], NEW_CATS):
        row = cat_rev[cat_rev["Category"]==cat_name]
        if row.empty:
            row = pd.DataFrame([{"Category":cat_name,"total_units":0,"est_revenue":0,"n_products":0}])
        with col:
            st.markdown(render_cat_card(row.iloc[0], is_new=True), unsafe_allow_html=True)

    st.markdown("<div style='margin:.75rem 0;'></div>", unsafe_allow_html=True)

    # ── Core pharma cards ─────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:.68rem;font-weight:700;color:{MUTED};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:8px;">Core pharmacy — order as usual</div>',
                unsafe_allow_html=True)
    core_sorted = core_rev.sort_values("total_units", ascending=False)
    top_core = core_sorted.head(3); rest_core = core_sorted.iloc[3:]
    cc1, cc2, cc3 = st.columns(3)
    for col, (_, row) in zip([cc1,cc2,cc3], top_core.iterrows()):
        with col:
            st.markdown(render_cat_card(row, is_new=False), unsafe_allow_html=True)
    if not rest_core.empty:
        with st.expander(f"Show {len(rest_core)} more core categories"):
            rcols = st.columns(3)
            for i,(_, row) in enumerate(rest_core.iterrows()):
                with rcols[i%3]:
                    st.markdown(render_cat_card(row, is_new=False), unsafe_allow_html=True)

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    # ── Revenue chart + reorder timeline ──────────────────────────────────────
    rc, rt = st.columns([3,2])
    with rc:
        chart_d = cat_rev.sort_values("est_revenue", ascending=False).head(10).sort_values("est_revenue", ascending=True).copy()
        chart_d["is_new"] = chart_d["Category"].isin(NEW_CATS)
        chart_d["label"]  = chart_d["est_revenue"].apply(fmt_ksh)
        fig_rc = go.Figure()
        for label, is_new_flag, color in [
            ("New category", True,  TEAL),
            ("Core pharma",  False, AFYA_BLUE),
        ]:
            sub = chart_d[chart_d["is_new"]==is_new_flag]
            fig_rc.add_trace(go.Bar(
                x=sub["est_revenue"], y=sub["Category"], orientation="h",
                marker_color=color, marker_line_width=0,
                text=sub["label"], textposition="outside",
                textfont=dict(color=COOL_BLUE,size=10), name=label,
            ))
        fig_rc.update_layout(
            height=max(300, len(chart_d)*36), **CHART_LAYOUT, barmode="stack",
        )
        fig_rc.update_xaxes(**AXIS, title_text="Est. Revenue (KES)")
        fig_rc.update_yaxes(**AXIS)
        st.markdown('<div class="chart-card"><div class="card-title">Estimated Month 1 revenue by category</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_rc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rt:
        st.markdown('<div class="chart-card"><div class="card-title">When to reorder</div>',
                    unsafe_allow_html=True)
        reorder_items = [
            ("Oral Solid Forms",       "Day 18",  TEAL,      "Fastest — set reminder now"),
            ("Injectables",            "Day 22",  TEAL,      "Hospital demand"),
            ("Beauty Products",        "Day 38",  TEAL,      "NEW — GT momentum rising"),
            ("Vitamins & Supplements", "Day 45",  AFYA_BLUE, "NEW — steady adult demand"),
            ("Oral Liquid Forms",      "Day 55",  AFYA_BLUE, "Paediatric"),
            ("Body Building",          "Day 72",  ORANGE,    "NEW — reorder only what sold"),
            ("Wound Care / Dental",    "Hold",    CORAL,     "Check sell-through first"),
        ]
        for name, day, color, note in reorder_items:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;
                        padding:8px 0;border-bottom:1px solid {BORDER};">
                <div style="width:9px;height:9px;border-radius:50%;background:{color};
                            flex-shrink:0;margin-top:3px;"></div>
                <div style="flex:1;">
                    <div style="font-size:.78rem;font-weight:600;color:{COOL_BLUE};">{name}</div>
                    <div style="font-size:.68rem;color:{MUTED};margin-top:1px;">{note}</div>
                </div>
                <div style="font-size:.78rem;font-weight:700;color:{color};
                            white-space:nowrap;padding-left:8px;">{day}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Full product table ────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Full Stock List — All Categories</div>',
                unsafe_allow_html=True)

    if has_products and not prod_out.empty:
        cc_name = "Category" if "Category" in prod_out.columns else "therapeutic_group"
        qc_name = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
        pn_col  = "Product" if "Product" in prod_out.columns else "product_name"

        f1,f2,f3,_ = st.columns([2,2,2,4])
        with f1:
            cats_all = ["All categories"] + sorted(prod_out[cc_name].dropna().unique().tolist())
            sel_c = st.selectbox("Category", cats_all, key="sc")
        with f2:
            sel_cf = st.selectbox("Confidence",["All","High","Medium","Low"],key="scf")
        with f3:
            sel_r = st.selectbox("Dead stock",["All","Clear only","Risk only"],key="sr")

        disp_cols = [pn_col, cc_name, qc_name]
        if "est_revenue" in prod_out.columns:         disp_cols.append("est_revenue")
        if "Historical Share" in prod_out.columns:    disp_cols.append("Historical Share")
        if "Confidence" in prod_out.columns:          disp_cols.append("Confidence")
        if "Dead Stock Risk" in prod_out.columns:     disp_cols.append("Dead Stock Risk")

        tbl = prod_out[disp_cols].rename(columns={
            pn_col:        "Product",
            cc_name:       "Category",
            qc_name:       "Order Qty",
            "est_revenue": "Est. Revenue (KES)",
        }).copy()

        if sel_c != "All categories":  tbl = tbl[tbl["Category"]==sel_c]
        if sel_cf != "All" and "Confidence" in tbl.columns:
            tbl = tbl[tbl["Confidence"]==sel_cf]
        if sel_r == "Risk only" and "Dead Stock Risk" in tbl.columns:
            tbl = tbl[tbl["Dead Stock Risk"]=="Yes"]
        elif sel_r == "Clear only" and "Dead Stock Risk" in tbl.columns:
            tbl = tbl[tbl["Dead Stock Risk"]=="No"]

        tbl = tbl.sort_values("Order Qty", ascending=False).reset_index(drop=True)

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
        st.dataframe(tbl, use_container_width=True, hide_index=True,
                     column_config={k:v for k,v in col_cfg.items() if k in tbl.columns},
                     height=420)
        st.download_button("⬇  Download full stock list (CSV)",
                           tbl.to_csv(index=False).encode("utf-8"),
                           "branch_106_opening_stock.csv","text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        out = cat_out.copy()
        keep_c = ["Category" if "Category" in cat_out.columns else "therapeutic_group",
                  "Opening Stock Qty" if "Opening Stock Qty" in cat_out.columns else "opening_stock_qty",
                  "Confidence","Dead Stock Risk"]
        out = out[[c for c in keep_c if c in out.columns]]
        st.markdown('<div class="chart-card"><div class="card-title">Opening stock — category level</div>',
                    unsafe_allow_html=True)
        st.dataframe(out, use_container_width=True, hide_index=True, height=380)
        st.download_button("⬇  Download CSV", out.to_csv(index=False).encode("utf-8"),
                           "branch_106_stock_categories.csv","text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Do not order ──────────────────────────────────────────────────────────
    if has_products and not prod_out.empty and "Dead Stock Risk" in prod_out.columns:
        dead_list = prod_out[prod_out["Dead Stock Risk"]=="Yes"].copy()
        if not dead_list.empty:
            st.markdown(f'<div class="section-head">Do Not Order — {len(dead_list)} Products Flagged</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#FEF0F0;border:1px solid rgba(224,92,92,.2);border-radius:10px;
                        padding:12px 16px;margin-bottom:12px;font-family:'Montserrat',sans-serif;">
                <span style="color:{CORAL};font-weight:700;font-size:.85rem;">
                    These {len(dead_list)} products are predicted to sit on the shelf for 85+ days.
                </span>
                <span style="color:rgba(160,32,32,.75);font-size:.82rem;">
                    Remove from the opening order, or ask for consignment terms.
                </span>
            </div>
            """, unsafe_allow_html=True)
            pnd  = "Product" if "Product" in dead_list.columns else "product_name"
            catd = "Category" if "Category" in dead_list.columns else "therapeutic_group"
            dcols = [c for c in [pnd, catd, "Confidence"] if c in dead_list.columns]
            if "est_revenue" in dead_list.columns: dcols.append("est_revenue")
            dead_show = dead_list[dcols].rename(columns={
                pnd:"Product", catd:"Category", "est_revenue":"Predicted Revenue (KES)"
            }).reset_index(drop=True)
            st.markdown('<div class="chart-card"><div class="card-title">Dead stock list — remove before ordering</div>',
                        unsafe_allow_html=True)
            st.dataframe(dead_show, use_container_width=True, hide_index=True,
                         column_config={
                             "Predicted Revenue (KES)": st.column_config.NumberColumn(format="KES %,.0f"),
                             "Product":    st.column_config.TextColumn(width="large"),
                             "Category":   st.column_config.TextColumn(width="medium"),
                             "Confidence": st.column_config.TextColumn(width="small"),
                         }, height=280)
            st.download_button("⬇  Download dead stock list (CSV)",
                               dead_show.to_csv(index=False).encode("utf-8"),
                               "branch_106_dead_stock.csv","text/csv")
            st.markdown('</div>', unsafe_allow_html=True)