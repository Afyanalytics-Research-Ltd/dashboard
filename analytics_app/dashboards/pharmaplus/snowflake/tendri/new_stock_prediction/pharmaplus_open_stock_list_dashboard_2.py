"""
Pharmaplus · Afya Analytics Platform
New Venture Headstart — Peri-Urban Branch Intelligence
======================================================
Tab 1: The Full Story  (Problem → Market → Stock List → Value)
Tab 2: Deep Dive       (Spend Behaviour · Generic/Branded · Disease Burden · Scenario)

Run:  streamlit run pharmaplus_dashboard_v3.py
Data: data_export.pkl from pharmaplus_model notebook
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image

# ── GLOBAL PLOTLY TEMPLATE ────────────────────────────────────────────────────
pio.templates["afya"] = pio.templates["plotly_white"]
pio.templates["afya"].layout.font        = dict(family="Montserrat, sans-serif", color="#0072CE", size=11)
pio.templates["afya"].layout.legend.font = dict(family="Montserrat, sans-serif", color="#0072CE", size=10)
pio.templates["afya"].layout.xaxis.tickfont   = dict(color="#0072CE", size=10)
pio.templates["afya"].layout.xaxis.title.font = dict(color="#0072CE", size=11)
pio.templates["afya"].layout.yaxis.tickfont   = dict(color="#0072CE", size=10)
pio.templates["afya"].layout.yaxis.title.font = dict(color="#0072CE", size=11)
pio.templates["afya"].layout.xaxis.gridcolor  = "#EBF3FB"
pio.templates["afya"].layout.yaxis.gridcolor  = "#EBF3FB"
pio.templates["afya"].layout.paper_bgcolor    = "#fff"
pio.templates["afya"].layout.plot_bgcolor     = "#fff"
pio.templates.default = "afya"

# ── THERAPEUTIC GROUP MAP ─────────────────────────────────────────────────────
THERAPEUTIC_GROUP_MAP = {
    "Analgesic / antipyretic":"Analgesics","NSAID":"Analgesics","NSAID + analgesic":"Analgesics",
    "NSAID + analgesic combination":"Analgesics","NSAID — COX-2 selective":"Analgesics",
    "Topical analgesic":"Analgesics","Non-opioid analgesic":"Analgesics",
    "Opioid analgesic (mild)":"Analgesics","Muscle relaxant":"Analgesics",
    "Antibiotic / antiprotozoal":"Antibiotics","Antibiotic — aminoglycoside":"Antibiotics",
    "Antibiotic — aminopenicillin":"Antibiotics","Antibiotic — beta-lactam combination":"Antibiotics",
    "Antibiotic — cephalosporin (1st gen)":"Antibiotics","Antibiotic — cephalosporin (2nd gen)":"Antibiotics",
    "Antibiotic — cephalosporin (3rd gen)":"Antibiotics","Antibiotic — fluoroquinolone":"Antibiotics",
    "Antibiotic — macrolide":"Antibiotics","Antibiotic — tetracycline":"Antibiotics",
    "Antibiotic — topical":"Antibiotics","Antimicrobial — topical":"Antibiotics",
    "Antihypertensive — ACE + diuretic":"Antihypertensives","Antihypertensive — ARB":"Antihypertensives",
    "Antihypertensive — CCB":"Antihypertensives","Antihypertensive — beta-blocker":"Antihypertensives",
    "Antihypertensive — diuretic":"Antihypertensives","Beta-blocker":"Antihypertensives",
    "Diuretic — loop":"Antihypertensives","Antiarrhythmic":"Antihypertensives",
    "Biguanide — antidiabetic":"Antidiabetics","Sulfonylurea — antidiabetic":"Antidiabetics",
    "DPP-4 inhibitor":"Antidiabetics","Insulin":"Antidiabetics",
    "Antimalarial":"Antimalarials","Antimalarial — ACT":"Antimalarials","Antimalarial — SP":"Antimalarials",
    "Antifungal — azole":"Antivirals & Antifungals","Antifungal — topical":"Antivirals & Antifungals",
    "Anthelmintic":"Antivirals & Antifungals",
    "Proton pump inhibitor (PPI)":"GI Agents","H2 antagonist":"GI Agents","Antacid":"GI Agents",
    "Antispasmodic":"GI Agents","Antiemetic — dopamine antagonist":"GI Agents",
    "Antidiarrhoeal":"GI Agents","Laxative — stimulant":"GI Agents",
    "Bronchodilator — SABA":"Respiratory","LABA + ICS combination":"Respiratory",
    "Expectorant":"Respiratory","Mucolytic":"Respiratory","Cough preparation":"Respiratory",
    "Nasal decongestant":"Respiratory",
    "SSRI antidepressant":"CNS & Mental Health","Antipsychotic":"CNS & Mental Health",
    "Anticonvulsant":"CNS & Mental Health","Benzodiazepine":"CNS & Mental Health",
    "Combined oral contraceptive":"Hormones & Contraceptives",
    "Emergency contraceptive":"Hormones & Contraceptives",
    "Progestogen":"Hormones & Contraceptives","Thyroid hormone":"Hormones & Contraceptives",
    "Corticosteroid — oral":"Corticosteroids","Corticosteroid — injectable":"Corticosteroids",
    "Corticosteroid — topical":"Dermatologicals","Emollient":"Dermatologicals",
    "Retinoid — topical":"Dermatologicals","Wound healing — topical":"Dermatologicals",
    "Ophthalmic — lubricant":"Ophthalmics","Ophthalmic — antibiotic":"Ophthalmics",
    "Ophthalmic — glaucoma":"Ophthalmics",
    "Calcium supplement":"Vitamins & Supplements","Iron supplement":"Vitamins & Supplements",
    "Vitamin B complex":"Vitamins & Supplements","Vitamin supplement":"Vitamins & Supplements",
    "Vitamins & Supplements":"Vitamins & Supplements","Multivitamin supplement":"Vitamins & Supplements",
    "Probiotic":"Vitamins & Supplements","Nutritional supplement":"Vitamins & Supplements",
    "Prenatal vitamin supplement":"Vitamins & Supplements",
    "Anticoagulant":"Anticoagulants","Antiplatelet":"Anticoagulants",
    "Local anaesthetic":"Anaesthetics",
    "PDE5 inhibitor":"Urology","Alpha-blocker — uroselective":"Urology",
    "IV fluid — carbohydrate":"IV & Hospital Fluids","Vasopressor / inotrope":"IV & Hospital Fluids",
    "Statin":"Statins & Lipid",
    "Beauty Products":"Beauty Products","Body Building":"Body Building",
}

# ── GENERIC DETECTION ─────────────────────────────────────────────────────────
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
    "garnier","maybelline","revlon","loreal","pantene","dove","sunsilk","the ordinary",
    "optimum nutrition","usn","bsn","muscletech","evox","dymatize","myprotein",
]

def infer_generic(product_name):
    name = str(product_name).lower()
    if any(b in name for b in KNOWN_BRANDS): return False
    if any(g in name for g in GENERIC_TERMS): return True
    return None

# ── SUB-CATEGORY CLASSIFICATION — ALL NEW CATEGORIES ─────────────────────────

BEAUTY_SUBCAT_MAP = [
    ("Skincare",   ["sunscreen","spf","serum","toner","moistur","cleanser","face wash",
                    "exfoliat","retinol","niacinamide","salicylic","benzoyl","acne","blemish",
                    "face cream","face mask","eye cream","vitamin c","hyaluronic","la roche",
                    "the ordinary","cosrx","olay","cerave","neutrogena","bioderma","eucerin",
                    "nivea face","loreal face","face sunscreen","daily sunscreen","tinted sunscreen",
                    "mineral sunscreen","uv protection","sunblock","spf 50","body sunscreen"]),
    ("Body Care",  ["body lotion","body butter","body wash","petroleum jelly","vaseline",
                    "aloe vera","calamine","hand cream","hand lotion","glycerin",
                    "cocoa butter","shea butter","nivea body","dove body","baby oil","eos"]),
    ("Hair Care",  ["shampoo","conditioner","hair oil","hair mask","hair growth","hair relaxer",
                    "hair dye","hair colour","cantu","dark and lovely","ors hair","pantene",
                    "tresemme","schwarzkopf","sunsilk","dove hair","hair serum","edge control",
                    "heat protectant","leave in","deep conditioner","anti dandruff","natural hair"]),
    ("Makeup",     ["foundation","bb cream","cc cream","concealer","mascara","eyeliner",
                    "eyeshadow","blush","highlighter","setting powder","setting spray",
                    "lipstick","lip gloss","lip liner","contour","bronzer","maybelline",
                    "revlon","black opal","wet n wild","nyx","mac cosmetics","rimmel",
                    "sleek","milani","catrice","essence","flormar","fenty","nars",
                    "charlotte tilbury","urban decay","too faced"]),
    ("Fragrance",  ["perfume","cologne","fragrance","body spray","deodorant","roll-on",
                    "antiperspirant","eau de","edp","edt"]),
]

SUPP_SUBCAT_MAP = [
    ("Immune & Vitamin C",  ["vitamin c","zinc","immune booster","effervescent","ester-c",
                              "redoxon","berocca","emergen-c","elderberry","zinnat",
                              "vitamin c tablet","vitamin c 1000","vitamin c powder"]),
    ("Multivitamins",       ["multivitamin","prenatal","wellwoman","wellman","centrum",
                              "seven seas","supradyn","vitabiotics","abidec","blackmores multi",
                              "solgar multi","nature made","garden of life","kirkland vitamin",
                              "gummy vitamin","daily vitamin","senior multi","mens multi",
                              "womens multi","children multi"]),
    ("Collagen & Beauty",   ["collagen","biotin","marine collagen","collagen powder","hair skin nail",
                              "beauty vitamin","keratin","perfectil","nourkrin","neocell",
                              "vital protein","sports research collagen","ancient nutrition",
                              "zeta white","biotin 10000"]),
    ("Bone & Heart",        ["vitamin d","calcium","omega","fish oil","cod liver","bone supplement",
                              "heart health","calcium d3","magnesium","vitamin d3","calcium magnesium",
                              "seven seas cod","omega h3","calcichew","caltrate","cardiowell",
                              "nordic naturals","blackmores fish","solgar omega","nature made fish"]),
    ("Energy & Stress",     ["b complex","vitamin b12","magnesium","energy supplement","stress",
                              "neurobion","becosules","slow-mag","magnesium b6","berocca performance",
                              "adaptogen","ashwagandha","fatigue","b12 injection","b12 tablet"]),
]

BB_SUBCAT_MAP = [
    ("Protein",     ["whey protein","protein powder","isolate protein","plant protein",
                     "protein shake","casein protein","egg white protein","protein bar",
                     "whey concentrate","vegan protein","optimum nutrition","usn protein",
                     "bsn syntha","muscletech","evox protein","dymatize","isopure",
                     "myprotein","musclemeds","rule 1 protein"]),
    ("Mass Gainer", ["mass gainer","weight gainer","bulk supplement","high calorie",
                     "hardgainer","lean mass gainer","3000 calorie","serious mass",
                     "usn muscle fuel","dymatize super mass","mutant mass","evox mass",
                     "optimum nutrition mass","bsn true mass","naked mass"]),
    ("Creatine",    ["creatine","creatine monohydrate","creatine powder","creatine supplement",
                     "creapure","creatine hcl","creatine capsule","creatine loading",
                     "creatine for women","gym creatine","usn creatine","muscletech creatine",
                     "kaged creatine","bulk powders creatine"]),
    ("Pre-Workout", ["pre workout","pre-workout","energy booster gym","caffeine pre",
                     "beta alanine","nitric oxide","citrulline","pump supplement",
                     "c4 pre workout","no xplode","ghost pre","usn 3xt","gorilla mode",
                     "total war","bucked up","kaged pre"]),
    ("Amino Acids", ["bcaa","eaa supplement","amino acid","glutamine","post workout",
                     "recovery supplement","intra workout","electrolyte",
                     "scivation xtend","optimum bcaa","usn bcaa","evox bcaa",
                     "kaged bcaa","cellucor bcaa","myprotein bcaa","bsn amino"]),
]

def classify_beauty_subcat(product_name):
    name = str(product_name).lower()
    for subcat, keywords in BEAUTY_SUBCAT_MAP:
        if any(kw in name for kw in keywords):
            return subcat
    return "Skincare"

def classify_supp_subcat(product_name):
    name = str(product_name).lower()
    for subcat, keywords in SUPP_SUBCAT_MAP:
        if any(kw in name for kw in keywords):
            return subcat
    return "Multivitamins"

def classify_bb_subcat(product_name):
    name = str(product_name).lower()
    for subcat, keywords in BB_SUBCAT_MAP:
        if any(kw in name for kw in keywords):
            return subcat
    return "Protein"

# ── PATHS ─────────────────────────────────────────────────────────────────────
_BASE   = os.path.dirname(os.path.abspath(__file__))   # .../tendri/new_stock_prediction/
_TENDRI = os.path.dirname(_BASE)                        # .../tendri/

PKL_PATH         = os.path.join(_TENDRI, "pickle_file", "data_export.pkl")
PROX_CSV         = os.path.join(_TENDRI, "data", "pharmaplus_proximity_data.csv")

# GT CSVs — match latest file with that prefix in case date suffix changes
def _find_latest(folder, prefix):
    matches = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".csv")]
    return os.path.join(folder, sorted(matches)[-1]) if matches else ""

_DATA_DIR        = os.path.join(_TENDRI, "data")
GT_INDEX_PATH    = _find_latest(_DATA_DIR, "embu_trends_category_index")
GT_FEATURES_PATH = _find_latest(_DATA_DIR, "embu_google_trends_features")

# Logo
LOGO_PATH = None
for _n in ["pharmaplus_logo.jpg", "pharmaplus_logo.png", "afya_logo.png"]:
    _p = os.path.join(_TENDRI, "images", _n)
    if os.path.exists(_p):
        LOGO_PATH = _p
        break

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
_BASE     = os.path.dirname(os.path.abspath(__file__))

# Look in images/ subfolder first, then next to the script
logo_img = Image.open(LOGO_PATH) if LOGO_PATH else None

st.set_page_config(
    page_title="Afya Analytics · New Venture Headstart",
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
BORDER    = "#D6E4F0"
BG_LIGHT  = "#F4F8FC"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, ORANGE, CORAL, PURPLE]
NEW_CATS  = ["Beauty Products","Vitamins & Supplements","Body Building"]
NEW_COLOR = {"Beauty Products":TEAL,"Vitamins & Supplements":AFYA_BLUE,"Body Building":ORANGE}
NEW_ICON  = {"Beauty Products":"✨","Vitamins & Supplements":"💊","Body Building":"💪"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
:root{color-scheme:light only!important}
html,body,[class*="css"],[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:#fff!important;color:#0072CE!important;
  font-family:'Montserrat',sans-serif!important;color-scheme:light!important}
[data-testid="stSidebar"]{background:#003467!important;border-right:none!important}
[data-testid="stSidebar"] *{color:#fff!important}
[data-testid="collapsedControl"]{background:#0072CE!important;border-radius:0 8px 8px 0!important;display:flex!important;visibility:visible!important;opacity:1!important}
[data-testid="collapsedControl"] svg{stroke:#fff!important;fill:#fff!important}
[data-testid="collapsedControl"] button{background:transparent!important;border:none!important}
[data-testid="collapsedControl"]{background:#0072CE!important;border-radius:0 8px 8px 0!important;border:none!important}
[data-testid="collapsedControl"] svg{stroke:#fff!important;fill:#fff!important;color:#fff!important}
button[data-testid="collapsedControl"]{visibility:visible!important;opacity:1!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#F4F8FC;border-radius:10px;padding:4px;
  border:1px solid #D6E4F0;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#0072CE;
  font-size:.83rem;font-weight:700;padding:.4rem 1.4rem;border:none;
  font-family:'Montserrat',sans-serif}
.stTabs [aria-selected="true"]{background:#0072CE!important;color:#fff!important}

/* Cards */
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
.problem-card{background:#003467;border-radius:14px;padding:28px 32px;margin-bottom:20px;color:#fff}
.signal-pill{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:.65rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
.value-card{border-radius:10px;padding:18px 20px;border:1px solid #D6E4F0}

/* Widget labels */
[data-testid="stSlider"] label p,[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label p,[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label p,[data-testid="stTextInput"] label,
[data-testid="stToggle"] label p,[data-testid="stToggle"] label,
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"],
div[class*="stSlider"] p,div[class*="stSelectbox"] p{
  color:#0072CE!important;font-family:'Montserrat',sans-serif!important;font-weight:600!important}

/* Slider thumb */
[data-testid="stSlider"] div[role="slider"]{
  background-color:#0072CE!important;border-color:#0072CE!important}
[data-testid="stSlider"] p,[data-testid="stSlider"] span{
  color:#0072CE!important;font-family:'Montserrat',sans-serif!important;font-weight:600!important}
input[type="range"]::-webkit-slider-thumb{
  background:#0072CE!important;border-color:#0072CE!important;-webkit-appearance:none!important}
input[type="range"]::-moz-range-thumb{background:#0072CE!important;border-color:#0072CE!important}

/* Toggle */
[data-testid="stToggle"] input:checked + div{background-color:#0072CE!important}

/* Expander — flex layout prevents arrow/text overlap */
[data-testid="stExpander"] details > summary{
  display:flex!important;flex-direction:row!important;align-items:center!important;
  justify-content:space-between!important;padding:.6rem .8rem!important;
  cursor:pointer!important;list-style:none!important}
[data-testid="stExpander"] details > summary > div{flex:1!important}
[data-testid="stExpander"] details > summary > div p{
  color:#0072CE!important;font-family:'Montserrat',sans-serif!important;
  font-weight:700!important;font-size:.85rem!important;margin:0!important}
[data-testid="stExpander"] details > summary > svg{
  flex-shrink:0!important;width:18px!important;height:18px!important;
  color:#0072CE!important;fill:#0072CE!important}
[data-testid="stExpander"] details > summary::-webkit-details-marker{display:none!important}

/* Dataframe headers */
[data-testid="stDataFrame"] th{
  color:#0072CE!important;font-family:'Montserrat',sans-serif!important;font-weight:700!important}

#MainMenu,footer,header{visibility:hidden}
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
    colorway=[TEAL,AFYA_BLUE,COOL_BLUE,ORANGE,CORAL,PURPLE],
)
AXIS = dict(
    showgrid=True,gridcolor="#EBF3FB",zeroline=False,color="#0072CE",
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

def signal_badge(label, color):
    return (f'<span class="signal-pill" '
            f'style="background:{color}20;color:{color};border:1px solid {color}40">'
            f'{label}</span>')

# ── DATA PATHS ────────────────────────────────────────────────────────────────
# All paths set via inline expander on the main page — no sidebar needed
_prox_candidates = [
    os.path.join(os.path.dirname(__file__), "pharmaplus_proximity_data.csv"),
    r"C:/Users/Mercy/Documents/Tendri/Snowflake Pulls/Xana/snowflake/tendri/data/pharmaplus_proximity_data.csv",
]
PROX_CSV             = next((p for p in _prox_candidates if os.path.exists(p)), _prox_candidates[0])

_gt_data_dir = r"C:/Users/Mercy/Documents/Tendri/Snowflake Pulls/Xana/snowflake/tendri/data"
_gt_idx_candidates = [
    os.path.join(os.path.dirname(__file__), "embu_trends_category_index_2026-04-09.csv"),
    os.path.join(_gt_data_dir, "embu_trends_category_index_2026-04-09.csv"),
]
_gt_feat_candidates = [
    os.path.join(os.path.dirname(__file__), "embu_google_trends_features_2026-04-09.csv"),
    os.path.join(_gt_data_dir, "embu_google_trends_features_2026-04-09.csv"),
]


# ── SESSION STATE INIT (must happen before sidebar renders) ───────────────────
if "pkl_path"  not in st.session_state: st.session_state.pkl_path  = PKL_PATH
if "prox_path" not in st.session_state: st.session_state.prox_path = PROX_CSV
if "gt_idx"    not in st.session_state: st.session_state.gt_idx    = GT_INDEX_PATH
if "gt_feat"   not in st.session_state: st.session_state.gt_feat   = GT_FEATURES_PATH

# ── SIDEBAR — DATA PATHS (upper left, always accessible) ─────────────────────
with st.sidebar:
    with st.expander("⚙️ Data paths", expanded=not os.path.exists(st.session_state.pkl_path)):
        st.session_state.pkl_path  = st.text_input("Pickle file",         value=st.session_state.pkl_path,  key="pkl_path_input")
        st.session_state.prox_path = st.text_input("Proximity CSV",       value=st.session_state.prox_path, key="prox_path_input")
        st.session_state.gt_idx    = st.text_input("GT index CSV",        value=st.session_state.gt_idx,    key="gt_idx_input",  placeholder="embu_trends_category_index_*.csv")
        st.session_state.gt_feat   = st.text_input("GT features CSV",     value=st.session_state.gt_feat,   key="gt_feat_input", placeholder="embu_google_trends_features_*.csv")
        if st.button("Reload data", key="reload_btn"):
            st.cache_data.clear(); st.rerun()

# Apply any path overrides from sidebar
PKL_PATH         = st.session_state.pkl_path
PROX_CSV         = st.session_state.prox_path if os.path.exists(st.session_state.prox_path) else PROX_CSV
GT_INDEX_PATH    = st.session_state.gt_idx.strip()
GT_FEATURES_PATH = st.session_state.gt_feat.strip()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pkl(path):
    with open(path,"rb") as f: return pickle.load(f)

if not os.path.exists(PKL_PATH):
    st.error(f"Data file not found: `{PKL_PATH}`. Run the notebook first.")
    st.stop()

with st.spinner("Loading…"):
    data = load_pkl(PKL_PATH)

disp          = data["disp"].copy()
inv           = data["inv"].copy()
pat           = data["pat"].copy()
diag_df       = data["diag_df"].copy()
disp_df       = data["disp_df"].copy()
pred_output   = data["pred"].copy()
product_intel = data.get("product_intel", pd.DataFrame())

# ── NEW: Google Trends datasets from pickle ────────────────────────────────
_gt_pkl_index = data.get("embu_trends_category_index", pd.DataFrame())
_gt_pkl_raw   = data.get("embu_google_trends",         pd.DataFrame())

# Normalise column names so downstream code works regardless of source
if not _gt_pkl_index.empty and "month" in _gt_pkl_index.columns:
    _gt_pkl_index["month"] = pd.to_datetime(_gt_pkl_index["month"])

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

if "therapeutic_group" not in disp_df.columns:
    if "correct_therapeutic_class" in disp_df.columns:
        disp_df["therapeutic_group"] = disp_df["correct_therapeutic_class"].map(THERAPEUTIC_GROUP_MAP)
    else:
        disp_df["therapeutic_group"] = disp_df.get("new_category_name", pd.NA)

if "product_name" not in disp_df.columns:
    _pn = disp[["product_id","product_name"]].drop_duplicates()
    disp_df = disp_df.merge(_pn, on="product_id", how="left")

# ── OPENING STOCK PREP ────────────────────────────────────────────────────────
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
    cat_rev["n_products"]  = 1

new_rev   = cat_rev[cat_rev["Category"].isin(NEW_CATS)]
core_rev  = cat_rev[~cat_rev["Category"].isin(NEW_CATS)]
total_units   = int(cat_rev["total_units"].sum())
total_revenue = cat_rev["est_revenue"].sum()
new_units     = int(new_rev["total_units"].sum())
new_revenue   = new_rev["est_revenue"].sum()

def cat_u(cat): return int(cat_rev[cat_rev["Category"]==cat]["total_units"].sum()) if cat in cat_rev["Category"].values else 0
def cat_r(cat): return cat_rev[cat_rev["Category"]==cat]["est_revenue"].sum() if cat in cat_rev["Category"].values else 0

bty_u=cat_u("Beauty Products");        bty_r=cat_r("Beauty Products")
sup_u=cat_u("Vitamins & Supplements"); sup_r=cat_r("Vitamins & Supplements")
bb_u =cat_u("Body Building");          bb_r =cat_r("Body Building")

# ── COMPETITOR PROXIMITY PREP ────────────────────────────────────────────────

PHARMACY_CHAINS_LIST = [
    "Chain pharmacy","Good Life","Medisel","Portal","Haltons","Faraja","Pharmaken",
    "HealthPlus","Health Plus","Medicare","Dawa","Peoples","Ladnan","Citylife",
    "Pharmaplus","Lifecare","Life Care",
]

@st.cache_data(show_spinner=False)
def load_proximity(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    return df

prox_df = load_proximity(PROX_CSV)

# Derive competitor summary from proximity data
chain_competitors = pd.DataFrame()
nearest_chain = None; nearest_chain_dist = None; nearest_chain_name = "None"; n_independents = 0
if not prox_df.empty and "category" in prox_df.columns:
    comp_df = prox_df[prox_df["category"]=="pharmacy_competitor"].copy()
    comp_df = comp_df[comp_df["distance_km"].notna()].sort_values("distance_km")
    chain_competitors = comp_df[comp_df["is_chain"]==True].drop_duplicates("place_name")
    nearest_chain     = chain_competitors.iloc[0] if not chain_competitors.empty else None
    n_independents    = len(comp_df[comp_df["is_chain"]==False].drop_duplicates("place_name"))
    nearest_chain_dist = round(float(nearest_chain["distance_km"]),2) if nearest_chain is not None else None
    nearest_chain_name = str(nearest_chain["chain_name"]) if nearest_chain is not None else "None"

# Derive competitor summary
chain_competitors = pd.DataFrame()
if not prox_df.empty and "category" in prox_df.columns:
    comp_df = prox_df[prox_df["category"]=="pharmacy_competitor"].copy()
    comp_df = comp_df[comp_df["distance_km"].notna()].sort_values("distance_km")
    chain_competitors = comp_df[comp_df["is_chain"]==True].drop_duplicates("place_name")
    # Nearest chain competitor
    nearest_chain = chain_competitors.iloc[0] if not chain_competitors.empty else None
    n_independents = len(comp_df[comp_df["is_chain"]==False].drop_duplicates("place_name"))
    nearest_chain_dist = round(float(nearest_chain["distance_km"]),2) if nearest_chain is not None else None
    nearest_chain_name = str(nearest_chain["chain_name"]) if nearest_chain is not None else "None"
else:
    nearest_chain = None; nearest_chain_dist = None; nearest_chain_name = "None"; n_independents = 0

# ── PATIENT JOIN ──────────────────────────────────────────────────────────────
d_pat = disp.merge(pat[["patient_id","sex","age_group"]], on="patient_id", how="left")
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
    if logo_img: st.image(logo_img, width=120)
with ht:
    st.markdown(f"""
    <div style="display:flex;align-items:center;height:64px">
      <div>
        <div style="font-size:.6rem;font-weight:800;letter-spacing:.16em;
                    text-transform:uppercase;color:{MUTED};margin-bottom:.1rem">
          PHARMAPLUS CHAIN ANALYTICS · AFYA ANALYTICS PLATFORM</div>
        <div style="font-size:1.45rem;font-weight:800;color:{COOL_BLUE};
                    font-family:'Montserrat',sans-serif;line-height:1.1">
          New Venture Headstart — New Branch Intelligence</div>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:2px solid #EBF3FB;margin:.4rem 0 1rem'>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "The Full Story",
    "Deep Dive",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THE FULL STORY
# Problem → Market Signals → Stock List → Value
# ══════════════════════════════════════════════════════════════════════════════
with tab1:



    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — PROBLEM
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="background:{COOL_BLUE};border-radius:12px;padding:20px 26px;margin-bottom:6px">
      <div style="font-size:.6rem;font-weight:800;color:rgba(255,255,255,.4);text-transform:uppercase;
                  letter-spacing:.18em;margin-bottom:14px">The problem with opening a new branch</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;
                  background:rgba(255,255,255,.1);border-radius:8px;overflow:hidden;margin-bottom:16px">
        <div style="background:{COOL_BLUE};padding:14px 18px">
          <div style="font-size:1.3rem;font-weight:800;color:#fff;margin-bottom:4px">Wrong mix</div>
          <div style="font-size:.75rem;color:rgba(255,255,255,.5)">
            Stock copied from another branch — different catchment, different customers</div>
        </div>
        <div style="background:{COOL_BLUE};padding:14px 18px;
                    border-left:1px solid rgba(255,255,255,.1);border-right:1px solid rgba(255,255,255,.1)">
          <div style="font-size:1.3rem;font-weight:800;color:{CORAL};margin-bottom:4px">Dead stock Day 1</div>
          <div style="font-size:.75rem;color:rgba(255,255,255,.5)">
            Capital tied up in products the catchment doesn't want</div>
        </div>
        <div style="background:{COOL_BLUE};padding:14px 18px">
          <div style="font-size:1.3rem;font-weight:800;color:{ORANGE};margin-bottom:4px">3–6 months</div>
          <div style="font-size:.75rem;color:rgba(255,255,255,.5)">
            To understand the catchment through trial and error</div>
        </div>
      </div>
      <div style="font-size:.9rem;font-weight:700;color:#fff;
                  border-top:1px solid rgba(255,255,255,.1);padding-top:14px">
        Before you order a single unit —
        <span style="color:{TEAL}">do you know who lives around this branch?</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Catchment profile ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)
    sh("Who is in this catchment")

    # 4 KPI cards
    dem_cols = st.columns(4)
    dem_kpis = [
        ("👩 Women 15–49",    "176,000+",          TEAL,      "Core beauty & supplements buyers"),
        ("🧑 Men 15–34",      "~17% of pop",        ORANGE,    "Body building & sports nutrition"),
        ("🎓 Students",       "3 colleges · 1.5km", AFYA_BLUE, "Multi-category demand driver"),
        ("🏥 ANC Visit Rate", "68%",                PURPLE,    "Prenatal vitamins & iron demand signal"),
    ]
    for col, (label, value, color, sub) in zip(dem_cols, dem_kpis):
        kpi_card(col, label, value, color, sub)

    st.markdown("<div style='margin:.75rem 0'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Age band chart — fixed categorical ordering, no colour coding
        age_order = [] # will be appended with values exactly from the Age Group column
        adf = d_pat.groupby("age_group")["patient_id"].nunique().reset_index()
        adf.columns = ["Age Group","Patients"]
        for val in adf["Age Group"].unique():
            age_order.append(val)

        adf = adf[adf["Age Group"].isin(age_order)].copy()
        adf["Age Group"] = pd.Categorical(adf["Age Group"], categories=age_order, ordered=True)
        adf = adf.sort_values("Age Group").dropna(subset=["Age Group"])

        if adf["Age Group"].nunique() > 1:
            fig_age = px.bar(adf, x="Patients", y="Age Group", orientation="h",
                             text="Patients",
                             labels={"Patients":"Unique patients","Age Group":""})
            fig_age.update_traces(textposition="outside",
                                   textfont=dict(color="#0072CE", size=9),
                                   marker_color=AFYA_BLUE, marker_line_width=0)
            fig_age.update_layout(**CHART_LAYOUT, height=320, showlegend=False)
            fig_age.update_xaxes(**AXIS)
            fig_age.update_yaxes(**AXIS, categoryorder="array", categoryarray=age_order[::-1])
            st.markdown('<div class="chart-card"><div class="card-title">Patient age bands</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Gender pie below age chart
        gdf = d_pat.groupby("sex")["patient_id"].nunique().reset_index()
        gdf.columns = ["Gender","Patients"]
        total_pat = gdf["Patients"].sum()
        female_pct = round(gdf[gdf["Gender"].isin(["female","F"])]["Patients"].sum() / total_pat * 100) if total_pat > 0 else 50
        fig_g = px.pie(gdf, names="Gender", values="Patients", hole=0.62,
                       color="Gender",
                       color_discrete_map={"female":TEAL,"male":AFYA_BLUE,"F":TEAL,"M":AFYA_BLUE})
        fig_g.update_traces(textposition="inside", textinfo="percent+label",
                             textfont=dict(size=11, color="#fff"))
        fig_g.update_layout(showlegend=True, **{k:v for k,v in CHART_LAYOUT.items() if k!="margin"},
                             height=200, margin=dict(t=0,b=4,l=0,r=0))
        st.markdown('<div class="chart-card"><div class="card-title">Gender split</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        # Population signals panel
        st.markdown(f"""
        <div class="chart-card" style="margin-bottom:10px">
          <div class="card-title">Population signals</div>
          <div style="padding:6px 0;border-bottom:1px solid {BORDER}">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <div><div style="font-size:.75rem;color:{COOL_BLUE};font-weight:600">ANC 4+ visit rate</div>
              <div style="font-size:.65rem;color:{MUTED}">Prenatal vitamins must-stock</div></div>
              <div style="font-size:.85rem;font-weight:800;color:{TEAL}">68%</div>
            </div>
          </div>
          <div style="padding:6px 0;border-bottom:1px solid {BORDER}">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <div><div style="font-size:.75rem;color:{COOL_BLUE};font-weight:600">Women 15–49</div>
              <div style="font-size:.65rem;color:{MUTED}">Core beauty buyers</div></div>
              <div style="font-size:.85rem;font-weight:800;color:{TEAL}">176K+</div>
            </div>
          </div>
          <div style="padding:6px 0;border-bottom:1px solid {BORDER}">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <div><div style="font-size:.75rem;color:{COOL_BLUE};font-weight:600">Population growth</div>
              <div style="font-size:.65rem;color:{MUTED}">Growing catchment</div></div>
              <div style="font-size:.85rem;font-weight:800;color:{AFYA_BLUE}">3.1%/yr</div>
            </div>
          </div>
          <div style="padding:6px 0;border-bottom:1px solid {BORDER}">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <div><div style="font-size:.75rem;color:{COOL_BLUE};font-weight:600">Median age</div>
              <div style="font-size:.65rem;color:{MUTED}">Young → beauty, BB, supplements</div></div>
              <div style="font-size:.85rem;font-weight:800;color:{AFYA_BLUE}">20 yrs</div>
            </div>
          </div>
          <div style="padding:6px 0">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <div><div style="font-size:.75rem;color:{COOL_BLUE};font-weight:600">Modern FP use</div>
              <div style="font-size:.65rem;color:{MUTED}">Contraceptive demand confirmed</div></div>
              <div style="font-size:.85rem;font-weight:800;color:{PURPLE}">52%</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Proximity — full width single row
    prox_items = []
    if not prox_df.empty and "category" in prox_df.columns:
        cats_wanted = {
            "gym_fitness":         ("🏋","Gym",ORANGE,"Body Building demand"),
            "beauty_salon_spa":    ("💇","Beauty salons",TEAL,"Beauty referral traffic"),
            "university_college":  ("🎓","Colleges",AFYA_BLUE,"Student segment"),
            "health_facility":     ("🏥","Health facility",AFYA_BLUE,"Rx spillover + supplements"),
            "supplements_vitamins":("💊","Supplement stores",PURPLE,"Category demand confirmed"),
            "supermarket_chain":   ("🛒","Chain supermarket",AFYA_BLUE,"Stock therapeutic range"),
            "pharmacy_competitor": ("🏪","Chain pharmacy",CORAL,"Differentiate on range"),
        }
        for cat_key,(icon,label,color,impl) in cats_wanted.items():
            sub = prox_df[(prox_df["category"]==cat_key) & (prox_df["distance_km"].notna())]
            if sub.empty: continue
            sub = sub.sort_values("distance_km")
            nearest_km = round(float(sub["distance_km"].iloc[0]),1)
            count = sub["place_name"].nunique()
            detail = f"{count} within {nearest_km}km" if nearest_km <= 3 else f"Nearest: {nearest_km}km"
            if cat_key == "pharmacy_competitor":
                detail = f"{nearest_km}km away"
            prox_items.append((icon,label,detail,impl,color))

    # Fallback static items if CSV not loading
    if not prox_items:
        prox_items = [
            ("🏋","Gym","1 · 1.8km away","Body Building demand",ORANGE),
            ("💇","Beauty salons","4+ within 1km","Beauty referral traffic",TEAL),
            ("🎓","Colleges","3 within 1.5km","Student segment",AFYA_BLUE),
            ("🏥","Health facility","Adjacent","Rx spillover + supplements",AFYA_BLUE),
            ("💊","Supplement stores","2 within 2km","Category demand confirmed",PURPLE),
            ("🛒","Chain supermarket","3 within 2km","Stock therapeutic range",AFYA_BLUE),
            ("🏪","Chain pharmacy","0.78km away","Differentiate on range",CORAL),
        ]

    prox_html = ""
    for icon,label,detail,impl,color in prox_items:
        bg = f"background:#FEF0F0;border:1px solid {CORAL}30" if color==CORAL else f"background:{color}08;border:1px solid {color}20"
        prox_html += f"""
        <div style="padding:9px 11px;border-radius:8px;{bg};flex:1;min-width:0">
          <div style="font-size:.78rem;font-weight:700;color:{'#e05c5c' if color==CORAL else COOL_BLUE};margin-bottom:2px">{icon} {label}</div>
          <div style="font-size:.7rem;color:{MUTED};margin-bottom:3px">{detail}</div>
          <div style="font-size:.68rem;font-weight:700;color:{color}">→ {impl}</div>
        </div>"""

    st.markdown(f'''
    <div class="chart-card" style="margin-bottom:10px">
      <div class="card-title">What is near this branch</div>
      <div style="display:flex;gap:8px">{prox_html}</div>
    </div>''', unsafe_allow_html=True)

    # Market context bars
    sh("Market context")
    market_bars = [
        (ORANGE,  "Jumia Kenya 2025",    "Beauty & Perfumes = #2 category nationally · 16% of rural orders · demand outpaces local supply", "16% rural orders"),
        (TEAL,    "HPC Magazine 2026",   "Kenya skincare market growing at 11% annually · projected KES 16B by 2026 · pharmacy is a key channel", "11% growth/yr"),
        (AFYA_BLUE,"Jumia Rural Report", "Rural consumers: price is #1 driver at 58.9% · generics preferred over branded originals", "58.9% price-driven"),
    ]
    for color, source, text, stat in market_bars:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:9px 14px;margin-bottom:6px;
                    background:#fff;border:1px solid {BORDER};border-left:4px solid {color};border-radius:0 8px 8px 0">
          <div style="font-size:.68rem;font-weight:700;color:{MUTED};white-space:nowrap;min-width:110px">{source}</div>
          <div style="flex:1;font-size:.75rem;color:{COOL_BLUE}">{text}</div>
          <div style="font-size:.82rem;font-weight:800;color:{color};white-space:nowrap">{stat}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)

    # ── Google Trends analysis ───────────────────────────────────────────────────
    st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)
    sh("Google Trends — are people searching for these products in this catchment?")

    gt_cat_sel = st.radio(
        "", ["✨ Beauty", "💊 Vitamins & Supplements", "💪 Body Building"],
        horizontal=True, key="gt_cat_radio", label_visibility="collapsed"
    )
    gt_filter_map = {
        "✨ Beauty": "Beauty Products",
        "💊 Vitamins & Supplements": "Vitamins & Supplements",
        "💪 Body Building": "Body Building",
    }
    gt_selected_cat = gt_filter_map[gt_cat_sel]


 # ── Resolve GT data: pickle takes priority, CSV is fallback ───────────────
    gt_index = pd.DataFrame()
    gt_feat  = pd.DataFrame()

    # 1. Try pickle first (already loaded above as _gt_pkl_index / _gt_pkl_raw)
    if not _gt_pkl_index.empty:
        gt_index = _gt_pkl_index.copy()
    if not _gt_pkl_raw.empty:
        gt_feat = _gt_pkl_raw.copy()

    # 2. Fall back to CSV paths if pickle didn't have them
    _gt_idx_path  = st.session_state.get("gt_idx",  "").strip()
    _gt_feat_path = st.session_state.get("gt_feat", "").strip()

    if gt_index.empty and _gt_idx_path and os.path.exists(_gt_idx_path):
        try:
            gt_index = pd.read_csv(_gt_idx_path, parse_dates=["month"])
        except Exception as _e:
            st.warning(f"Could not load GT index CSV: {_e}")

    if gt_feat.empty and _gt_feat_path and os.path.exists(_gt_feat_path):
        try:
            gt_feat = pd.read_csv(_gt_feat_path)
        except Exception as _e:
            st.warning(f"Could not load GT features CSV: {_e}")

    if gt_index.empty and not _gt_idx_path:
        st.info("GT data not found in pickle and no CSV path set. "
                "Click ⚙️ Data paths and paste the CSV paths, then Reload data.")

    if not gt_index.empty:
        # Brand keys per category
        beauty_keys = ["olay","bioderma","la_roche_posay","the_ordinary","cosrx",
                       "neutrogena","cerave","vaseline","eos","nivea"]
        supp_keys   = ["centrum","berocca","neurobion","redoxon","seven_seas",
                       "vitabiotics","blackmores","solgar","nature_made","becosules"]
        bb_keys     = ["optimum_nutrition","usn_protein","muscletech","myprotein",
                       "evox_protein","dymatize_protein","bsn_syntha_6","creatine"]

        cat_brand_map = {
            "Beauty Products":        (beauty_keys, TEAL),
            "Vitamins & Supplements": (supp_keys,   AFYA_BLUE),
            "Body Building":          (bb_keys,     ORANGE),
        }

        # Sub-category units from prod_out
        subcat_units = {}
        if has_products and not prod_out.empty:
            cc_b = "Category" if "Category" in prod_out.columns else "therapeutic_group"
            pn_b = "Product"  if "Product"  in prod_out.columns else "product_name"
            qc_b = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
            for cat_name, fn, label_map in [
                ("Beauty Products",        classify_beauty_subcat,
                 {"Skincare":"Skincare","Body Care":"Body Care","Hair Care":"Hair Care","Makeup":"Makeup","Fragrance":"Fragrance"}),
                ("Vitamins & Supplements", classify_supp_subcat,
                 {"Immune & Vitamin C":"Immune & Vit C","Multivitamins":"Multivitamins",
                  "Collagen & Beauty":"Collagen","Bone & Heart":"Bone & Heart","Energy & Stress":"Energy"}),
                ("Body Building",          classify_bb_subcat,
                 {"Protein":"Protein","Mass Gainer":"Mass Gainer","Creatine":"Creatine",
                  "Pre-Workout":"Pre-Workout","Amino Acids":"Amino Acids"}),
            ]:
                sub_prods = prod_out[prod_out[cc_b] == cat_name].copy()
                if not sub_prods.empty:
                    sub_prods["_sub"] = sub_prods[pn_b].apply(fn)
                    subcat_units[cat_name] = (sub_prods.groupby("_sub")[qc_b].sum()
                                               .sort_values(ascending=False).to_dict())

        # Build selected category panel
        sel_keys, sel_color = cat_brand_map.get(gt_selected_cat, (beauty_keys, TEAL))

        gt_left, gt_right = st.columns([3, 2])

        with gt_left:
            if not gt_feat.empty:
                bm_rows = []
                for b in sel_keys:
                    m_row = gt_feat[gt_feat["feature"] == f"gt_{b}_momentum"]
                    if not m_row.empty:
                        bm_rows.append({"brand": b.replace("_"," ").title(),
                                        "score": float(m_row["value"].iloc[0])})
                bm_rows.sort(key=lambda x: x["score"], reverse=True)

                def mom_color(s):
                    return TEAL if s >= 0.5 else (AFYA_BLUE if s >= 0.4 else "#adb5bd")

                rows_html = ""
                for r in bm_rows:
                    pct = int(r["score"] * 100)
                    col = mom_color(r["score"])
                    rows_html += (
                        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">' +
                        f'<div style="width:120px;flex-shrink:0;font-size:.75rem;font-weight:600;color:{COOL_BLUE};text-align:right">{r["brand"]}</div>' +
                        f'<div style="flex:1;height:8px;border-radius:4px;background:#EBF3FB;overflow:hidden">' +
                        f'<div style="height:100%;width:{pct}%;background:{col};border-radius:4px"></div></div>' +
                        f'<div style="width:30px;text-align:right;font-size:.72rem;font-weight:700;color:{col}">{r["score"]:.2f}</div>' +
                        f'</div>'
                    )
                cat_short = gt_selected_cat.replace("Vitamins & Supplements","Supplements")
                st.markdown(
                    f'<div class="chart-card"><div class="card-title">{cat_short} — brand momentum</div>' +
                    rows_html + '</div>',
                    unsafe_allow_html=True
                )

        with gt_right:
            # Category index + sparkline + sub-category units
            latest = gt_index.groupby("category")["category_index"].last().to_dict()
            idx_val = latest.get(gt_selected_cat, 0)

            # Avg momentum
            mom_vals = []
            if not gt_feat.empty:
                for b in sel_keys:
                    r = gt_feat[gt_feat["feature"] == f"gt_{b}_momentum"]
                    if not r.empty: mom_vals.append(float(r["value"].iloc[0]))
            avg_mom   = sum(mom_vals)/len(mom_vals) if mom_vals else 0
            trend_arrow = "↑ rising" if avg_mom > 0.5 else ("→ stable" if avg_mom > 0.3 else "↓ slowing")
            trend_color = TEAL if avg_mom > 0.5 else (ORANGE if avg_mom > 0.3 else CORAL)

            # Sparkline bars
            cat_trend = gt_index[gt_index["category"] == gt_selected_cat].sort_values("month")
            max_idx   = cat_trend["category_index"].max() if not cat_trend.empty else 1
            spark_bars = ""
            prev_year  = None
            for _, tr in cat_trend.iterrows():
                h  = int((tr["category_index"] / max_idx) * 32) if max_idx > 0 else 4
                h  = max(h, 3)
                yr = str(tr["month"])[:4]
                yr_label = (f'<div style="font-size:8px;color:{MUTED};text-align:center;margin-top:1px">{yr}</div>'
                            if yr != prev_year else
                            '<div style="font-size:8px;color:transparent">·</div>')
                prev_year = yr
                spark_bars += (
                    f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;flex:1">' +
                    f'<div style="width:100%;background:{sel_color};border-radius:2px 2px 0 0;height:{h}px"></div>' +
                    f'{yr_label}</div>'
                )

            # Sub-category rows
            scu = subcat_units.get(gt_selected_cat, {})
            subcat_rows = ""
            for sub_name, units in list(scu.items())[:6]:
                subcat_rows += (
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;' +
                    f'padding:4px 0;border-bottom:0.5px solid {BORDER}">' +
                    f'<div style="font-size:.72rem;color:{COOL_BLUE}">{sub_name}</div>' +
                    f'<div style="font-size:.75rem;font-weight:700;color:{sel_color}">{int(units):,} units</div>' +
                    f'</div>'
                )

            snap_html = (
                f'<div class="chart-card">' +
                f'<div class="card-title">Category index &middot; trend</div>' +
                f'<div style="font-size:1.8rem;font-weight:800;color:{sel_color};line-height:1">{idx_val:.2f}</div>' +
                f'<div style="font-size:.72rem;font-weight:700;color:{trend_color};margin-top:3px;margin-bottom:10px">{trend_arrow}</div>' +
                f'<div style="display:flex;align-items:flex-end;gap:1px;height:40px;margin-bottom:2px">{spark_bars}</div>' +
                f'</div>'
            )
            st.markdown(snap_html, unsafe_allow_html=True)

            if subcat_rows:
                sub_html = (
                    f'<div class="chart-card" style="margin-top:10px">' +
                    f'<div class="card-title">Sub-category breakdown — linked to stock list</div>' +
                    subcat_rows +
                    f'</div>'
                )
                st.markdown(sub_html, unsafe_allow_html=True)

    elif not _gt_idx_path:
        st.info("Click ⚙️ Data paths above and paste the Google Trends CSV paths, then click Reload data.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — THE STOCK LIST
    # ══════════════════════════════════════════════════════════════════════════
    sh("What to order — Day 1 stock list built from all signals above", teal=True)

    # Hero strip
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
          Sized at 35% of mature branch velocity — Month 1 penetration factor.<br>
          Beauty and Supplements adjusted upward by Google Trends momentum signal.<br>
          Generic/branded mix calibrated to peri-urban income profile.
        </div>
      </div>
      <div style="flex:1;padding:0 18px;border-right:1px solid rgba(255,255,255,.12)">
        <div style="display:inline-block;background:{TEAL};color:#fff;font-size:.62rem;
                    font-weight:800;padding:2px 9px;border-radius:20px;margin-bottom:6px">✨ BEAUTY</div>
        <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1">{bty_u:,}</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:2px">units to order</div>
        <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px">{fmt_ksh(bty_r)}</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.4)">est. Month 1</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:5px;line-height:1.5">
          Salons · GT 0.78 · Jumia rural signal · Women 15–49</div>
      </div>
      <div style="flex:1;padding:0 18px;border-right:1px solid rgba(255,255,255,.12)">
        <div style="display:inline-block;background:{AFYA_BLUE};color:#fff;font-size:.62rem;
                    font-weight:800;padding:2px 9px;border-radius:20px;margin-bottom:6px">💊 SUPPLEMENTS</div>
        <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1">{sup_u:,}</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:2px">units to order</div>
        <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px">{fmt_ksh(sup_r)}</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.4)">est. Month 1</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:5px;line-height:1.5">
          68% ANC · Adults 25–64 · Hospital spillover</div>
      </div>
      <div style="flex:1;padding:0 0 0 18px">
        <div style="display:inline-block;background:{ORANGE};color:#fff;font-size:.62rem;
                    font-weight:800;padding:2px 9px;border-radius:20px;margin-bottom:6px">💪 BODY BUILDING</div>
        <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1">{bb_u:,}</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:2px">units to order</div>
        <div style="font-size:1rem;font-weight:700;color:{TEAL};margin-top:5px">{fmt_ksh(bb_r)}</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.4)">est. Month 1</div>
        <div style="font-size:.68rem;color:rgba(255,255,255,.45);margin-top:5px;line-height:1.5">
          Gym 1.8km · 3 colleges · Men 15–34</div>
      </div>
    </div>""",unsafe_allow_html=True)

    # KPI row
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

    # Category cards
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
        "Beauty Products": [
            "4 salons within 1km — referral traffic",
            "176K women aged 15–49 in catchment",
            "GT beauty index 0.78 · rising",
            "Dermo-cosmetics market: KES 16B by 2026",
            "Rural beauty = #1 e-commerce category",
            "Generic/branded mix calibrated to income level",
        ],
        "Vitamins & Supplements": [
            "68% ANC visit rate — prenatal vitamins must-stock",
            "Health facility adjacent — Rx spillover",
            "Adults 25–64 = 45% of catchment",
            "GT supplements index 0.64 · stable",
            "3 colleges — student immunity demand",
        ],
        "Body Building": [
            "Gym confirmed within 2km",
            "3 colleges within 1.5km — student fitness",
            "Men 15–34 = 17% of catchment",
            "GT body building index 0.41 · seasonal",
            "Start lean — reorder Day 72 on sell-through",
        ],
    }
    CORE_WHY = {
        "Oral Solid Forms":      "Antibiotics, antidiabetics, antimalarials. High NCD + malaria burden in matched branches.",
        "Injectables":           "Adjacent health facility + 68% ANC visit rate. Artemether and oxytocin move fastest.",
        "Oral Liquid Forms":     "30%+ of catchment is under 15. Paediatric syrups and ORS are steady sellers.",
        "IV Fluids & Infusions": "Hospital-proximity product. Order conservatively — lower walk-in demand.",
        "Topical Preparations":  "Skin conditions common in matched branches. Moderate volume.",
    }

    def _conf_color(c): return TEAL if c=="High" else (ORANGE if c=="Medium" else CORAL)

    def render_cat_card(row, is_new):
        cat   = row["Category"]
        units = int(row["total_units"])
        rev   = row["est_revenue"]
        n_p   = int(row.get("n_products",0))
        why_raw = NEW_WHY.get(cat, None)
        if why_raw is not None and isinstance(why_raw, list):
            why = "".join([
                f'<div style="display:flex;gap:5px;margin-bottom:3px">' +
                f'<span style="color:{TEAL};font-weight:700;flex-shrink:0">·</span>' +
                f'<span>{item}</span></div>'
                for item in why_raw
            ])
        else:
            why = why_raw if why_raw else CORE_WHY.get(cat,"Based on catchment profile and KNN-matched branch data.")
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
          <div style="padding:10px 14px 7px;border-bottom:1px solid {BORDER}">
            <div style="font-size:.85rem;font-weight:800;color:{COOL_BLUE}">{cat}</div>
            <div style="font-size:.62rem;font-weight:700;color:{tlc};margin-top:2px;
                        text-transform:uppercase;letter-spacing:.06em">{tl}</div>
          </div>
          <div style="padding:12px 14px">
            <div style="font-size:1.75rem;font-weight:800;color:{COOL_BLUE};line-height:1">{units:,}</div>
            <div style="font-size:.68rem;color:{MUTED};margin-bottom:4px">units &middot; {n_p} products</div>
            <div style="font-size:.88rem;font-weight:700;color:{TEAL};margin-bottom:8px">{fmt_ksh(rev)} est. Month 1</div>
            <div style="font-size:.72rem;color:{COOL_BLUE};line-height:1.6;
                        opacity:.75;margin-bottom:8px">{why}</div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:7px">
              <div style="font-size:.62rem;color:{MUTED};width:58px;flex-shrink:0">Confidence</div>
              <div style="flex:1;height:5px;background:#EBF3FB;border-radius:3px;overflow:hidden">
                <div style="width:{cpct}%;height:100%;background:{cfc};border-radius:3px"></div>
              </div>
              <div style="font-size:.65rem;font-weight:700;color:{cfc};min-width:36px;text-align:right">{top_conf}</div>
            </div>
            <div style="background:{rcolor}15;border-radius:5px;padding:5px 9px;font-size:.72rem;
                        font-weight:700;color:{rcolor}">Reorder: {day}</div>
          </div>
        </div>"""

    # New category cards
    st.markdown(f'<div style="font-size:.65rem;font-weight:800;color:{TEAL};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:8px">New growth categories — order these first</div>',
                unsafe_allow_html=True)
    nc1,nc2,nc3 = st.columns(3)
    for col,cat_name in zip([nc1,nc2,nc3],NEW_CATS):
        row = cat_rev[cat_rev["Category"]==cat_name]
        if row.empty:
            row = pd.DataFrame([{"Category":cat_name,"total_units":0,"est_revenue":0,"n_products":0}])
        with col:
            st.markdown(render_cat_card(row.iloc[0],is_new=True),unsafe_allow_html=True)

    st.markdown("<div style='margin:.5rem 0'></div>",unsafe_allow_html=True)

    # Core pharma cards
    st.markdown(f'<div style="font-size:.65rem;font-weight:800;color:{MUTED};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:8px">Core pharmacy — order as usual</div>',
                unsafe_allow_html=True)
    core_sorted = core_rev.sort_values("total_units",ascending=False)
    top_core    = core_sorted.head(3); rest_core = core_sorted.iloc[3:]
    cc1,cc2,cc3 = st.columns(3)
    for col,(_,row) in zip([cc1,cc2,cc3],top_core.iterrows()):
        with col:
            st.markdown(render_cat_card(row,is_new=False),unsafe_allow_html=True)

    if not rest_core.empty:
        with st.expander(f"Show all core categories ({len(rest_core)} more)"):
            rcols = st.columns(3)
            for i,(_,row) in enumerate(rest_core.iterrows()):
                with rcols[i%3]:
                    st.markdown(render_cat_card(row,is_new=False),unsafe_allow_html=True)

    st.markdown("<div style='margin:.5rem 0'></div>",unsafe_allow_html=True)

    # Full product table
    sh("Full Stock List — Filter and Download")

    if has_products and not prod_out.empty:
        cc_name = "Category" if "Category" in prod_out.columns else "therapeutic_group"
        qc_name = "Opening Stock Qty" if "Opening Stock Qty" in prod_out.columns else "product_opening_qty"
        pn_col  = "Product" if "Product" in prod_out.columns else "product_name"

        # Build prod_out_display FIRST before filters reference it
        prod_out_display = prod_out.copy()
        prod_out_display["Sub-category"] = ""

        # New categories — keyword classification
        for mask_cat, fn in [
            ("Beauty Products",        classify_beauty_subcat),
            ("Vitamins & Supplements", classify_supp_subcat),
            ("Body Building",          classify_bb_subcat),
        ]:
            mask = prod_out_display[cc_name] == mask_cat
            if mask.any():
                prod_out_display.loc[mask, "Sub-category"] = (
                    prod_out_display.loc[mask, pn_col].apply(fn)
                )

        # Pharma — sub-category from new_category_name in disp_df
        new_cats = ["Beauty Products", "Vitamins & Supplements", "Body Building"]
        pharma_mask = ~prod_out_display[cc_name].isin(new_cats)
        if pharma_mask.any() and not disp_df.empty and "new_category_name" in disp_df.columns:
            name_to_subcat = (disp_df[["product_name","new_category_name"]]
                              .dropna(subset=["product_name","new_category_name"])
                              .drop_duplicates("product_name")
                              .set_index("product_name")["new_category_name"]
                              .to_dict())
            prod_out_display.loc[pharma_mask, "Sub-category"] = (
                prod_out_display.loc[pharma_mask, pn_col]
                .map(name_to_subcat)
                .fillna("")
            )

        f1,f2,f3,f4 = st.columns([2,2,2,2])
        with f1:
            cats_all = ["All categories"] + sorted(prod_out_display[cc_name].dropna().unique().tolist())
            sel_c = st.selectbox("Category",cats_all,key="sc_t1")
        with f2:
            if sel_c != "All categories":
                _sub_opts = (prod_out_display[prod_out_display[cc_name]==sel_c]["Sub-category"]
                             .dropna().replace("","").dropna().unique().tolist())
                _sub_opts = sorted([s for s in _sub_opts if s])
                sub_cats = ["All sub-categories"] + _sub_opts
            else:
                sub_cats = ["All sub-categories"]
            sel_sub = st.selectbox("Sub-category",sub_cats,key="ssc_t1")
        with f3:
            sel_cf = st.selectbox("Confidence",["All","High","Medium","Low"],key="scf_t1")
        with f4:
            sel_r = st.selectbox("Dead stock",["All","Clear only","Risk only"],key="sr_t1")

        disp_cols = [pn_col,cc_name,qc_name]
        if "Sub-category" in prod_out_display.columns: disp_cols.append("Sub-category")
        if "est_revenue" in prod_out_display.columns:      disp_cols.append("est_revenue")
        if "Historical Share" in prod_out_display.columns: disp_cols.append("Historical Share")
        if "Confidence" in prod_out_display.columns:       disp_cols.append("Confidence")
        if "Dead Stock Risk" in prod_out_display.columns:  disp_cols.append("Dead Stock Risk")

        tbl = prod_out_display[disp_cols].rename(columns={
            pn_col:"Product",cc_name:"Category",qc_name:"Order Qty",
            "est_revenue":"Est. Revenue (KES)",
        }).copy()

        if sel_c != "All categories": tbl = tbl[tbl["Category"]==sel_c]
        if sel_sub != "All sub-categories" and "Sub-category" in tbl.columns:
            tbl = tbl[tbl["Sub-category"]==sel_sub]
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
            "Sub-category":       st.column_config.TextColumn(width="medium"),
            "Confidence":         st.column_config.TextColumn(width="small"),
            "Dead Stock Risk":    st.column_config.TextColumn(width="small"),
            "Historical Share":   st.column_config.TextColumn(width="small"),
        }
        st.markdown(f'<div class="chart-card"><div class="card-title">Recommended opening stock — {len(tbl):,} products</div>',
                    unsafe_allow_html=True)
        st.dataframe(tbl,use_container_width=True,hide_index=True,
                     column_config={k:v for k,v in col_cfg.items() if k in tbl.columns},
                     height=400)
        st.download_button("Download full stock list (CSV)",
                           tbl.to_csv(index=False).encode("utf-8"),
                           "new_branch_opening_stock.csv","text/csv")
        st.markdown('</div>',unsafe_allow_html=True)

    # Dead stock list
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
                         },height=260)
            st.download_button("Download dead stock list (CSV)",
                               dead_show.to_csv(index=False).encode("utf-8"),
                               "new_branch_dead_stock.csv","text/csv")

    st.markdown("<div style='margin:.75rem 0'></div>",unsafe_allow_html=True)

    # Revenue + reorder
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
                textfont=dict(color="#0072CE",size=10),name=label,
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
            ("Oral Solid Forms","Day 18",TEAL,"Fastest movers — set reminder now"),
            ("Injectables","Day 22",TEAL,"Hospital demand — don't stockout"),
            ("Beauty Products","Day 38",TEAL,"NEW — GT momentum rising"),
            ("Vitamins & Supplements","Day 45",AFYA_BLUE,"NEW — steady adult demand"),
            ("Oral Liquid Forms","Day 55",AFYA_BLUE,"Paediatric — seasonal"),
            ("Body Building","Day 72",ORANGE,"NEW — reorder only what sold"),
            ("Wound Care","Hold",CORAL,"Check sell-through first"),
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
              <div style="font-size:.82rem;font-weight:800;color:{color};
                          white-space:nowrap;padding-left:8px">{day}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)



    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — MONTH 1 RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    sh("Month 1 recommendations — what to do and why")

    recs = [
        {
            "title": "Stock skincare generics first, dermo range second",
            "category": "Beauty Products",
            "color": TEAL,
            "icon": "✨",
            "action": "Lead with accessible generics — body lotion, face wash, SPF 30. "
                      "Add therapeutic dermo range (serums, acne treatments) as second tier.",
            "signals": [
                "4+ salons within 1km — referral traffic confirmed",
                "Catchment price sensitivity — generics over premium brands",
                "Chain supermarket nearby — differentiate on therapeutic range, not commodity",
                "Dermo-cosmetics market growing 11% annually — stock the right tier",
            ],
            "expected": f"Beauty: {bty_u:,} units · {fmt_ksh(bty_r)} est. Month 1",
        },
        {
            "title": "Prioritise prenatal vitamins and immunity supplements",
            "category": "Vitamins & Supplements",
            "color": AFYA_BLUE,
            "icon": "💊",
            "action": "Open with prenatal vitamins, iron, folic acid, and vitamin C. "
                      "Add immunity stack and multivitamins for adults 25–64.",
            "signals": [
                "68% ANC visit rate — prenatal supplements are non-negotiable",
                "Adjacent health facility — prescription spillover directly into supplements",
                "Adults 25–64 = 45% of catchment — multivitamin is core",
                "GT supplements index 0.64 — immunity trending upward",
            ],
            "expected": f"Supplements: {sup_u:,} units · {fmt_ksh(sup_r)} est. Month 1",
        },
        {
            "title": "Stock body building conservatively — reorder only what sells",
            "category": "Body Building",
            "color": ORANGE,
            "icon": "💪",
            "action": "Open with whey protein and creatine only. "
                      "Reorder on Day 72 based on actual sell-through. Expand range in Month 2.",
            "signals": [
                "Gym confirmed within 2km — demand exists but is catchment-specific",
                "3 colleges nearby — student fitness segment present",
                "GT body building index 0.41 — lower momentum than beauty and supplements",
                "Peri-urban income profile — start lean to avoid dead stock risk",
            ],
            "expected": f"Body Building: {bb_u:,} units · {fmt_ksh(bb_r)} est. Month 1",
        },
        {
            "title": "Core pharma: generic mix calibrated to this catchment",
            "category": "Core Pharma",
            "color": COOL_BLUE,
            "icon": "💉",
            "action": "Stock antibiotics, antimalarials, antidiabetics, antihypertensives. "
                      "Generic versions take priority. Injectables reorder Day 22.",
            "signals": [
                "Peri-urban catchment — branded originals do not match income profile",
                "Rural consumers: price is #1 purchase driver at 58.9%",
                "Adjacent health facility — injectables and Rx spillover confirmed",
                "KNN matched branches — disease burden anchors pharma quantities",
            ],
            "expected": f"Core pharma: {total_units - new_units:,} units · {fmt_ksh(total_revenue - new_revenue)} est. Month 1",
        },
    ]

    for rec in recs:
        sig_html = "".join([
            f'<div style="display:flex;align-items:flex-start;gap:6px;margin-bottom:3px">' +
            f'<div style="width:5px;height:5px;border-radius:50%;background:{rec["color"]};margin-top:5px;flex-shrink:0"></div>' +
            f'<div style="font-size:.72rem;color:{COOL_BLUE};opacity:.85">{s}</div></div>'
            for s in rec["signals"]
        ])
        st.markdown(f"""
        <div style="background:#fff;border:1px solid {BORDER};border-left:4px solid {rec['color']};
                    border-radius:10px;padding:16px 20px;margin-bottom:10px">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px">
            <div style="flex:1">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap">
                <span style="font-size:1.1rem">{rec['icon']}</span>
                <div style="font-size:.88rem;font-weight:800;color:{COOL_BLUE}">{rec['title']}</div>
                <span style="background:{rec['color']}18;color:{rec['color']};font-size:.6rem;
                             font-weight:800;padding:2px 8px;border-radius:20px;
                             text-transform:uppercase;letter-spacing:.06em">{rec['category']}</span>
              </div>
              <div style="font-size:.78rem;color:{COOL_BLUE};line-height:1.6;margin-bottom:8px;opacity:.85">
                {rec['action']}</div>
              <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                          letter-spacing:.08em;color:{MUTED};margin-bottom:5px">Supported by</div>
              {sig_html}
            </div>
            <div style="flex-shrink:0;text-align:right;min-width:170px">
              <div style="font-size:.6rem;font-weight:800;text-transform:uppercase;
                          letter-spacing:.1em;color:{MUTED};margin-bottom:4px">Expected Month 1</div>
              <div style="font-size:.82rem;font-weight:800;color:{rec['color']};line-height:1.5">
                {rec['expected']}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# Customer Spend · Generic/Branded · Disease Burden · Scenario Modelling
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    dd_tab1, dd_tab2, dd_tab3, dd_tab4 = st.tabs([
        "Customer Spend",
        "Generic vs Branded",
        "Disease Burden",
        "Scenario Modelling",
    ])

    # ── CUSTOMER SPEND ────────────────────────────────────────────────────────
    with dd_tab1:
        sh("How they spend — Sales does not equal profit")
        info("Fast-moving products drive volume. But slow movers and dead stock tie up capital. "
             "The dashboard surfaces both so your opening order avoids the mistakes other branches made.",
             ORANGE)

        sb1, sb2 = st.columns(2)
        with sb1:
            mv_counts = prod_monthly["movement"].value_counts().reset_index()
            mv_counts.columns = ["Movement","Products"]
            fig_mv = px.pie(mv_counts,names="Movement",values="Products",hole=0.55,
                            color="Movement",
                            color_discrete_map={"Fast":TEAL,"Medium":AFYA_BLUE,"Slow":ORANGE})
            fig_mv.update_traces(textposition="inside",textinfo="percent+label",
                                  textfont=dict(size=12,color="#fff"))
            fig_mv.update_layout(showlegend=True,**CHART_LAYOUT,height=280)
            st.markdown('<div class="chart-card"><div class="card-title">Product movement — existing branches</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_mv,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        with sb2:
            if not product_intel.empty and "monthly_velocity" in product_intel.columns:
                sp_df = (product_intel.groupby("category")
                         .agg(volume=("monthly_velocity","sum"),
                              margin=("monthly_margin_contribution_kes","sum"))
                         .reset_index().rename(columns={"category":"Category"}))
                sp_df["vol_share"]    = sp_df["volume"]/sp_df["volume"].sum()*100
                sp_df["margin_share"] = sp_df["margin"]/sp_df["margin"].sum()*100
            else:
                sp_df = pd.DataFrame({
                    "Category":["Pharma","Beauty Products","Vitamins & Supplements","Body Building"],
                    "vol_share":[72,11,13,4],"margin_share":[45,20,28,7],
                })
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
            st.markdown('<div class="chart-card"><div class="card-title">Volume share vs margin share</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_sp,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        sh("Recurring vs Episodic revenue", teal=True)
        re1, re2 = st.columns(2)
        with re1:
            rc_data = pd.DataFrame({
                "Category":["Pharma","Beauty Products","Vitamins & Supplements","Body Building"],
                "Monthly repurchase %":[15,30,65,70],
            })
            fig_rc2 = px.bar(rc_data,x="Category",y="Monthly repurchase %",
                             color="Category",
                             color_discrete_map={"Pharma":MUTED,"Beauty Products":TEAL,
                                                 "Vitamins & Supplements":AFYA_BLUE,"Body Building":ORANGE},
                             text="Monthly repurchase %")
            fig_rc2.update_traces(texttemplate="%{text}%",textposition="outside",marker_line_width=0,
                                   textfont=dict(size=9,color="#0072CE"))
            fig_rc2.update_layout(**CHART_LAYOUT,height=280,showlegend=False)
            fig_rc2.update_xaxes(**AXIS,tickangle=-20)
            fig_rc2.update_yaxes(**AXIS,ticksuffix="%")
            st.markdown('<div class="chart-card"><div class="card-title">Monthly repurchase rate by category</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_rc2,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        with re2:
            st.markdown(f"""
            <div style="padding:20px;background:#F4F8FC;border-radius:10px;
                        border:1px solid {BORDER};height:100%">
              <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                          letter-spacing:.1em;color:{AFYA_BLUE};margin-bottom:16px">
                Why repurchase cycle matters</div>
              <div style="display:flex;gap:10px;margin-bottom:12px">
                <div style="width:3px;background:{CORAL};border-radius:2px;flex-shrink:0"></div>
                <div>
                  <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">Pharma — episodic</div>
                  <div style="font-size:.75rem;color:{COOL_BLUE};line-height:1.55;opacity:.8">
                    A patient buys antibiotics when sick. Revenue tied to illness, not loyalty.</div>
                </div>
              </div>
              <div style="display:flex;gap:10px;margin-bottom:12px">
                <div style="width:3px;background:{TEAL};border-radius:2px;flex-shrink:0"></div>
                <div>
                  <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">Supplements — recurring</div>
                  <div style="font-size:.75rem;color:{COOL_BLUE};line-height:1.55;opacity:.8">
                    One multivitamin customer = 12 transactions per year. Loyalty compounds.</div>
                </div>
              </div>
              <div style="display:flex;gap:10px">
                <div style="width:3px;background:{ORANGE};border-radius:2px;flex-shrink:0"></div>
                <div>
                  <div style="font-size:.82rem;font-weight:700;color:{COOL_BLUE}">Body Building — loyal</div>
                  <div style="font-size:.75rem;color:{COOL_BLUE};line-height:1.55;opacity:.8">
                    Gym-goers who find their protein brand rarely switch. High lifetime value.</div>
                </div>
              </div>
            </div>""",unsafe_allow_html=True)

    # ── GENERIC VS BRANDED ────────────────────────────────────────────────────
    with dd_tab2:
        sh("Generic vs Branded — Getting the mix right from Day 1")
        info("This branch opens in a peri-urban catchment where income levels favour generics. "
             "A previous branch stocked too many branded originals for its demographics — "
             "this intelligence layer prevents that from happening again.",CORAL)

        pharma_disp = disp.copy()
        pharma_disp["is_generic"] = pharma_disp["product_name"].apply(infer_generic)
        pharma_classified = pharma_disp[pharma_disp["is_generic"].notna()].copy()

        gc1, gc2 = st.columns(2)
        with gc1:
            if not pharma_classified.empty:
                gen_vol = (pharma_classified.groupby("is_generic")
                           .agg(qty=("qty_dispensed","sum"),revenue=("total_sales_value","sum"))
                           .reset_index())
                gen_vol["Type"] = gen_vol["is_generic"].map({True:"Generic",False:"Branded"})
                fig_gv = px.bar(gen_vol,x="Type",y="qty",text="qty",
                                color="Type",color_discrete_map={"Generic":TEAL,"Branded":CORAL},
                                labels={"qty":"Units dispensed"})
                fig_gv.update_traces(texttemplate="%{text:,}",textposition="outside",
                                      marker_line_width=0,textfont=dict(color="#0072CE",size=10))
                fig_gv.update_layout(**CHART_LAYOUT,height=260,showlegend=False)
                fig_gv.update_xaxes(**AXIS); fig_gv.update_yaxes(**AXIS)
                st.markdown('<div class="chart-card"><div class="card-title">Generic vs branded — dispensing volume</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_gv,use_container_width=True)
                st.markdown('</div>',unsafe_allow_html=True)

        with gc2:
            if not pharma_classified.empty:
                gen_rev = pharma_classified.groupby("is_generic")["total_sales_value"].sum().reset_index()
                gen_rev["Type"] = gen_rev["is_generic"].map({True:"Generic",False:"Branded"})
                fig_gr = px.pie(gen_rev,names="Type",values="total_sales_value",hole=0.55,
                                color="Type",color_discrete_map={"Generic":TEAL,"Branded":CORAL})
                fig_gr.update_traces(textposition="inside",textinfo="percent+label",
                                      textfont=dict(size=13,color="#fff"))
                fig_gr.update_layout(showlegend=False,**CHART_LAYOUT,height=260)
                st.markdown('<div class="chart-card"><div class="card-title">Generic vs branded — revenue split</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_gr,use_container_width=True)
                st.markdown('</div>',unsafe_allow_html=True)

        if not product_intel.empty and "is_generic" in product_intel.columns:
            sh("New category generic/branded margin advantage",teal=True)
            nc_gb = (product_intel.groupby(["category","is_generic"])
                     .agg(products=("product_id","count"),avg_margin=("gross_margin_pct","mean"))
                     .reset_index())
            nc_gb["Type"] = nc_gb["is_generic"].map({True:"Generic",False:"Branded"})
            nc_gb["Avg margin %"] = (nc_gb["avg_margin"]*100).round(1)
            ni1, ni2 = st.columns(2)
            with ni1:
                fig_ni = px.bar(nc_gb,x="category",y="products",color="Type",barmode="group",
                                color_discrete_map={"Generic":TEAL,"Branded":CORAL},
                                labels={"category":"Category","products":"Product count"})
                fig_ni.update_traces(marker_line_width=0)
                fig_ni.update_layout(**CHART_LAYOUT,height=260)
                fig_ni.update_xaxes(**AXIS,tickangle=-20); fig_ni.update_yaxes(**AXIS)
                st.markdown('<div class="chart-card"><div class="card-title">Generic vs branded — new categories</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_ni,use_container_width=True)
                st.markdown('</div>',unsafe_allow_html=True)
            with ni2:
                fig_ma = px.bar(nc_gb,x="category",y="Avg margin %",color="Type",barmode="group",
                                color_discrete_map={"Generic":TEAL,"Branded":CORAL},
                                text="Avg margin %",labels={"category":"Category"})
                fig_ma.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                                      marker_line_width=0,textfont=dict(size=9,color="#0072CE"))
                fig_ma.update_layout(**CHART_LAYOUT,height=260)
                fig_ma.update_xaxes(**AXIS,tickangle=-20)
                fig_ma.update_yaxes(**AXIS,ticksuffix="%")
                st.markdown('<div class="chart-card"><div class="card-title">Margin advantage — generics earn more per unit</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_ma,use_container_width=True)
                st.markdown('</div>',unsafe_allow_html=True)

    # ── DISEASE BURDEN ────────────────────────────────────────────────────────
    with dd_tab3:
        sh("Disease burden — what conditions drive pharmacy visits")

        burden_exp = (diag_df.assign(group=diag_df["diagnosis_burden_group"].str.split("|"))
                      .explode("group"))
        burden_exp["group"] = burden_exp["group"].str.strip()

        def bundle_burden(g):
            g = str(g).lower()
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
        bundled_sum = bundled_sum.sort_values("Consultations",ascending=True)

        db1, db2 = st.columns(2)
        with db1:
            fig_bd = px.bar(bundled_sum,x="Consultations",y="Category",orientation="h",
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
            st.markdown('<div class="chart-card"><div class="card-title">NCD sub-categories</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_ns,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        sh("Antibiotic Stewardship",teal=True)
        FIRST  = ["amoxicillin","ampicillin","penicillin","cotrimoxazole","erythromycin"]
        SECOND = ["azithromycin","ciprofloxacin","augmentin","ceftriaxone","doxycycline","gentamicin"]
        THIRD  = ["meropenem","vancomycin","imipenem","colistin","linezolid"]
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
                st.markdown('<div class="chart-card"><div class="card-title">1st vs 2nd vs 3rd line per branch</div>',
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
                    <div style="background:#F4F8FC;border-radius:8px;padding:.55rem .8rem;
                                margin-bottom:.4rem;border-left:3px solid {c}">
                      <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                                  color:{MUTED}">{row['abx_tier']}</div>
                      <div style="color:{c};font-size:1.15rem;font-weight:700">
                        {int(row['qty_dispensed']):,} units</div>
                      <div style="font-size:.7rem;color:{MUTED}">{pct}% of antibiotics</div>
                    </div>""",unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)

    # ── SCENARIO MODELLING ────────────────────────────────────────────────────
    with dd_tab4:
        sh("Scenario modelling — what if you run a campaign?")
        info("Adjust the sliders to model the impact of partnerships and campaigns on opening stock. "
             "These multipliers apply on top of the base KNN prediction.", TEAL)

        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            gym_uplift      = st.slider("Gym partnership uplift — Body Building",0,50,0,5,
                                         help="Partner with nearby gym. Expected BB demand uplift %.")
            salon_uplift    = st.slider("Salon partnership uplift — Beauty",0,50,0,5,
                                         help="Referral with nearby salons. Expected Beauty demand uplift %.")
            wellness_uplift = st.slider("Wellness / NCD drive — Supplements",0,40,0,5,
                                         help="NCD screening + supplement recommendation campaign.")
        with sc_col2:
            student_uplift  = st.slider("Student drive — all categories",0,30,0,5,
                                         help="Campus activation campaign.")
            social_uplift   = st.slider("Social media campaign — Beauty + Supplements",0,30,0,5,
                                         help="Instagram/TikTok campaign targeting this catchment.")

        bb_adj  = bb_u  * (1 + (gym_uplift + student_uplift) / 100)
        bty_adj = bty_u * (1 + (salon_uplift + social_uplift + student_uplift) / 100)
        sup_adj = sup_u * (1 + (wellness_uplift + social_uplift + student_uplift) / 100)
        bb_rev_adj  = bb_r  * (1 + (gym_uplift + student_uplift) / 100)
        bty_rev_adj = bty_r * (1 + (salon_uplift + social_uplift + student_uplift) / 100)
        sup_rev_adj = sup_r * (1 + (wellness_uplift + social_uplift + student_uplift) / 100)
        rev_uplift  = (bb_rev_adj + bty_rev_adj + sup_rev_adj) - new_revenue

        st.markdown("<div style='margin:.75rem 0'></div>",unsafe_allow_html=True)
        sm1,sm2,sm3,sm4 = st.columns(4)
        kpi_card(sm1,"Beauty (adjusted)",f"{int(bty_adj):,}",TEAL,f"{fmt_ksh(bty_rev_adj)} est.")
        kpi_card(sm2,"Supplements (adjusted)",f"{int(sup_adj):,}",AFYA_BLUE,f"{fmt_ksh(sup_rev_adj)} est.")
        kpi_card(sm3,"Body Building (adjusted)",f"{int(bb_adj):,}",ORANGE,f"{fmt_ksh(bb_rev_adj)} est.")
        kpi_card(sm4,"Campaign revenue uplift",fmt_ksh(rev_uplift),
                 TEAL if rev_uplift>0 else MUTED,"vs base prediction")