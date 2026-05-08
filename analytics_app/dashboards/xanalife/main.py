import streamlit as st

try:
    st.set_page_config(
        page_title="XanaLife Dashboards",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except Exception:
    pass  

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="block-container"] { padding: 1.5rem 2rem !important; }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #f5f7fb !important;
    }

    .hero {
        position: relative;
        border-radius: 18px;
        overflow: hidden;
        background: linear-gradient(135deg, #0d6efd 0%, #4e73df 55%, #6f42c1 100%);
        padding: 40px 48px;
        box-shadow: 0 18px 40px rgba(13,110,253,.22);
        margin-bottom: 32px;
    }
    .hero::before {
        content: "";
        position: absolute; inset: 0;
        background-image:
            radial-gradient(circle at 90% -10%, rgba(255,255,255,.22), transparent 45%),
            radial-gradient(circle at 0%  110%, rgba(255,255,255,.10), transparent 40%);
        pointer-events: none;
    }
    .hero-inner { position: relative; z-index: 1; }
    .hero-eyebrow {
        display: inline-block; padding: 5px 14px;
        font-size: 11px; font-weight: 500; letter-spacing: .7px; text-transform: uppercase;
        background: rgba(255,255,255,.18); border-radius: 99px;
        margin-bottom: 14px; color: #fff;
    }
    .hero h1 { font-size: 28px; font-weight: 600; margin: 0 0 10px; color: #fff; line-height: 1.2; }
    .hero p  { font-size: 15px; opacity: .9; margin: 0; color: #fff; }

    .dash-card {
        background: #fff;
        border-radius: 16px; padding: 22px;
        border: 1px solid #eef2f7;
        box-shadow: 0 4px 18px rgba(15,23,42,.04);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        cursor: pointer;
        display: flex; flex-direction: column;
        height: 190px; margin-bottom: 4px;
    }
    .dash-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 30px rgba(13,110,253,.12);
        border-color: #bfdbfe;
    }
    .dash-icon {
        width: 44px; height: 44px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; margin-bottom: 14px; flex-shrink: 0;
    }
    .icon-blue   { background: linear-gradient(135deg, #dbeafe, #93c5fd); }
    .icon-teal   { background: linear-gradient(135deg, #ccfbf1, #5eead4); }
    .icon-purple { background: linear-gradient(135deg, #ede9fe, #c4b5fd); }
    .icon-orange { background: linear-gradient(135deg, #ffedd5, #fed7aa); }

    .dash-name { font-size: 15px; font-weight: 600; color: #0f172a; margin: 0 0 6px; }
    .dash-desc { font-size: 12px; color: #64748b; line-height: 1.55; flex: 1; }
    .dash-footer { display: flex; justify-content: space-between; align-items: center; margin-top: 14px; }
    .dash-badge { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; }
    .dash-arrow {
        width: 28px; height: 28px; border-radius: 8px; background: #f1f5f9;
        display: flex; align-items: center; justify-content: center;
        color: #0d6efd; font-weight: bold; font-size: 14px;
    }
    .section-label { font-size: 11px; font-weight: 600; color: #0d6efd; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 4px; }
    .section-title { font-size: 22px; font-weight: 600; color: #0f172a; margin: 0 0 4px; }
    .section-sub   { font-size: 14px; color: #64748b; margin: 0 0 22px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">XanaLife Analytics</div>
    <h1>XanaLife Dashboards</h1>
    <p>4 dashboards available — click any card to open.</p>
  </div>
</div>
<div class="section-label">Available Dashboards</div>
<div class="section-title">Choose a dashboard</div>
<div class="section-sub">All views are scoped to your organization and updated in real time.</div>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components

DASHBOARDS = [
    {"name": "Customer Analytics",     "slug": "customer_analytics_dashboard",   "desc": "Analyze customer behavior, segments, and lifetime value.",     "icon": "👥", "color": "icon-blue"},
    {"name": "Revenue",                "slug": "revenue_dashboard",              "desc": "Track revenue trends, collections, and financial performance.", "icon": "💰", "color": "icon-teal"},
    {"name": "Cross Selling Intelligence & Inventory", "slug": "cross_sell_inventory_dashboard", "desc": "Cross-sell opportunities and inventory health metrics.",        "icon": "📦", "color": "icon-orange"},
    {"name": "Predictions",            "slug": "predictions_dashboard",          "desc": "Machine learning forecasts and predictive analytics.",          "icon": "🔮", "color": "icon-purple"},
]

card_items = ""
for d in DASHBOARDS:
    path = f"/dashboards/dashboards/{d['slug']}/"
    card_items += f"""
    <div class="dash-card" onclick="window.top.postMessage({{navigate: '{path}'}}, '*')">
      <div class="dash-icon {d['color']}">{d['icon']}</div>
      <div class="dash-name">{d['name']}</div>
      <div class="dash-desc">{d['desc']}</div>
      <div class="dash-footer">
        <span class="dash-badge">Analytics</span>
        <div class="dash-arrow">&#8594;</div>
      </div>
    </div>
    """

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; padding: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
  .dash-card {{
    background: #fff; border-radius: 16px; padding: 22px;
    border: 1px solid #eef2f7;
    box-shadow: 0 4px 18px rgba(15,23,42,.04);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    cursor: pointer; display: flex; flex-direction: column; height: 190px;
  }}
  .dash-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 30px rgba(13,110,253,.12);
    border-color: #bfdbfe;
  }}
  .dash-icon {{
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 14px; flex-shrink: 0;
  }}
  .icon-blue   {{ background: linear-gradient(135deg, #dbeafe, #93c5fd); }}
  .icon-teal   {{ background: linear-gradient(135deg, #ccfbf1, #5eead4); }}
  .icon-purple {{ background: linear-gradient(135deg, #ede9fe, #c4b5fd); }}
  .icon-orange {{ background: linear-gradient(135deg, #ffedd5, #fed7aa); }}
  .dash-name {{ font-size: 15px; font-weight: 600; color: #0f172a; margin: 0 0 6px; }}
  .dash-desc {{ font-size: 12px; color: #64748b; line-height: 1.55; flex: 1; }}
  .dash-footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: 14px; }}
  .dash-badge {{ font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; }}
  .dash-arrow {{
    width: 28px; height: 28px; border-radius: 8px; background: #f1f5f9;
    display: flex; align-items: center; justify-content: center;
    color: #0d6efd; font-weight: bold; font-size: 14px;
  }}
</style>
</head>
<body>
  <div class="grid">
    {card_items}
  </div>
</body>
</html>
""", height=460, scrolling=False)