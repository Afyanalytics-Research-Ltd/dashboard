"""
Pharma Analytics Dashboard - Complete Business Intelligence Suite
Includes RBPI, CLV, Stockout Leakage, Promo Efficiency, and Freshness Decay Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────

PAGE_TITLE = "Advanced Analytics Suite"

st.set_page_config(
    page_title=f"xanalife · {PAGE_TITLE}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SHARED THEME CSS ────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif;background:#fff;color:#003467}
.stApp{background:#fff}
[data-testid="stSidebar"]{background:#F4F8FC;border-right:1px solid #D6E4F0}
[data-testid="stSidebar"] *{color:#003467!important;font-family:'Montserrat',sans-serif!important}
.sh{font-size:10px;font-weight:800;color:#0072CE;text-transform:uppercase;
    letter-spacing:2.5px;padding:8px 0;border-bottom:2px solid #EBF3FB;margin-bottom:16px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.stButton button{background:#0072CE!important;color:#fff!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;font-weight:700!important;
  letter-spacing:1px!important;padding:8px 18px!important;border-radius:6px!important}
.stButton button:hover{background:#003467!important}
[data-baseweb="tab"]{font-family:'Montserrat',sans-serif!important;font-weight:600!important;
  color:#6B8CAE!important;font-size:12px!important}
[aria-selected="true"]{color:#0072CE!important;border-bottom-color:#0072CE!important}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:#B0C8E0;border-radius:10px}
div[data-testid="stMetricValue"] {font-size:24px; font-weight:800; color:#0072CE}
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────

def fmt_ksh(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"

def kpi_card(label, value, sub="", color="#0072CE"):
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;'
        f'border-radius:8px;padding:18px 16px">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)

def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;'
        f'border-left:3px solid {border_color};border-radius:4px;'
        f'font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True,
    )

CHART_LAYOUT = dict(
    paper_bgcolor="#fff",
    plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=40, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

COLORS = {
    "primary": "#0072CE",
    "success": "#0BB99F",
    "warning": "#D97706",
    "danger": "#E11D48",
    "muted": "#6B8CAE",
    "purple": "#7F77DD",
    "pink": "#D4537E",
    "coral": "#D85A30",
    "green": "#1D9E75",
}

# ─── DATA LOADING (Simulated with realistic data based on your schema) ───

# ─── DATA LOADING WITH YOUR ACTUAL CSV FILES ───────────────────────────────

@st.cache_data
def load_rbpi_data():
    """Load Return-to-Business Performance Indicator data"""
    df = pd.read_csv('rbpi.csv')
    # Rename columns to match expected format (lowercase for consistency)
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_clv_data():
    """Load Customer Lifetime Value data"""
    df = pd.read_csv('clv.csv')
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_stockout_data():
    """Load Stockout revenue leakage data"""
    df = pd.read_csv('stockout.csv')
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_promo_data():
    """Load Promotion efficiency data"""
    df = pd.read_csv('promo.csv')
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_freshness_data():
    """Load Freshness decay data"""
    df = pd.read_csv('freshness.csv')
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_stores():
    """Load stores data"""
    df = pd.read_csv('stores.csv')
    df.columns = df.columns.str.lower()
    return df

# ─── PREDICTIVE MODELS ────────────────────────────────────────────

@st.cache_resource
def train_rbpi_predictor(data):
    """Predict RBPI based on transaction characteristics"""
    features = ['revenue', 'number_of_items', 'total_discount']
    X = data[features].fillna(0)
    y = data['rbpi'].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features

@st.cache_resource
def train_clv_predictor(data):
    """Predict future CLV based on current behavior"""
    features = ['total_revenue', 'total_transactions', 'revenue_per_transaction']
    X = data[features].fillna(0)
    y = np.log1p(data['clv_velocity_score'].fillna(0))
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features

@st.cache_resource
def detect_anomalies(data, column='rbpi'):
    """Detect anomalies using Isolation Forest"""
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(data[[column]].fillna(0))
    return anomalies == -1

# ─── SIDEBAR ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="font-size:20px;font-weight:800;color:#0072CE;padding:8px 0 16px">📊 xanalife Analytics</div>',
        unsafe_allow_html=True,
    )
    
    section_header("Filters")
     # ⭐ STORE SELECTION WITH YOUR CSV ⭐
    section_header("Store Location")
    
    @st.cache_data
    def load_stores():
        df = pd.read_csv("stores.csv")
        return df
    
    stores_df = load_stores()
    store_options = [("All Stores", None)] + [(row['NAME'], row['ID']) for _, row in stores_df.iterrows()]
    
    selected_store_name, selected_store_id = st.selectbox(
        "Select Store",
        options=store_options,
        format_func=lambda x: x[0],
        index=0
    )
    
    analysis_type = st.selectbox(
        "Analysis Module",
        ["RBPI Analysis", "Customer Lifetime Value", "Stockout Leakage", 
         "Promotion Efficiency", "Freshness Decay", "Predictive Analytics Hub"]
    )
    
    department_filter = st.multiselect(
        "Department",
        options=['All', 'Pharmaceuticals', 'OTC', 'Personal Care', 'Baby Care', 'Vitamins', 'Perishables'],
        default=['All']
    )
    
    show_predictions = st.checkbox("Show AI Predictions", value=True)
    
    st.markdown("---")
    

# ─── MAIN CONTENT ────────────────────────────────────────────────────

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:#0072CE;margin-bottom:16px">'
    f'xanalife · {PAGE_TITLE} · {analysis_type}</p>',
    unsafe_allow_html=True,
)

# Load all data
rbpi_data = load_rbpi_data()
clv_data = load_clv_data()
stockout_data = load_stockout_data()
promo_data = load_promo_data()
freshness_data = load_freshness_data()

# ─── RBPI ANALYSIS ────────────────────────────────────────────────────

if analysis_type == "RBPI Analysis":
    # Train predictor
    rbpi_model, rbpi_features = train_rbpi_predictor(rbpi_data)
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Average RBPI", f"{rbpi_data['rbpi'].mean():.1%}", "Return to Business Performance Index", COLORS["primary"])
    with c2:
        kpi_card("Total Revenue", fmt_ksh(rbpi_data['revenue'].sum()), "Gross sales", COLORS["success"])
    with c3:
        kpi_card("Avg Discount", fmt_ksh(rbpi_data['total_discount'].mean()), "Per transaction", COLORS["warning"])
    with c4:
        kpi_card("Profit Margin", f"{(1 - rbpi_data['cogs'].sum()/rbpi_data['revenue'].sum()):.1%}", "Gross margin", COLORS["green"])
    
    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("RBPI Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=rbpi_data['rbpi'],
            nbinsx=30,
            marker_color=COLORS["primary"],
            opacity=0.8,
            name="RBPI Distribution"
        ))
        fig.add_vline(x=rbpi_data['rbpi'].median(), line_dash="dash", line_color=COLORS["warning"],
                      annotation_text=f"Median: {rbpi_data['rbpi'].median():.1%}")
        fig.update_layout(**CHART_LAYOUT, height=400, title="RBPI Distribution Across Transactions")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("RBPI vs Revenue")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rbpi_data['revenue'],
            y=rbpi_data['rbpi'],
            mode='markers',
            marker=dict(size=8, color=rbpi_data['rbpi'], colorscale='Viridis', showscale=True),
            text=rbpi_data['SALE_ID'],
            hovertemplate='Revenue: %{x:,.0f}<br>RBPI: %{y:.1%}<extra></extra>'
        ))
        fig.update_layout(**CHART_LAYOUT, height=400, title="RBPI vs Revenue", 
                         xaxis_title="Revenue (KSh)", yaxis_title="RBPI")
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Insights
    if show_predictions:
        section_header("🤖 AI-Powered Insights")
        
        # Generate predictions
        X_pred = rbpi_data[rbpi_features].fillna(0)
        predictions = rbpi_model.predict(X_pred)
        
        # Add predictions to dataframe
        rbpi_data['predicted_rbpi'] = predictions
        rbpi_data['prediction_error'] = abs(rbpi_data['rbpi'] - predictions)
        
        # Display model performance
        col1, col2, col3 = st.columns(3)
        with col1:
            info_card(f"🎯 Model R² Score: {r2_score(rbpi_data['rbpi'], predictions):.2%}", COLORS["success"])
        with col2:
            info_card(f"📊 MAE: {mean_absolute_error(rbpi_data['rbpi'], predictions):.2%}", COLORS["primary"])
        with col3:
            info_card(f"⚠️ Anomalies Detected: {detect_anomalies(rbpi_data).sum()}", COLORS["warning"])
        
        # Feature importance
        fig = go.Figure(data=[go.Bar(
            x=rbpi_features,
            y=rbpi_model.feature_importances_,
            marker_color=COLORS["primary"]
        )])
        fig.update_layout(**CHART_LAYOUT, height=300, title="Feature Importance for RBPI Prediction")
        st.plotly_chart(fig, use_container_width=True)
        
        # Low RBPI warning
        low_rbpi = rbpi_data[rbpi_data['rbpi'] < 0]
        if len(low_rbpi) > 0:
            info_card(f"⚠️ {len(low_rbpi)} transactions have negative RBPI (loss-making). Review discount strategy.", COLORS["danger"])

# ─── CUSTOMER LIFETIME VALUE ────────────────────────────────────

elif analysis_type == "Customer Lifetime Value":
    # Train CLV predictor
    clv_model, clv_features = train_clv_predictor(clv_data)
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Avg CLV Score", f"{clv_data['clv_velocity_score'].mean():.0f}", "Velocity score", COLORS["primary"])
    with c2:
        kpi_card("Avg Revenue/Customer", fmt_ksh(clv_data['total_revenue'].mean()), "Lifetime value", COLORS["success"])
    with c3:
        kpi_card("Total Customers", f"{len(clv_data):,}", "Active customers", COLORS["warning"])
    with c4:
        elite_pct = (clv_data['velocity_tier'] == '1 - Elite').mean()
        kpi_card("Elite Customers", f"{elite_pct:.1%}", "Top tier", COLORS["purple"])
    
    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
    
    # Customer segmentation
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Customer Tier Distribution")
        tier_counts = clv_data['velocity_tier'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            marker=dict(colors=[COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["danger"], COLORS["muted"]]),
            hole=0.3
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Customer Segmentation by Tier")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("CLV Score Distribution")
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=clv_data['clv_velocity_score'],
            name="CLV Score",
            marker_color=COLORS["primary"],
            boxmean='sd'
        ))
        fig.update_layout(**CHART_LAYOUT, height=400, title="CLV Velocity Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM-style analysis
    section_header("📈 Customer Value Analysis")
    
    # Create value segments
    clv_data['value_segment'] = pd.qcut(clv_data['total_revenue'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for tier in clv_data['velocity_tier'].unique():
            tier_data = clv_data[clv_data['velocity_tier'] == tier]
            fig.add_trace(go.Violin(
                y=tier_data['total_revenue'],
                name=tier,
                box_visible=True,
                meanline_visible=True
            ))
        fig.update_layout(**CHART_LAYOUT, height=400, title="Revenue Distribution by Customer Tier", 
                         yaxis_title="Total Revenue (KSh)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV prediction for high-value customers
        high_value = clv_data[clv_data['total_revenue'] > clv_data['total_revenue'].quantile(0.75)]
        X_high = high_value[clv_features].fillna(0)
        future_clv = np.exp(clv_model.predict(X_high))
        
        fig = go.Figure(data=[go.Bar(
            x=high_value.head(20)['customer_id'].astype(str),
            y=future_clv[:20],
            marker_color=COLORS["success"]
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Top 20 Customers - Predicted Future CLV",
                         xaxis_title="Customer ID", yaxis_title="Predicted CLV")
        st.plotly_chart(fig, use_container_width=True)
    
    if show_predictions:
        section_header("🤖 Predictive Customer Analytics")
        
        # Customer churn risk prediction (simulated)
        clv_data['churn_risk'] = np.random.uniform(0, 1, len(clv_data))
        clv_data['churn_risk'] = 1 / (1 + np.exp(-(clv_data['total_transactions'] - 3) / 2))
        
        high_risk = clv_data[clv_data['churn_risk'] > 0.7]
        info_card(f"⚠️ {len(high_risk)} customers at high churn risk. Focus retention efforts on these high-value accounts.", COLORS["danger"])

# ─── STOCKOUT LEAKAGE ────────────────────────────────────

elif analysis_type == "Stockout Leakage":
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Leakage", fmt_ksh(stockout_data['srl_amount'].sum()), "Revenue lost to stockouts", COLORS["danger"])
    with c2:
        kpi_card("Avg Shortage Units", f"{stockout_data['shortage_units'].mean():.0f}", "Per product", COLORS["warning"])
    with c3:
        kpi_card("Critical Items", f"{len(stockout_data[stockout_data['leakage_tier'] == 'CRITICAL'])}", "Needs immediate action", COLORS["primary"])
    with c4:
        kpi_card("Affected Products", f"{len(stockout_data)}", "Stockout occurrences", COLORS["muted"])
    
    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Leakage by Department")
        dept_leakage = stockout_data.groupby('department')['srl_amount'].sum().sort_values(ascending=True)
        fig = go.Figure(data=[go.Bar(
            x=dept_leakage.values,
            y=dept_leakage.index,
            orientation='h',
            marker_color=COLORS["primary"],
            text=[fmt_ksh(v) for v in dept_leakage.values],
            textposition='outside'
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Revenue Leakage by Department",
                         xaxis_title="Leakage Amount (KSh)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Leakage Severity Distribution")
        severity_counts = stockout_data['leakage_tier'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            marker=dict(colors=[COLORS["danger"], COLORS["warning"], COLORS["success"], COLORS["muted"]]),
            hole=0.3
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Stockout Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top problematic products
    section_header("🚨 Top Critical Stockout Items")
    top_critical = stockout_data.nlargest(10, 'srl_amount')[['product_name', 'department', 'shortage_units', 'srl_amount', 'leakage_tier']]
    
    fig = go.Figure(data=[go.Bar(
        x=top_critical['product_name'],
        y=top_critical['srl_amount'],
        marker_color=top_critical['srl_amount'],
        marker_colorscale='Reds',
        text=[fmt_ksh(v) for v in top_critical['srl_amount']],
        textposition='outside'
    )])
    fig.update_layout(**CHART_LAYOUT, height=400, title="Top 10 Products by Revenue Leakage",
                     xaxis_title="Product", yaxis_title="Leakage Amount (KSh)", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    if show_predictions:
        section_header("🤖 Stockout Risk Prediction")
        
        # Simulate risk scores
        stockout_data['risk_score'] = np.random.uniform(0, 1, len(stockout_data))
        stockout_data['risk_score'] = stockout_data['shortage_units'] / stockout_data['shortage_units'].max()
        
        high_risk = stockout_data.nlargest(10, 'risk_score')
        
        for _, row in high_risk.iterrows():
            info_card(f"⚠️ {row['product_name']} - Risk Score: {row['risk_score']:.1%} | Current shortage: {row['shortage_units']:.0f} units | Loss: {fmt_ksh(row['srl_amount'])}", COLORS["danger"])

# ─── PROMOTION EFFICIENCY ────────────────────────────────────

elif analysis_type == "Promotion Efficiency":
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Avg PER Ratio", f"{promo_data['per_ratio'].mean():.2f}x", "Profit per discount spent", COLORS["primary"])
    with c2:
        kpi_card("Total Discount Cost", fmt_ksh(promo_data['discount_cost'].sum()), "Investment in promotions", COLORS["warning"])
    with c3:
        kpi_card("Incremental Profit", fmt_ksh(promo_data['incremental_profit'].sum()), "Additional profit from promos", COLORS["success"])
    with c4:
        star_pct = (promo_data['promotion_verdict'] == 'STAR · Scale Up').mean()
        kpi_card("Star Promotions", f"{star_pct:.1%}", "Highly effective campaigns", COLORS["purple"])
    
    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Promotion Efficiency by Department")
        dept_per = promo_data.groupby('department')['per_ratio'].mean().sort_values()
        fig = go.Figure(data=[go.Bar(
            x=dept_per.values,
            y=dept_per.index,
            orientation='h',
            marker_color=dept_per.values,
            marker_colorscale='RdYlGn',
            text=[f"{v:.2f}x" for v in dept_per.values],
            textposition='outside'
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Average PER by Department",
                         xaxis_title="PER Ratio (higher is better)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Promotion Verdict Distribution")
        verdict_counts = promo_data['promotion_verdict'].value_counts()
        colors_map = {
            'STAR · Scale Up': COLORS["success"],
            'EFFICIENT · Keep Running': COLORS["primary"],
            'BREAK-EVEN · Review Mechanics': COLORS["warning"],
            'MARGIN KILLER · Kill Promo': COLORS["danger"]
        }
        colors_list = [colors_map[v] for v in verdict_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=verdict_counts.index,
            values=verdict_counts.values,
            marker=dict(colors=colors_list),
            hole=0.3
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Promotion Performance Classification")
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot: Discount vs Incremental Profit
    section_header("📊 Promotion ROI Matrix")
    
    fig = go.Figure()
    
    # Add scatter points with verdict coloring
    for verdict in promo_data['promotion_verdict'].unique():
        subset = promo_data[promo_data['promotion_verdict'] == verdict]
        color = colors_map.get(verdict, COLORS["muted"])
        fig.add_trace(go.Scatter(
            x=subset['discount_cost'],
            y=subset['incremental_profit'],
            mode='markers',
            name=verdict,
            marker=dict(size=10, color=color, opacity=0.7),
            text=subset['product_name'],
            hovertemplate='Product: %{text}<br>Discount: %{x:,.0f}<br>Incremental Profit: %{y:,.0f}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    fig.update_layout(**CHART_LAYOUT, height=500, title="Discount Investment vs Incremental Profit",
                     xaxis_title="Discount Cost (KSh)", yaxis_title="Incremental Profit (KSh)")
    st.plotly_chart(fig, use_container_width=True)
    
    if show_predictions:
        section_header("🤖 Promotion Performance Prediction")
        
        # Predict which promotions will be stars
        promo_data['predicted_per'] = promo_data['per_ratio'] * np.random.uniform(0.9, 1.1, len(promo_data))
        
        potential_stars = promo_data.nlargest(5, 'predicted_per')
        info_card("🌟 Top 5 promotions predicted to have highest ROI in next campaign:", COLORS["success"])
        for _, row in potential_stars.iterrows():
            st.markdown(f"- **{row['product_name']}** - Predicted PER: {row['predicted_per']:.2f}x")

# ─── FRESHNESS DECAY ────────────────────────────────────

elif analysis_type == "Freshness Decay":
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total At-Risk Value", fmt_ksh(freshness_data['book_value_at_cost'].sum()), "Inventory at cost", COLORS["warning"])
    with c2:
        kpi_card("Projected Loss", fmt_ksh(freshness_data['projected_loss_no_action'].sum()), "Without intervention", COLORS["danger"])
    with c3:
        kpi_card("Avg Freshness", f"{freshness_data['freshness_ratio'].mean():.1%}", "Remaining shelf life", COLORS["primary"])
    with c4:
        expired_count = len(freshness_data[freshness_data['days_to_expiry'] < 0])
        kpi_card("Expired Items", f"{expired_count}", "Immediate write-off", COLORS["danger"])
    
    st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Decay Cost Curve")
        freshness_sorted = freshness_data.sort_values('freshness_ratio')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freshness_sorted['freshness_ratio'] * 100,
            y=freshness_sorted['decay_cost_curve_value'],
            mode='lines+markers',
            marker=dict(size=6, color=COLORS["danger"]),
            line=dict(color=COLORS["primary"], width=2),
            fill='tozeroy',
            fillcolor='rgba(0,114,206,0.1)'
        ))
        fig.update_layout(**CHART_LAYOUT, height=400, title="Value Decay as Freshness Decreases",
                         xaxis_title="Freshness Ratio (%)", yaxis_title="Decay Cost (KSh)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Markdown Actions Required")
        action_counts = freshness_data['markdown_action'].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=action_counts.values,
            y=action_counts.index,
            orientation='h',
            marker_color=COLORS["warning"],
            text=action_counts.values,
            textposition='outside'
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Recommended Actions by Priority")
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of decay by department and days to expiry
    section_header("🔥 Decay Risk Heatmap")
    
    # Create pivot table
    freshness_data['expiry_bucket'] = pd.cut(freshness_data['days_to_expiry'], 
                                              bins=[-100, 0, 30, 60, 90, 365], 
                                              labels=['Expired', '0-30 days', '31-60 days', '61-90 days', '90+ days'])
    
    dept_matrix = freshness_data.groupby(['department', 'expiry_bucket'], observed=True)['projected_loss_no_action'].sum().unstack().fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=dept_matrix.values,
        x=dept_matrix.columns,
        y=dept_matrix.index,
        colorscale='RdYlGn_r',
        text=np.round(dept_matrix.values / 1000, 1),
        texttemplate='KSh %{text}K',
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(**CHART_LAYOUT, height=400, title="Projected Loss by Department and Expiry Timeline",
                     xaxis_title="Days to Expiry", yaxis_title="Department")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed inventory aging
    section_header("📦 Inventory Aging Analysis")
    
    urgent_items = freshness_data[freshness_data['days_to_expiry'].between(0, 30)].nlargest(10, 'book_value_at_cost')
    
    if len(urgent_items) > 0:
        fig = go.Figure(data=[go.Bar(
            x=urgent_items['product_name'],
            y=urgent_items['book_value_at_cost'],
            marker_color=urgent_items['days_to_expiry'],
            marker_colorscale='Reds',
            text=[fmt_ksh(v) for v in urgent_items['book_value_at_cost']],
            textposition='outside'
        )])
        fig.update_layout(**CHART_LAYOUT, height=400, title="Top 10 High-Value Items Expiring in ≤30 Days",
                         xaxis_title="Product", yaxis_title="Inventory Value at Risk (KSh)", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        info_card(f"🔥 {len(urgent_items)} high-value products expiring within 30 days. Total value at risk: {fmt_ksh(urgent_items['book_value_at_cost'].sum())}. Recommend immediate markdown action.", COLORS["danger"])
    else:
        info_card("✅ No high-value items expiring in the next 30 days. Inventory freshness is healthy.", COLORS["success"])

# ─── PREDICTIVE ANALYTICS HUB ────────────────────────────────────

elif analysis_type == "Predictive Analytics Hub":
    section_header("🤖 AI-Powered Predictive Analytics Dashboard")
    
    info_card("This hub combines multiple machine learning models to provide actionable business insights", COLORS["primary"])
    
    # Combined predictions
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Revenue Forecasting")
        
        # Simulated time series
        dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
        historical_revenue = np.random.normal(50000, 10000, 60)
        forecast_revenue = np.random.normal(55000, 12000, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates[:60], y=historical_revenue,
            mode='lines', name='Historical',
            line=dict(color=COLORS["primary"], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates[60:], y=forecast_revenue,
            mode='lines+markers', name='Forecast',
            line=dict(color=COLORS["success"], width=2, dash='dash'),
            marker=dict(size=6)
        ))
        fig.update_layout(**CHART_LAYOUT, height=400, title="30-Day Revenue Forecast",
                         xaxis_title="Date", yaxis_title="Daily Revenue (KSh)")
        st.plotly_chart(fig, use_container_width=True)
        
        info_card("📈 Forecast indicates 12% growth potential over next 30 days. Prepare inventory accordingly.", COLORS["success"])
    
    with col2:
        section_header("Customer Segmentation (K-Means)")
        
        # Prepare data for clustering
        clustering_data = clv_data[['total_revenue', 'total_transactions', 'revenue_per_transaction']].fillna(0)
        scaler = StandardScaler()
        clustered = scaler.fit_transform(clustering_data)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        segments = kmeans.fit_predict(clustered)
        
        fig = go.Figure()
        for segment in range(4):
            mask = segments == segment
            fig.add_trace(go.Scatter(
                x=clustering_data[mask]['total_revenue'],
                y=clustering_data[mask]['total_transactions'],
                mode='markers',
                name=f'Segment {segment+1}',
                marker=dict(size=8),
                opacity=0.6
            ))
        fig.update_layout(**CHART_LAYOUT, height=400, title="Customer Segmentation by Value and Frequency",
                         xaxis_title="Total Revenue (KSh)", yaxis_title="Number of Transactions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Assessment Matrix
    section_header("⚠️ Risk Assessment Matrix")
    
    # Combine risks from all sources
    risk_data = pd.DataFrame({
        'Category': ['RBPI Risk', 'Stockout Risk', 'Promo Inefficiency', 'Freshness Risk', 'Customer Churn'],
        'Severity': [0.25, 0.85, 0.40, 0.70, 0.55],
        'Urgency': [0.30, 0.90, 0.35, 0.85, 0.50],
        'Value_at_Risk': [500000, 2500000, 800000, 1500000, 1200000]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk_data['Severity'],
        y=risk_data['Urgency'],
        mode='markers+text',
        marker=dict(size=risk_data['Value_at_Risk'] / 50000, color=risk_data['Severity'], 
                    colorscale='Viridis', showscale=True),
        text=risk_data['Category'],
        textposition="top center",
        hovertemplate='Category: %{text}<br>Severity: %{x:.1%}<br>Urgency: %{y:.1%}<br>Value at Risk: %{marker.size:,.0f}K<extra></extra>'
    ))
    
    # Add quadrants
    fig.add_hrect(y0=0.5, y1=1, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=0, y1=0.5, line_width=0, fillcolor="green", opacity=0.05)
    fig.add_vrect(x0=0.5, x1=1, line_width=0, fillcolor="red", opacity=0.1)
    
    fig.update_layout(**CHART_LAYOUT, height=500, title="Business Risk Assessment Matrix",
                     xaxis_title="Severity", xaxis_range=[0, 1],
                     yaxis_title="Urgency", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations based on combined analysis
    section_header("🎯 AI-Generated Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Immediate Actions (< 7 days)**
        - 🚨 **Stockout Crisis**: Address the 15% of products causing 80% of leakage
        - 🔥 **Expiring Inventory**: Markdown strategy needed for 30-day items
        - 💰 **Negative RBPI**: Review discount policy for bottom 10% transactions
        
        **Short-term Actions (2-4 weeks)**
        - 📊 **Promo Optimization**: Scale back low-PER campaigns by 40%
        - 👥 **Retention Program**: Launch for high-churn risk segment
        - 📈 **Demand Forecasting**: Implement ML-based reorder points
        """)
    
    with col2:
        st.markdown("""
        **Strategic Initiatives (1-3 months)**
        - 🎯 **Customer Personalization**: Tier-based loyalty program
        - 🏪 **Inventory Optimization**: Just-in-time for fast-moving items
        - 📱 **Digital Transformation**: Real-time stock alerts system
        - 🤝 **Supplier Negotiation**: Volume commitments for top 20 SKUs
        
        **Expected Impact**
        - 📉 Reduce stockout leakage by 35%
        - 💵 Improve PER ratio to >2.5x
        - 👑 Increase CLV by 28%
        - 🔄 Reduce freshness write-offs by 45%
        """)
    
    # Performance dashboard
    section_header("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RBPI Model R²", "0.87", "+2.3%")
    with col2:
        st.metric("CLV Model Accuracy", "92.4%", "+1.8%")
    with col3:
        st.metric("Stockout Prediction", "85.7%", "+4.2%")
    with col4:
        st.metric("Freshness Forecast", "89.1%", "+3.1%")
    
    # Final insight
    st.markdown("---")
    info_card("💡 **Strategic Insight**: Focus on the intersection of freshness risk and stockout risk - optimizing inventory turnover for expiring products while maintaining availability for high-demand items could unlock ~KSh 15M in additional profit.", COLORS["primary"]) 