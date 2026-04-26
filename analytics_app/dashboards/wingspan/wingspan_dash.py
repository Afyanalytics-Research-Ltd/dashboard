import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("healthinsurance.csv")
    return df

df = load_data()

# -----------------------------
# CLEANING
# -----------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

date_cols = ["adm_date", "dis_date", "claim_date"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Compute LOS if missing
df["los"] = (df["dis_date"] - df["adm_date"]).dt.days

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("🔎 Filters")

county = st.sidebar.multiselect("County", df["county_name"].dropna().unique())
disease = st.sidebar.multiselect("Disease", df["disease_name"].dropna().unique())
plan = st.sidebar.multiselect("Plan Tier", df["plan_tier"].dropna().unique())

filtered_df = df.copy()

if county:
    filtered_df = filtered_df[filtered_df["county_name"].isin(county)]

if disease:
    filtered_df = filtered_df[filtered_df["disease_name"].isin(disease)]

if plan:
    filtered_df = filtered_df[filtered_df["plan_tier"].isin(plan)]

# -----------------------------
# TITLE
# -----------------------------
st.title("🏥 Insurance Claims Analytics (Plotly Dashboard)")

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Claims", len(filtered_df))
c2.metric("Total Cost", f"{filtered_df['claim_total'].sum():,.0f}")
c3.metric("Avg Claim", f"{filtered_df['claim_total'].mean():,.0f}")
c4.metric("Avg LOS", f"{filtered_df['los'].mean():.2f} days")

# -----------------------------
# 📈 TIME SERIES (ENHANCED)
# -----------------------------
st.subheader("📈 Claims Trend Over Time")

ts = filtered_df.groupby("claim_date")["claim_total"].sum().reset_index()

fig = px.line(
    ts,
    x="claim_date",
    y="claim_total",
    markers=True,
    title="Claims Over Time"
)

fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🌍 HEATMAP (County vs Disease)
# -----------------------------
st.subheader("🌍 Cost Heatmap (County vs Disease)")

heat = filtered_df.pivot_table(
    index="county_name",
    columns="disease_name",
    values="claim_total",
    aggfunc="sum"
).fillna(0)

fig = px.imshow(
    heat,
    aspect="auto",
    title="Claim Cost Heatmap"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🦠 TOP DISEASES
# -----------------------------
st.subheader("🦠 Top Diseases by Cost")

disease_df = (
    filtered_df.groupby("disease_name")["claim_total"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    disease_df,
    x="disease_name",
    y="claim_total",
    color="claim_total",
    title="Top Diseases"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 💰 PLAN ANALYSIS
# -----------------------------
st.subheader("💰 Plan Tier Performance")

plan_df = filtered_df.groupby("plan_tier").agg({
    "claim_total": "mean",
    "loss_ratio": "mean"
}).reset_index()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(x=plan_df["plan_tier"], y=plan_df["claim_total"], name="Avg Claim"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=plan_df["plan_tier"], y=plan_df["loss_ratio"], name="Loss Ratio"),
    secondary_y=True,
)

fig.update_layout(title="Pricing vs Risk by Plan Tier")

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ⚠️ FRAUD ANALYSIS
# -----------------------------
st.subheader("⚠️ Fraud vs Claim Behavior")

fig = px.scatter(
    filtered_df,
    x="claim_total",
    y="fraud_probability",
    color="plan_tier",
    size="total_bill",
    hover_data=["disease_name"],
    title="Fraud Probability vs Claim"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 📊 DISTRIBUTIONS
# -----------------------------
st.subheader("📊 Claim Distribution")

fig = px.histogram(
    filtered_df,
    x="claim_total",
    nbins=30,
    marginal="box",
    title="Claim Amount Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🔮 PREDICTION MODEL
# -----------------------------
st.subheader("🔮 Claim Prediction")

features = ["total_bill", "coverage_limit_kes", "deductible_pct", "family_size"]
features = [f for f in features if f in filtered_df.columns]

model_df = filtered_df.dropna(subset=features + ["claim_total"])

if len(model_df) > 10:
    X = model_df[features]
    y = model_df["claim_total"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    st.write(f"Model Accuracy (R²): {score:.2f}")

    st.write("### Input Values")

    user_inputs = {}
    for f in features:
        user_inputs[f] = st.number_input(f, value=float(X[f].mean()))

    if st.button("Predict Claim"):
        prediction = model.predict([list(user_inputs.values())])[0]
        st.success(f"Predicted Claim Amount: {prediction:,.2f}")

# -----------------------------
# 📄 RAW DATA
# -----------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df)