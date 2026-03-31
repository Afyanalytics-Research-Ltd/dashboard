import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import os

# --- DATABASE CONNECTION ---
DB_USER = os.getenv('DB_USER').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD').strip()
DB_HOST = os.getenv('DB_HOST').strip()
DB_PORT = os.getenv('DB_PORT').strip()
DB_NAME = os.getenv('DB_NAME').strip()

db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url, echo=False)

st.set_page_config(page_title="Enterprise Inventory Dashboard", layout="wide")
st.title("🏥 Inventory Analytics & Predictive Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Get stores matching 'pharm'
stores_df = pd.read_sql(
    "SELECT id, name FROM inventory_stores WHERE LOWER(name) LIKE '%%pharm%%'", 
    engine
)
store_options = ["All Stores"] + stores_df['name'].tolist()
selected_store_name = st.sidebar.selectbox("Select Store", store_options)

# Map store name to store_id
if selected_store_name != "All Stores":
    selected_store_id = stores_df.loc[stores_df['name']==selected_store_name, 'id'].values[0]
else:
    selected_store_id = None

# --- Load Inventory Usage Data ---
query_usage = "SELECT * FROM inventory_evaluation_dispensing_details"
df = pd.read_sql(query_usage, engine)

# --- Clean Data ---
expected_cols = ['id','batch','product','prescription_id','quantity','price','discount','status',
                 'invoiced','deleted_at','created_at','updated_at','previous_quantity','new_quantity','store_id']

if 'store_id' not in df.columns:
    df['store_id'] = 'All Stores'

df.columns = expected_cols
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['previous_quantity'] = pd.to_numeric(df['previous_quantity'], errors='coerce')
df['new_quantity'] = pd.to_numeric(df['new_quantity'], errors='coerce')
df['quantity_used'] = df['previous_quantity'] - df['new_quantity']
df = df[df['quantity_used'] >= 0]

# Apply store filter
if selected_store_id:
    df_filtered = df[df['store_id'] == selected_store_id]
else:
    df_filtered = df.copy()

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Units Used", f"{df_filtered['quantity_used'].sum():,.0f}")
col2.metric("Transactions", len(df_filtered))
col3.metric("Unique Products", df_filtered['product'].nunique())

st.divider()

# --- Fast / Slow Movers ---
st.subheader("🔥 Fast Moving Products")
fast_moving = df_filtered.groupby('product')['quantity_used'].sum().sort_values(ascending=False).head(10)
st.dataframe(fast_moving.reset_index())

st.subheader("🐢 Slow Moving Products")
last_activity = df_filtered.groupby('product')['created_at'].max().reset_index()
last_activity['days_since_last_use'] = (pd.Timestamp.now() - last_activity['created_at']).dt.days
slow_moving = last_activity.sort_values(by='days_since_last_use', ascending=False).head(10)
st.dataframe(slow_moving)

# --- Senior-Level Forecasting ---
st.subheader("🔮 Demand Forecasting (Prophet)")
forecast_products = st.multiselect(
    "Select Products to Forecast", options=df_filtered['product'].unique(), default=df_filtered['product'].unique()[:3]
)
forecast_horizon = st.number_input("Forecast Horizon (days)", min_value=1, max_value=90, value=14)

for product in forecast_products:
    prod_df = df_filtered[df_filtered['product'] == product]
    ts = prod_df.groupby('created_at')['quantity_used'].sum().reset_index()
    ts = ts.rename(columns={'created_at':'ds','quantity_used':'y'})
    if len(ts) < 10:
        st.warning(f"Not enough data to forecast {product}")
        continue

    m = Prophet(interval_width=0.95, daily_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=forecast_horizon)
    forecast = m.predict(future)

    fig, ax = plt.subplots()
    ax.plot(ts['ds'], ts['y'], label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
    ax.set_title(f"Forecast for {product}")
    ax.set_ylabel("Quantity Used")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon))

# --- Anomaly Detection ---
st.subheader("⚠️ Anomaly Detection")
daily_usage = df_filtered.groupby(['product','created_at'])['quantity_used'].sum().reset_index()
daily_usage['quantity_filled'] = daily_usage['quantity_used'].fillna(0)

if len(daily_usage) > 20:
    iso = IsolationForest(contamination=0.05, random_state=42)
    daily_usage['anomaly'] = iso.fit_predict(daily_usage[['quantity_filled']])
    anomalies = daily_usage[daily_usage['anomaly'] == -1]
    st.write(f"Detected {len(anomalies)} anomalies")
    st.dataframe(anomalies[['product','created_at','quantity_used']])
else:
    st.warning("Not enough data for anomaly detection")

# --- Suggested Products based on other stores ---
st.subheader("💡 Suggested Products (Based on Similar Stores)")

if selected_store_id:
    query_suggested =  f"""
        WITH ranking_table AS (
        SELECT
            similar.store_id,
            COUNT(*) AS rank_count
        FROM inventory_store_products target
        JOIN inventory_store_products similar
            ON target.product_id = similar.product_id
            AND target.store_id != similar.store_id
        JOIN inventory_stores s
            ON similar.store_id = s.id
        WHERE
            target.store_id = {selected_store_id}
            AND LOWER(s.name) LIKE '%%pharm%%'
        GROUP BY similar.store_id
        )
        SELECT
        isp.product_id,
        ip.name AS product_name,
        SUM(rt.rank_count) AS total_rank
        FROM ranking_table rt
        JOIN inventory_store_products isp
        ON rt.store_id = isp.store_id
        JOIN inventory_products ip
        ON isp.product_id = ip.id
        LEFT JOIN inventory_store_products target
        ON target.store_id = {selected_store_id}
        AND target.product_id = isp.product_id
        WHERE
        target.product_id IS NULL
        AND ip.name IS NOT NULL
        GROUP BY isp.product_id, ip.name
        ORDER BY total_rank DESC
        LIMIT 20;
        """
    suggested_df = pd.read_sql(query_suggested, engine)
    st.dataframe(suggested_df)