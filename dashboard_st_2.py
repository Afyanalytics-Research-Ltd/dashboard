import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Advanced Inventory Analytics Dashboard", layout="wide")

st.title("📦 Inventory Analytics")

# --- Load CSV ---
df = pd.read_csv(
    "/mnt/c/ProgramData/MySQL/MySQL Server 8.0/Uploads/evaluation_dispensing_details.csv",
    index_col=False
)

# --- Clean Data ---
df.columns = [
    'id','batch','product','prescription_id','quantity','price','discount','status',
    'invoiced','deleted_at','created_at','updated_at','previous_quantity','new_quantity',
]

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['previous_quantity'] = pd.to_numeric(df['previous_quantity'], errors='coerce')
df['new_quantity'] = pd.to_numeric(df['new_quantity'], errors='coerce')
df['quantity_used'] = df['previous_quantity'] - df['new_quantity']
df = df[df['quantity_used'] >= 0]

st.success("Data loaded and cleaned successfully!")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['created_at'].min(), df['created_at'].max()]
)

# store_list = df['store_id'].dropna().unique()
# selected_store = st.sidebar.selectbox("Select Store", ["All"] + sorted(store_list.tolist()))

# Apply filters
df_filtered = df[
    (df['created_at'] >= pd.to_datetime(date_range[0])) &
    (df['created_at'] <= pd.to_datetime(date_range[1]))
]

# if selected_store != "All":
    # df_filtered = df_filtered[df_filtered['store_id'] == selected_store]

# st.write(f"📍 Selected Store: {selected_store}")

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Units Used", f"{df_filtered['quantity_used'].sum():,.0f}")
col2.metric("Transactions", len(df_filtered))
col3.metric("Unique Products", df_filtered['product'].nunique())

st.divider()

# --- Fast Moving Products ---
st.subheader("🔥 Top Products by Usage")
fast_moving = (
    df_filtered.groupby('product')['quantity_used']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.dataframe(fast_moving.reset_index())

fig1, ax1 = plt.subplots()
fast_moving.plot(kind='bar', ax=ax1)
ax1.set_ylabel("Quantity Used")
ax1.set_title("Top 10 Fast Moving Products")
st.pyplot(fig1)

# --- Daily Consumption Trend ---
st.subheader("📈 Daily Consumption Trend")
daily_usage = df_filtered.groupby(pd.Grouper(key='created_at', freq='D'))['quantity_used'].sum()

fig2, ax2 = plt.subplots()
daily_usage.plot(ax=ax2)
ax2.set_ylabel("Quantity Used")
ax2.set_title("Daily Usage")
st.pyplot(fig2)

# --- Senior-Level Forecasting ---
st.subheader("🔮 Senior-Level Demand Forecast")

forecast_products = st.multiselect(
    "Select products to forecast (multi-select)",
    options=df_filtered['product'].unique(),
    default=df_filtered['product'].unique()[:3]
)

forecast_horizon = st.number_input("Forecast Horizon (days)", min_value=1, max_value=90, value=14)

for product in forecast_products:
    st.markdown(f"### Product: {product}")
    prod_df = df_filtered[df_filtered['product'] == product]
    
    # Aggregate daily usage
    ts = prod_df.groupby('created_at')['quantity_used'].sum().reset_index()
    ts = ts.rename(columns={'created_at':'ds','quantity_used':'y'})
    
    if len(ts) < 10:
        st.warning(f"Not enough data to forecast for {product}")
        continue

    # Prophet model
    m = Prophet(interval_width=0.95, daily_seasonality=True)
    m.fit(ts)

    future = m.make_future_dataframe(periods=forecast_horizon)
    forecast = m.predict(future)

    # Plot forecast
    fig3, ax3 = plt.subplots()
    ax3.plot(ts['ds'], ts['y'], label="Actual")
    ax3.plot(forecast['ds'], forecast['yhat'], label="Forecast")
    ax3.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
    ax3.set_title(f"Forecast for {product}")
    ax3.set_ylabel("Quantity Used")
    ax3.legend()
    st.pyplot(fig3)

    # Show forecast table
    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon))

# --- Anomaly Detection / Stock Risk ---
st.subheader("⚠️ Product Usage Anomalies / Stock Risk")

if len(df_filtered) > 20:
    # Use Isolation Forest on daily product usage
    daily_product_usage = df_filtered.groupby(['product','created_at'])['quantity_used'].sum().reset_index()
    model = IsolationForest(contamination=0.05, random_state=42)
    daily_product_usage['quantity_used_filled'] = daily_product_usage['quantity_used'].fillna(0)
    daily_product_usage['anomaly'] = model.fit_predict(daily_product_usage[['quantity_used_filled']])
    
    anomalies = daily_product_usage[daily_product_usage['anomaly'] == -1]
    
    st.write(f"Detected {len(anomalies)} anomalous records")
    st.dataframe(anomalies[['product','created_at','quantity_used']].sort_values(by='created_at', ascending=False))
else:
    st.warning("Not enough data for anomaly detection")

# --- Raw Data Preview ---
st.subheader("📄 Raw Data Preview")
st.dataframe(df_filtered.head(100))