import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Inventory Analytics Dashboard", layout="wide")

st.title("📦 Inventory Dispensing Analytics")

# --- File Upload ---
df = pd.read_csv("/mnt/c/ProgramData/MySQL/MySQL Server 8.0/Uploads/evaluation_dispensing_details.csv",index_col=False)
# --- Data Cleaning ---
df.columns = ['id','batch','product','prescription_id','quantity','price','discount','status','invoiced','deleted_at','created_at','updated_at','previous_quantity','new_quantity']
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
# Convert both columns to numeric, turning non-numbers to NaN
df['previous_quantity'] = pd.to_numeric(df['previous_quantity'], errors='coerce')
df['new_quantity'] = pd.to_numeric(df['new_quantity'], errors='coerce')

# Now calculate
df['quantity_used'] = df['previous_quantity'] - df['new_quantity']

# Remove invalid rows
df = df[df['quantity_used'] >= 0]

st.success("Data loaded successfully!")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['created_at'].min(), df['created_at'].max()]
)

df_filtered = df[
    (df['created_at'] >= pd.to_datetime(date_range[0])) &
    (df['created_at'] <= pd.to_datetime(date_range[1]))
]

# --- KPIs ---
col1, col2, col3 = st.columns(3)

total_usage = df_filtered['quantity_used'].sum()
total_transactions = len(df_filtered)
unique_products = df_filtered['product'].nunique()

col1.metric("Total Units Used", f"{total_usage:,.0f}")
col2.metric("Transactions", total_transactions)
col3.metric("Unique Products", unique_products)

st.divider()

# --- Fast Moving Products ---
st.subheader("🔥 Fast Moving Products")

fast_moving = (
    df_filtered.groupby('product')['quantity_used']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(fast_moving.reset_index())

fig1, ax1 = plt.subplots()
fast_moving.plot(kind='bar', ax=ax1)
ax1.set_title("Top 10 Fast Moving Products")
ax1.set_ylabel("Quantity Used")
st.pyplot(fig1)

# --- Daily Consumption Trend ---
st.subheader("📈 Daily Consumption Trend")

daily_usage = (
    df_filtered.groupby(pd.Grouper(key='created_at', freq='D'))['quantity_used']
    .sum()
)

fig2, ax2 = plt.subplots()
daily_usage.plot(ax=ax2)
ax2.set_title("Daily Usage")
ax2.set_ylabel("Quantity Used")
st.pyplot(fig2)

# --- Product Velocity ---
st.subheader("⚡ Product Velocity")

velocity = (
    df_filtered.groupby('product')
    .agg(
        total_used=('quantity_used', 'sum'),
        active_days=('created_at', lambda x: x.dt.date.nunique())
    )
)

velocity['daily_velocity'] = velocity['total_used'] / velocity['active_days']
velocity = velocity.sort_values(by='daily_velocity', ascending=False)

st.dataframe(velocity.head(10).reset_index())

# --- Slow Moving / Dead Stock ---
st.subheader("🐢 Slow Moving Products")

last_activity = (
    df_filtered.groupby('product')['created_at']
    .max()
    .reset_index()
)

last_activity['days_since_last_use'] = (
    pd.Timestamp.now() - last_activity['created_at']
).dt.days

slow_moving = last_activity.sort_values(by='days_since_last_use', ascending=False)

st.dataframe(slow_moving.head(10))

# --- Raw Data ---
st.subheader("📄 Raw Data Preview")
st.dataframe(df_filtered.head(100))

