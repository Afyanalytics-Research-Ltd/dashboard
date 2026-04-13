import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import os
sns.set_style("whitegrid")

# --- DATABASE CONNECTION ---
DB_USER = os.getenv('DB_USER').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD').strip()
DB_HOST = os.getenv('DB_HOST').strip()
DB_PORT = os.getenv('DB_PORT').strip()
DB_NAME = os.getenv('DB_NAME').strip()

db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url, echo=False)
st.image("logo_pharmaplus.png", width=150)
st.set_page_config(page_title="Enterprise Dashboard", layout="wide")
st.title("Revenue Analytics & Predictive Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Product Health", "Recommendation Engine", "Pricing Engine"])
# --- Sidebar Filters ---
st.sidebar.header("Filters")

with tab1:
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
    query_usage = "select ied.*,ip.name as product_name from inventory_evaluation_dispensing_details ied left join inventory_products ip on ied.product=ip.id"
    df = pd.read_sql(query_usage, engine)

    # --- Clean Data ---
    expected_cols = ['id','batch','product','prescription_id','quantity','price','discount','status',
                    'invoiced','deleted_at','created_at','updated_at','previous_quantity','new_quantity','product_name','store_id']

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
    # Top N
    fast_moving = df_filtered.groupby(['product'])['quantity_used'].sum().sort_values(ascending=False).head(10)
    fast_moving = fast_moving.reset_index()

    TOP_N = 10

    df_top = fast_moving.head(TOP_N).copy()

    # Cumulative %
    df_top['cum_pct'] = df_top['quantity_used'].cumsum() / df_top['quantity_used'].sum()

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Barplot (Seaborn)
    sns.barplot(
        data=df_top,
        x='product',
        y='quantity_used',
        ax=ax1
    )

    # Rotate labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Titles
    ax1.set_title("Top Products Driving Pharmacy Demand", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Quantity Used")
    ax1.set_xlabel("")

    # --- Cumulative Line ---
    ax2 = ax1.twinx()
    ax2.plot(df_top['cum_pct'], marker='o')
    ax2.set_ylabel("Cumulative %")

    # Add % labels
    for i, v in enumerate(df_top['cum_pct']):
        ax2.text(i, v, f"{v:.0%}", ha='center', fontsize=9)

    # Clean spines
    sns.despine(left=False, bottom=False)

    plt.tight_layout()

    # Streamlit
    st.pyplot(fig)
    st.subheader("Fast Moving Products")
    st.dataframe(fast_moving.reset_index())
    st.subheader("Slow Moving Products")
    last_activity = df_filtered.groupby(['product'])['created_at'].max().reset_index()
    last_activity['days_since_last_use'] = (pd.Timestamp.now() - last_activity['created_at']).dt.days
    slow_moving = last_activity.sort_values(by='days_since_last_use', ascending=False).head(10)
    st.dataframe(slow_moving)

with tab2:
    # --- Senior-Level Forecasting ---
    st.subheader("Demand Forecasting (Prophet)")
    forecast_products = st.multiselect(
        "Select Products to Forecast", options=df_filtered['product'].unique(), default=df_filtered['product'].unique()[:1]
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
        ax.set_xticklabels(ts['ds'], rotation=45, ha='right')  # ha='right' aligns text nicely
        ax.set_ylabel("Quantity Used")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon))

    # --- Anomaly Detection ---
    st.subheader("Anomaly Detection")
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

with tab3:
    # --- Suggested Products based on other stores ---
    st.subheader("Suggested Products (Based on Similar Stores)")

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

with tab4:
    # cost_df = pd.read_csv('cost.csv')

    # demand_df = pd.read_sql("""
    # SELECT 
    #     product as product_id,
    #     SUM(quantity) AS total_qty,
    #     SUM(quantity) / COUNT(DISTINCT DATE(created_at)) AS daily_velocity
    # FROM (
    #     SELECT product, quantity, created_at FROM inventory_evaluation_dispensing_details
    #     UNION ALL
    #     SELECT store_product_id , quantity, created_at FROM evaluation_pos_sale_details
    #     UNION ALL
    #     SELECT item_id, units, created_at FROM finance_invoice_items
    # ) t
    # GROUP BY product_id
    # LIMIT 18446744073709551615 OFFSET 1
    # """,engine)

    # products_df = pd.read_sql("""
    # SELECT p.id as product_id, p.name, p.category, s.selling_price as current_price, s.insurance_price as original_price
    # FROM inventory_products p
    # left join inventory_store_products s on s.product_id  = p.id
    # """, engine))
    # comp_df = pd.read_csv("competitors.csv")

    # def simple_match(product_name, comp_df):
    #     for _, row in comp_df.iterrows():
    #         if row['name'].lower() in product_name.lower() or product_name.lower() in row['name'].lower():
    #             return row['current_price'], row['original_price']
        
    #     return None, None
    # products_df[['competitor_price', 'competitor_original_price']] = products_df['name'].apply(
    #     lambda x: pd.Series(simple_match(x, comp_df))
    # )
    # def get_effective_competitor_price(row):
    #     # if not pd.isna(row['competitor_price']):
    #     #     return row['competitor_price']
        
    #     # fallback → use your own pricing logic
    #     if not pd.isna(row.get('current_price')):
    #         return (row['current_price'] + row['original_price'])/2
        
    #     if not pd.isna(row.get('original_price')):
    #         return row['original_price']
        
    #     return None
    # products_df['effective_competitor_price'] = products_df.apply(get_effective_competitor_price, axis=1)
    # df = products_df.merge(cost_df, on='product_id', how='left')
    # df = df.merge(demand_df, on='product_id', how='left')
    # df = df[['product_id', 'name', 'avg_cost', 'daily_velocity', 'effective_competitor_price']]
    pricing_df = pd.read_csv('pricing.csv')
    
    def compute_target_price(row):
        # --- 1. Get inputs ---
        cost = row['avg_cost']
        current_price = row.get('current_price', np.nan)
        comp_price = row.get('effective_competitor_price', np.nan)
        velocity = row.get('daily_velocity', 0)

        # --- 2. FIX COST (auto unit correction) ---
        # If cost is too small vs price → scale it up
        if not pd.isna(current_price) and not pd.isna(cost):
            if cost < current_price * 0.2:
                cost = current_price * 0.4   # assume 40% cost baseline

        # fallback if cost missing
        if pd.isna(cost):
            if not pd.isna(current_price):
                cost = current_price * 0.4
            else:
                return None

        # --- 3. DEMAND ADJUSTMENT ---
        if pd.isna(velocity):
            velocity = 0

        if velocity > 20:
            demand_factor = 1.2
        elif velocity < 5:
            demand_factor = 0.9
        else:
            demand_factor = 1.0

        # --- 4. BASE PRICE ---
        base_price = cost * 1.3 * demand_factor

        # --- 5. COMPETITOR CONSTRAINT ---
        if not pd.isna(comp_price):

            lower_bound = cost * 1.1           # protect margin
            upper_bound = comp_price * 1.05    # stay competitive

            final_price = np.clip(base_price, lower_bound, upper_bound)

        else:
            # --- 6. SELF-COMPETE (fallback) ---
            if not pd.isna(current_price):
                lower_bound = cost * 1.1
                upper_bound = current_price * 1.1

                final_price = np.clip(base_price, lower_bound, upper_bound)
            else:
                final_price = base_price

        return round(final_price, 2)
    # pricing_df['target_price'] = pricing_df.apply(compute_target_price, axis=1)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    features = ['avg_cost', 'daily_velocity', 'effective_competitor_price']
    
    # keep rows where ALL features and target are valid
    ml_df = pricing_df.dropna(subset=features + ['target_price'])
    ml_df.shape
    X = ml_df[features]
    y = ml_df['target_price']
    # X = pricing_df[features].fillna(0)
    # y = pricing_df['target_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    def predict_price(product, df, model):
        # safe filtering
        type(product)
        print(df.info())
        product = df[df['product_id'] == int(product)]

        if product.empty:
            return None

        row = product.iloc[0]

        X_input = pd.DataFrame([{
            'avg_cost': row['avg_cost'] or 0,
            'daily_velocity': row['daily_velocity'] or 0,
            'effective_competitor_price': row['effective_competitor_price'] or 0
        }])

        predicted_price = model.predict(X_input)[0]

        return {
            "predicted_price": round(predicted_price, 2),
            "cost": row['avg_cost'],
            "velocity": row['daily_velocity'],
            "competitor_price": row['effective_competitor_price']
        }

    st.title("Dynamic Pharmacy Pricing Model")

    product_input = st.text_input("Enter product name")

    if product_input:
        result = predict_price(product_input, pricing_df, model)

        if result:
            st.metric("Predicted Price", result['predicted_price'])
            st.write(f"Cost: {result['cost']}")
            st.write(f"Demand (velocity): {result['velocity']}")
            st.write(f"Competitor Price: {result['competitor_price']}")
        else:
            st.error("Product not found")
    st.dataframe(pricing_df.head())

