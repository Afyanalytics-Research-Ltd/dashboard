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




# detecting a struggling store
pass


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
    import optuna
    import warnings
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    @st.cache_resource
    def load_and_train(n_trials=30):
        pricing_df = pd.read_csv('pricing.csv')

        # --- Feature Engineering ---
        if 'current_price' in pricing_df.columns:
            pricing_df['margin_pct'] = (
                (pricing_df['current_price'] - pricing_df['avg_cost'])
                / pricing_df['avg_cost'].replace(0, np.nan)
            )
            pricing_df['price_to_comp_ratio'] = (
                pricing_df['current_price']
                / pricing_df['effective_competitor_price'].replace(0, np.nan)
            )

        base_features = ['avg_cost', 'daily_velocity', 'effective_competitor_price']
        extra_features = ['margin_pct', 'price_to_comp_ratio']
        features = base_features + [f for f in extra_features if f in pricing_df.columns]

        ml_df = pricing_df.dropna(subset=features + ['target_price']).copy()
        X = ml_df[features]
        y = ml_df['target_price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Optuna Hyperparameter Tuning ---
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'n_jobs': -1,
            }
            mdl = XGBRegressor(**params)
            score = cross_val_score(mdl, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_mae': -cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error').mean(),
        }

        importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)

        return best_model, pricing_df, features, metrics, importances, study.best_params

    st.title("Dynamic Pharmacy Pricing Model")

    n_trials = st.sidebar.slider("Optuna tuning trials", min_value=10, max_value=100, value=30, step=10)

    with st.spinner("Training model with hyperparameter tuning..."):
        model, pricing_df, features, metrics, importances, best_params = load_and_train(n_trials)

    # --- Model Performance ---
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (test)", f"{metrics['mae']:.2f}")
    m2.metric("R² (test)", f"{metrics['r2']:.3f}")
    m3.metric("CV MAE (5-fold)", f"{metrics['cv_mae']:.2f}")

    with st.expander("Best hyperparameters"):
        st.json(best_params)

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index, ax=ax_imp)
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("XGBoost Feature Importances")
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.divider()

    # --- Price Prediction ---
    def predict_price(query, df, mdl, feature_cols):
        # Search by name (partial, case-insensitive) or product_id
        if 'name' in df.columns:
            match = df[df['name'].str.contains(query, case=False, na=False)]
        else:
            try:
                match = df[df['product_id'] == int(query)]
            except ValueError:
                return None

        if match.empty:
            return None

        row = match.iloc[0]
        X_input = pd.DataFrame([{f: row.get(f, 0) or 0 for f in feature_cols}])
        predicted_price = mdl.predict(X_input)[0]

        return {
            "product": row.get('name', row.get('product_id', 'N/A')),
            "predicted_price": round(float(predicted_price), 2),
            "cost": row.get('avg_cost'),
            "velocity": row.get('daily_velocity'),
            "competitor_price": row.get('effective_competitor_price'),
        }

    product_input = st.text_input("Search product by name or ID")

    if product_input:
        result = predict_price(product_input, pricing_df, model, features)
        if result:
            st.success(f"Results for: **{result['product']}**")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Price", f"{result['predicted_price']}")
            r2.metric("Avg Cost", f"{result['cost']}")
            r3.metric("Daily Velocity", f"{result['velocity']}")
            r4.metric("Competitor Price", f"{result['competitor_price']}")
        else:
            st.error("Product not found")

    st.subheader("Pricing Data Preview")
    st.dataframe(pricing_df.head())

