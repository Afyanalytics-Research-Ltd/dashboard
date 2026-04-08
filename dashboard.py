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
    # detecting a struggling store
    now = pd.Timestamp.now()
    recent = df[df['created_at'] >= now - pd.Timedelta(days=30)]
    prior  = df[(df['created_at'] >= now - pd.Timedelta(days=60)) &
                (df['created_at'] <  now - pd.Timedelta(days=30))]

    recent_sales = recent.groupby('store_id')['quantity_used'].sum().rename('recent_units')
    prior_sales  = prior.groupby('store_id')['quantity_used'].sum().rename('prior_units')

    store_health = df.groupby('store_id').agg(
        total_units=('quantity_used', 'sum'),
        transactions=('id', 'count'),
        last_activity=('created_at', 'max'),
    ).reset_index()

    store_health = store_health.merge(recent_sales, on='store_id', how='left')
    store_health = store_health.merge(prior_sales,  on='store_id', how='left')
    store_health['trend_pct'] = (
        (store_health['recent_units'] - store_health['prior_units'])
        / store_health['prior_units'].replace(0, np.nan)
    )
    store_health['days_inactive'] = (now - store_health['last_activity']).dt.days
    store_health['struggling'] = (
        (store_health['trend_pct'] < -0.20) |   # >20% decline
        (store_health['days_inactive'] > 14)     # no activity in 2 weeks
    )
    struggling_stores = store_health[store_health['struggling']]
    if not struggling_stores.empty:
        st.warning(f"{len(struggling_stores)} store(s) flagged as struggling")
        st.dataframe(struggling_stores[['store_id','trend_pct','days_inactive','transactions']])
    else:
        st.warning("No struggling stores at the moment")
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
    def load_and_train(n_trials=5):
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
    # --- Store-Aware Dynamic Pricing ---
    st.subheader("Store Health & Price Adjustment")

    adjustment_factor = 1.0
    store_msg = None

    if selected_store_id is not None:
        now = pd.Timestamp.now()
        recent = df[df['created_at'] >= now - pd.Timedelta(days=30)]
        prior  = df[(df['created_at'] >= now - pd.Timedelta(days=60)) &
                    (df['created_at'] <  now - pd.Timedelta(days=30))]

        recent_units = recent[recent['store_id'] == selected_store_id]['quantity_used'].sum()
        prior_units  = prior[prior['store_id'] == selected_store_id]['quantity_used'].sum()

        if prior_units > 0:
            trend_pct = (recent_units - prior_units) / prior_units
            if trend_pct < -0.20:
                # Scale discount proportionally to decline, cap at 20% off
                adjustment_factor = max(0.80, 1 + trend_pct * 0.5)
                store_msg = (
                    f"Store is **struggling** (sales {trend_pct:+.0%} vs prior 30 days). "
                    f"Applying a **{(1 - adjustment_factor):.0%} price reduction** to drive volume."
                )
            else:
                store_msg = f"Store is **healthy** (sales {trend_pct:+.0%} vs prior 30 days). No adjustment applied."
        else:
            store_msg = "Insufficient prior-period data to assess store health."
    else:
        store_msg = "Select a specific store to enable dynamic price adjustment."

    if adjustment_factor < 1.0:
        st.warning(store_msg)
    else:
        st.info(store_msg)

    st.caption(f"Active adjustment factor: **{adjustment_factor:.2f}x**")

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
            adjusted_price = round(result['predicted_price'] * adjustment_factor, 2)
            st.success(f"Results for: **{result['product']}**")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Base Predicted Price", f"{result['predicted_price']}")
            r2.metric("Adjusted Price", f"{adjusted_price}",
                    delta=f"{adjusted_price - result['predicted_price']:.2f}")
            r3.metric("Avg Cost", f"{result['cost']}")
            r4.metric("Daily Velocity", f"{result['velocity']}")
            r5.metric("Competitor Price", f"{result['competitor_price']}")
        else:
            st.error("Product not found")


    st.subheader("Pricing Data Preview")
    st.dataframe(pricing_df.head())

    st.divider()

    # --- Profit Simulation: Dynamic Pricing for Struggling Stores ---
    st.subheader("Profit Simulation: Dynamic Pricing Impact")
    st.caption(
        "Estimates 30-day profit change if adjusted prices are applied across all products. "
        "Volume uplift is modelled via price elasticity of demand."
    )

    sim_required = ['avg_cost', 'daily_velocity', 'current_price']
    if not all(c in pricing_df.columns for c in sim_required):
        st.warning(f"Simulation requires columns: {sim_required}. Not all found in pricing.csv.")
    else:
        elasticity = st.slider(
            "Price Elasticity of Demand",
            min_value=-3.0, max_value=-0.1, value=-1.5, step=0.1,
            help="How much % volume rises when price drops 1%. Pharmacy typically -1.0 to -2.0."
        )
        sim_horizon = st.number_input("Simulation horizon (days)", min_value=7, max_value=90, value=30)

        sim = pricing_df.dropna(subset=sim_required).copy()

        # Predict adjusted price for every product using the trained XGBoost model
        X_all = sim[features].fillna(0)
        sim['predicted_price'] = model.predict(X_all)
        sim['adjusted_price'] = (sim['predicted_price'] * adjustment_factor).round(2)

        # Price change %
        sim['price_chg_pct'] = (sim['adjusted_price'] - sim['current_price']) / sim['current_price'].replace(0, np.nan)

        # Volume uplift: lower price → higher demand (elasticity is negative)
        sim['volume_uplift'] = 1 + (elasticity * sim['price_chg_pct'])
        sim['volume_uplift'] = sim['volume_uplift'].clip(lower=0.5)  # floor: never below 50% of baseline

        # Units over sim horizon
        sim['baseline_units'] = sim['daily_velocity'] * sim_horizon
        sim['adjusted_units'] = sim['daily_velocity'] * sim['volume_uplift'] * sim_horizon

        # Revenue & profit
        sim['baseline_revenue'] = sim['current_price']  * sim['baseline_units']
        sim['adjusted_revenue'] = sim['adjusted_price'] * sim['adjusted_units']

        sim['baseline_profit']  = (sim['current_price']  - sim['avg_cost']) * sim['baseline_units']
        sim['adjusted_profit']  = (sim['adjusted_price'] - sim['avg_cost']) * sim['adjusted_units']

        sim['incremental_profit']  = sim['adjusted_profit']  - sim['baseline_profit']
        sim['incremental_revenue'] = sim['adjusted_revenue'] - sim['baseline_revenue']

        # --- KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Baseline Profit",   f"{sim['baseline_profit'].sum():,.0f}")
        k2.metric("Projected Profit",  f"{sim['adjusted_profit'].sum():,.0f}",
                  delta=f"{sim['incremental_profit'].sum():+,.0f}")
        k3.metric("Revenue Change",    f"{sim['incremental_revenue'].sum():+,.0f}")
        k4.metric("Avg Volume Uplift", f"{(sim['volume_uplift'].mean() - 1):+.1%}")

        # --- Charts ---
        label_col = 'name' if 'name' in sim.columns else 'product_id'
        top_sim = sim.nlargest(15, 'incremental_profit').copy()
        top_sim[label_col] = top_sim[label_col].astype(str).str[:20]

        fig_sim, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Incremental profit bar
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in top_sim['incremental_profit']]
        axes[0].barh(top_sim[label_col], top_sim['incremental_profit'], color=colors)
        axes[0].axvline(0, color='black', linewidth=0.8)
        axes[0].set_title(f"Incremental Profit – Top 15 Products ({sim_horizon}d)")
        axes[0].set_xlabel("Profit Change")

        # Baseline vs adjusted profit scatter
        axes[1].scatter(top_sim['baseline_profit'], top_sim['adjusted_profit'],
                        c=top_sim['incremental_profit'], cmap='RdYlGn', s=80,
                        edgecolors='k', linewidths=0.5)
        lim = max(top_sim['baseline_profit'].max(), top_sim['adjusted_profit'].max()) * 1.05
        axes[1].plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='Break-even line')
        axes[1].set_xlabel("Baseline Profit")
        axes[1].set_ylabel("Adjusted Profit")
        axes[1].set_title("Baseline vs Adjusted Profit per Product")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig_sim)

        with st.expander("Full Simulation Table"):
            st.dataframe(
                sim[[label_col, 'current_price', 'adjusted_price', 'price_chg_pct',
                     'volume_uplift', 'baseline_profit', 'adjusted_profit', 'incremental_profit']]
                .sort_values('incremental_profit', ascending=False)
            )

