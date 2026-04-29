import streamlit as st
import pandas as pd

st.title("Analytics Dashboard")

df = pd.DataFrame({
    "a": [1,2,3],
    "b": [10,20,30]
})

st.dataframe(df)

st.markdown("[🏠 Cross Sell Inventory ](http://127.0.0.1:8000/dashboards/dashboards/{slug}/)".format(slug='cross_sell_inventory_dashboard'), unsafe_allow_html=True)
st.markdown("[🏠 Cross Sell Inventory ](http://127.0.0.1:8000/dashboards/dashboards/{slug}/)".format(slug='predictions_dashboard'), unsafe_allow_html=True)
st.markdown("[🏠 Cross Sell Inventory ](http://127.0.0.1:8000/dashboards/dashboards/{slug}/)".format(slug='cross_sell_inventory_dashboard'), unsafe_allow_html=True)