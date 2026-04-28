import streamlit as st
import pandas as pd

st.title("Analytics Dashboard")

df = pd.DataFrame({
    "a": [1,2,3],
    "b": [10,20,30]
})

st.dataframe(df)