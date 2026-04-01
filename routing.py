import streamlit as st

def page_2():
    st.title("Page 2")

pg = st.navigation(
    [ 
        "dashboard_st_2.py",
        "dashboard_st_3.py",
        "dashboard.py",
        "dashboard_st.py",   
        page_2
    ]
)

pg.run()