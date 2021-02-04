import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

@st.cache(suppress_st_warning=True)
def write(state_df):
    st.header("Data Explotary Analysis")
    
    with st.spinner("Loading Data Info ..."):
        if state_df is not None:
            pr = ProfileReport(state_df, explorative=True,minimal=True)
            st_profile_report(pr)
        else:
            st.error("Please upload dataset first!")
            