import streamlit as st
import pandas as pd


def write(state):
    with st.spinner("Loading Home ..."):
        state = st.file_uploader('Upload csv file for project', type=["csv", "xlsx"])
        
        if state is not None:
            file_extension = state.name.split('.')[1]
            if file_extension == "csv":
                state.df = pd.read_csv(state)
            else:
                state.df = pd.read_excel(state)
            st.header("The First 20 Rows of Data")
            st.write(state.df.head(20))
            
            return state.df
        else:
            st.header("Please upload csv or excel file first!")
            
        

