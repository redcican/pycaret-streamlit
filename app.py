import streamlit as st
from pages import (home,data_info, reg_preprocessing,prediction, reg_training, 
                   reg_model_analysis,cls_preprocessing,cls_training,cls_model_analysis,
                   cluster_preprocessing,cluster_training,cluster_model_analysis, backward_analysis)
# from utils.session import _get_state
from pathlib import Path
from utils.image_loader import *

    
PAGES = {
    "Home": home,
    "DataInfo": data_info,
    "Preprocessing": (reg_preprocessing,cls_preprocessing,cluster_preprocessing),
    "Training" : (reg_training, cls_training,cluster_training),
    "Model Analysis": (reg_model_analysis, cls_model_analysis,cluster_model_analysis),
    "Prediction and Save": prediction,
    "Backward Analysis": backward_analysis,
}

IMAGE_FOLDER = Path("images/")

def run():
    # state = _get_state()
    state = st.session_state
    st.set_page_config(
        page_title="EidoData App",
        page_icon=':shark:',
        layout="centered",
        initial_sidebar_state='expanded'
    )
    load_nav_image(IMAGE_FOLDER/'EIDOlogo.png')
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection == "Home":
        load_header_image(IMAGE_FOLDER/'EIDOname.png')
        try:
            state_df,task = PAGES[selection].write(state)
            state.df, state.task = state_df,task
            
            state.log_history = {}
            state.is_remove = False
            state.ignore_columns = []
        except:
            st.header("Please Upload Csv or Excel File first!")
            st.stop()
    if selection == "DataInfo":
        PAGES[selection].write(state.df)
    
    if selection == "Preprocessing":
        if state.task == "Regression":
            PAGES[selection][0].write(state)
        elif state.task =="Classification":
            PAGES[selection][1].write(state)
        else:
            PAGES[selection][2].write(state)

    if selection == "Training":
        if state.task == "Regression":
            PAGES[selection][0].write(state)
        elif state.task =="Classification":
            PAGES[selection][1].write(state)
        else:
            PAGES[selection][2].write(state)
    if selection == "Model Analysis":
        if state.task == "Regression":
            PAGES[selection][0].write(state)
        elif state.task =="Classification":
            PAGES[selection][1].write(state)
        else:
            PAGES[selection][2].write(state)
    if selection == "Prediction and Save":
        PAGES[selection].write(state)
        
    if selection == "Backward Analysis":
        if state.task == "Regression":
            PAGES[selection].write(state)
        else:
            st.header("Only Support for Regression Task!")

    # state.sync()

if __name__ == '__main__':
    run()