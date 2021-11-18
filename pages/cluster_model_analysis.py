import streamlit as st
from pycaret.clustering import *


def write(state):
    
    if state.trained_model is not None:
        model = state.trained_model
        st.subheader("Analyzing Performance of Trained Machine Learning Model")
        with st.container():
            with st.expander("Show Training Performance Plots"):
                plot = st.selectbox('List of available plots', options=['cluster','tsne','elbow','silhouette','distance','distribution'])
                try:     
                    plot_model(model=model, plot=plot, display_format='streamlit')
                except:
                    st.error("Plot Not Available for the Trained Model.")
        return state
    else:
        st.error("Please Train a Model first!")
        