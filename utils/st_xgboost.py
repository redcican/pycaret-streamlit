import streamlit as st
from pycaret.regression import create_model

@st.cache(hash_funcs={'xgboost.sklearn.XGBRegressor':lambda _: None},suppress_st_warning=True)
def train_xgboost_regression(fold, cross_validation):
    create_model('xgboost',fold=fold, cross_validation=cross_validation)