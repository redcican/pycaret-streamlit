import streamlit as st
from pycaret.regression import *
from utils.plot_regression import get_plotly_act_vs_predict
from utils.plot_shap import plot_reg_shap_global_and_local


def write(state):
    st.subheader("Analyzing Performance of Trained Machine Learning Model")
    state.X_train = get_config('X_train')
    state.X_test = get_config('X_test')
    state.y_train = get_config('y_train')
    state.y_test = get_config('y_test')
    
    
    if state.trained_model is not None:
        model = state.trained_model
        X_train = state.X_train
        X_test = state.X_test
        y_train = state.y_train
        y_test = state.y_test

        with st.container():
            with st.expander("Show Actual vs. Prediction Plot"):
                # plot actual vs prediction plot 
                act_vs_pred_plot = get_plotly_act_vs_predict(model,X_train, X_test, y_train, y_test) 
                st.plotly_chart(act_vs_pred_plot, use_container_width=True)
            
            with st.expander("Show Training Performance Plots"):
                # plot all pycaret support diagrams
                plot = st.selectbox('Select a Plot', options=['residuals','error','cooks','rfe','learning','vc','manifold'])
                try:     
                    plot_model(estimator=model, plot=plot, display_format='streamlit')
                except:
                    st.error("Plot Not Available for multiclass problems.")    
                 

            if state.transform_target:
                st.markdown('<p style="color:#f42756">SHAP Value is not supported for Model Transform Target</p>',unsafe_allow_html=True)       
            elif state.is_ensemble:
                st.markdown('<p style="color:#f42756">SHAP Value is not supported for Ensemble Model</p>',unsafe_allow_html=True)       
            else:
                st.markdown("---")
                is_shap = st.checkbox("Do You Want to Check SHAP Value?", value=False)
                kernel_regressor = ["CatBoostRegressor","RANSACRegressor","KernelRidge","SVR",
                                    "KNeighborsRegressor", "MLPRegressor",
                                    "AdaBoostRegressor"]
                if is_shap:
                    if model.__class__.__name__ in kernel_regressor: 
                        options = ['default','bar','violin']
                    else:
                        options=['bar','beeswarm','heatmap']
                    with st.container():
                        with st.expander("Interpret the Model with Global SHAP Value"):
                            plot_type = st.selectbox('Select a Type of Plot', options=options)
                            max_display = st.slider('Maximum Number to Display', min_value=1, max_value=X_train.shape[1],value=10,key=1)
                            try:
                                plot_reg_shap_global_and_local('global',model, X_train, plot_type, max_display)
                            except:
                                st.error("Plot Not Available for the Model.")    

                            
                        with st.expander("Interpret the Model with Local SHAP Value"):
                            if model.__class__.__name__ in kernel_regressor: 
                                max_display_local = None
                            else:
                                max_display_local = st.slider('Maximum Number to Display', min_value=1, max_value=X_train.shape[1],value=10,key=2)
                            
                            index_of_explain = st.number_input('Index to Explain from Prediction',min_value=0,max_value=X_train.shape[0],value=0)
                            try:
                                plot_reg_shap_global_and_local('local',model,X_train,None,max_display_local,index_of_explain)
                            except:
                                st.error("Plot Not Available for the Model.")    

        return state
    else:
        st.error("Please Train a Model first!")

