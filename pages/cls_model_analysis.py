import streamlit as st
from pycaret.classification import *
from utils.plot_shap import plot_cls_shap_global_and_local

def write(state):
    st.subheader("Analyzing Performance of Trained Machine Learning Model")
    state.X_train = get_config('X_train')
    state.X_test = get_config('X_test')
    state.y_train = get_config('y_train')
    state.y_test = get_config('y_test')
    
    
    if state.trained_model is not None:
        model = state.trained_model
        task_type = state.classification_task
        X_train = state.X_train
        # X_test = state.X_test
        # y_train = state.y_train
        # y_test = state.y_test

        with st.container():
            
            with st.expander("Show Training Performance Plots"):
                # plot all pycaret support diagrams
                plot = st.selectbox('Select a Plot', options=['auc','threshold','pr','confusion_matrix','error',
                                                              'class_report','boundary','rfe','learning','manifold'
                                                              ,'vc','dimension',])
                try:     
                    plot_model(estimator=model, plot=plot, display_format='streamlit')
                except:
                    st.error("Plot Not Available for multiclass problems.")     

            if state.is_ensemble:
                st.markdown('<p style="color:#f42756">SHAP Value is not supported for Ensemble Model</p>',unsafe_allow_html=True)       
            else:
                st.markdown("---")
                is_shap = st.checkbox("Do You Want to Check SHAP Value?", value=False)
                kernel_classifier= ["KNeighborsClassifier","CatBoostClassifier","AdaBoostClassifier",
                                    "QuadraticDiscriminantAnalysis","NaiveBayes", "GaussianProcessClassifier","MLPClassifier"]
                bar_tree_classifier = ["ExtraTreesClassifier","RandomForestClassifier","DecisionTreeClassifier"]
                multi_bar_tree_classifier =["ExtraTreesClassifier","CatBoostClassifier","RandomForestClassifier",
                                            "DecisionTreeClassifier","ExtremeGradientBoosting","LightGradientBoostingMachine"]
                if is_shap:
                    if task_type == 'Binary':
                        if model.__class__.__name__ in kernel_classifier: 
                            options = ['default','bar','violin']
                        elif model.__class__.__name__ in bar_tree_classifier:
                            options = ["bar"]
                        else:
                            options=['bar','beeswarm','heatmap']
                    else:
                        if model.__class__.__name__ in multi_bar_tree_classifier:
                            options=["bar"]
                        else:
                            options = ['default','bar','violin']
                    with st.container():
                        with st.expander("Interpret the Model with Global SHAP Value"):
                            plot_type = st.selectbox('Select a Type of Plot', options=options)
                            if 'beeswarm' in options:
                                max_display = st.slider('Maximum Number to Display', min_value=1, max_value=X_train.shape[1],value=10,key=1)
                            else:
                                max_display=None
                            try:
                                plot_cls_shap_global_and_local('global',model, X_train, task_type,plot_type,max_display)
                            except:
                                st.error("Plot Not Available for the Model.")    
                                
                    with st.expander("Interpret the Model with Local SHAP Value"):
                        if 'beeswarm' in options:
                            max_display_local = st.slider('Maximum Number to Display', min_value=1, max_value=X_train.shape[1],value=10,key=2)
                        else:
                            max_display_local = None
                        
                        index_of_explain = st.number_input('Index to Explain from Prediction',min_value=0,max_value=X_train.shape[0],value=0)
                        try:
                            plot_cls_shap_global_and_local('local',model,X_train,task_type,None,max_display_local,index_of_explain)
                        except:
                            st.error("Plot Not Available for the Model.")    



        return state
    else:
        st.error("Please Train a Model first!")

