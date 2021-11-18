import streamlit as st
from utils.retrieve_models_name import retrieve_models_name
from utils.convert_dict_to_df import convert_dict_to_df
from pycaret.clustering import *
from st_aggrid import AgGrid

def write(state):
    st.subheader("Create Model for Unsupervised Learning")
    
    if state.is_set_up:
        all_models = retrieve_models_name(type="Clustering")
        #st.write(all_models)
        
        select_model = None 
        select_model_names = list(all_models.keys())
        
        with st.expander("Select Parameters for Creating Model"):
            select_model = st.selectbox('Select a Model to Create', options=select_model_names)
            num_clusters = st.number_input('The number of clusters to form', min_value=1, value=4)
            button_create = st.button('Training a Single Model')
            try:
                if button_create:
                    with st.spinner("Training Model..."):
                        state.trained_model = create_model(model=all_models[select_model], num_clusters=num_clusters)
                    state.log_history["create_model"] = pull(True)
            except:
                st.error("Please Set Up Dataset first!")   

            st.markdown('<p style="color:#1386fc">Show All the Metrics Results After Tuning.</p>',unsafe_allow_html=True)       
            button_after_create = st.button("Show Model Result")
            try:
                if button_after_create:
                    with st.spinner("Show All the Results..."):
                        AgGrid(state.log_history["create_model"])
            except:
                st.error("Please Train a Model first!")
                
            is_tuning = st.checkbox("Do You want to Tune the Number of Clusters?", value=False)
            if is_tuning:
                if state.supervised_target is not None:
                        model_tune = st.selectbox('Select a Model to Tune', options=['kmeans','sc','hclust','birch','kmodes'])
                        supervised_type = st.selectbox('Type of Task', options=['None', 'regression','classification'])
                        supervised_estimator = 'lr'
                        optimize = None
                        
                        if supervised_type == 'None':
                            supervised_type = None
                            supervised_estimator_list = None
                            optimize_list = None
                        elif supervised_type == 'regression':
                            # supervised_estimator_list =  list(retrieve_models_name(type="Regression").values())
                            supervised_estimator_list =  ['lr','lasso','ridge','en','lar','llar','omp',
                                                          'br','ard','par','ransac','tr','huber','kr',
                                                          'svm','knn','dt','rf','et','ada','gbr','mlp',
                                                          'xgboost','lightgbm','catboost']
                            optimize_list = ['R2','MSE','RMSE','MAE','RMSLE','MAPE']
                        else:
                            # supervised_estimator_list = list(retrieve_models_name(type="Classification").values())
                            supervised_estimator_list =  ['lr','knn','nb','dt','svm','rbfsvm','mlp','gpc',
                                                          'ridge','rf','qda','ada','gbc','lda','et','xgboost',
                                                          'lightgbm','catboost']
                            optimize_list = ['Accuracy', 'AUC','Recall','Precision','F1','Kappa']
                        
                        if supervised_type is not None:
                            supervised_estimator = st.selectbox('Supervised Model', options=supervised_estimator_list)
                            optimize = st.selectbox('Metric to Optimize', options=optimize_list)
                        fold = st.number_input('Number of Folds to Used in Kfold CV', min_value=2, value=10)
                        
                        with st.spinner("Tuning..."):
                            button_tuning = st.button('Tune Hyperparameter')
                            if button_tuning:

                                state.trained_model = tune_model(model_tune,state.supervised_target, supervised_type,
                                                            supervised_estimator, optimize,fold=fold)

                                state.log_history["tuned_models"] = pull(True)

                        st.markdown('<p style="color:#1386fc">Show All the Metrics Results After Tuning.</p>',unsafe_allow_html=True)       
                        button_after_tuning = st.button("Show Tuning Model Result")
                        if button_after_tuning:
                            with st.spinner("Show All the Results..."):
                                # st.write(convert_dict_to_df(state.log_history["tuned_models"]))
                                AgGrid(state.log_history["tuned_models"])
                    
                else:
                    st.error("Please Select a Supervised Target from Data Setup Step!")
            
            state.X_before_preprocess = get_config('data_before_preprocess')
        return state
    else:
        st.error("Please Set Up Dataset first!")