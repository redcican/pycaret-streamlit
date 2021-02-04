import streamlit as st
from utils.retrieve_models_name import retrieve_models_name
from utils.convert_dict_to_df import convert_dict_to_df
from pycaret.regression import *

def write(state):
    st.subheader("Create Model from Best Result or Select Another Model?")
    
    if state.is_set_up:
    
        all_models = retrieve_models_name(is_regression=True)

        select_model = None 
        select_model_names = list(all_models.keys())

        if state.best is None:
            select_model = st.selectbox('Select Another Model to Create', options=select_model_names)
        else:
            # check if target transformation is applied
            if state.transform_target:
                best_name = state.best.__dict__['regressor'].__class__.__name__
                st.write(best_name)
            else:
                best_name = state.best.__class__.__name__
            # to consider: when apply to target value: PowerTransformedTargetRegressor
            if best_name == "Lasso":
                best_name = "LassoRegression"
                
            from_best_model = st.checkbox('Create Model from Best Result', value=True)
            if from_best_model:
                select_model = best_name
            else:
                select_model_names.remove(best_name)
                models_names = select_model_names
                select_model = st.selectbox('Select Another Model to Create', options=models_names)
        
        st.subheader("Create Model")
        with st.beta_expander("Select Parameters for Creating Model"):
            fold_text = st.text_input('Control Cross Validation Folds (int or None)', value='None',key=1)
            fold = None if fold_text == 'None' else int(fold_text)
            cross_validation = st.checkbox('Allow Cross Validation or not', value=True)
            
            button_create = st.button('Create Model')
            if button_create:
                with st.spinner("Creating Model..."):
                    state.trained_model = create_model(estimator=all_models[select_model], fold=fold, cross_validation=cross_validation)
                state.log_history["create_model"] = pull(True).to_dict()
            
            st.write("Show All the Metrics Results After Creating.")
            button_after_create = st.button("Show Model Result")
            if button_after_create:
                with st.spinner("Show All the Results..."):
                    st.write(convert_dict_to_df(state.log_history["create_model"]))
            
            is_tuning = st.checkbox("Do You want to Tune the Hyperparemters?", value=False)
            if is_tuning:
                fold_text_tune = st.text_input('Control CV Folds (int or None)', value='None',key=2)
                fold_tune = None if fold_text_tune == 'None' else int(fold_text_tune)
                n_iter = st.number_input("Number of iterations in the Grid Search", min_value = 1, value=10)
                optimize = st.selectbox('Metric Name to be Evaluated for Hyperparameter Tuning', options=['R2','MAE','MSE','RMSE','RMSLE','MAPE'], )
                search_library = st.selectbox('The Search Library Used for Tuning Hyperparameters.',options=['scikit-learn',
                                                                                                            'scikit-optimize',
                                                                                                            'tune-sklearn',
                                                                                                            'optuna'])
                search_algorithms = []
                if search_library == 'scikit-learn':
                    search_algorithms = ['random','grid']
                elif search_library == 'scikit-optimize':
                    search_algorithms = ['bayesian']
                elif search_library  == 'tune-sklearn':
                    search_algorithms = ['random','grid','bayesian','hyperopt','optuna','bohb']
                else:
                    search_algorithms = ['random','tpe']
                search_algorithm = st.selectbox('The Search Algorithm Dependes on Search Library',options=search_algorithms)
                early_stopping = st.checkbox('Stop Fitting to a Hyperparameter Configuration if it Performs Pooly', value=False)
                early_stopping_iter = 10
                if early_stopping:
                    early_stopping_iter = st.number_input("Maximum Number of Epochs to Run", min_value=1, value=10)
                choose_better = st.checkbox('When Set to True, the Returned Object is Always Betting Performing', value=False)
                button_tuning = st.button('Tune Hyperparameter')
                if button_tuning:
                    with st.spinner("Search Hyperparamters..."):
                        created_model = create_model(estimator=all_models[select_model])
                        state.trained_model = tune_model(created_model, fold=fold_tune, n_iter=n_iter,
                                            optimize=optimize, search_library=search_library, search_algorithm=search_algorithm,
                                            early_stopping=early_stopping, early_stopping_max_iters=early_stopping_iter,
                                            choose_better = choose_better)
                        state.log_history["tuned_models"] = pull(True).to_dict()
                
                st.write("Show All the Metrics Results After Tuning.")
                button_tuning = st.button("Show Tuning Model Result")
                if button_tuning:
                    with st.spinner("Show All the Results..."):
                        st.write(convert_dict_to_df(state.log_history["tuned_models"]))
            
            state.X_before_preprocess = get_config('data_before_preprocess')
            state.y_before_preprocess = get_config('target_param')

        return state
            
    else:
        st.error("Please Set Up Dataset first!")           
    