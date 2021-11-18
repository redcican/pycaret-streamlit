import streamlit as st
from utils.retrieve_models_name import retrieve_models_name
from st_aggrid import AgGrid
from pycaret.classification import *

def write(state):
    st.subheader("Create Model from Best Result or Select Single Model or Ensemble?")
    
    if state.is_set_up:
    
        all_models = retrieve_models_name(type="Classification")

        select_model = None 
        select_model_names = list(all_models.keys())
        select_model_names.remove('ExtremeGradientBoosting')
        
        best_name = ""
        ensemble_method =""
        select_ensemble_method = ""
        select_model_blend = []
        select_model_stack_first = []
        select_model_stack_seconnd = ""

        if state.best is not None:
            train_options = ["From Best", "Single Model", "Ensemble Model"]
            best_name = state.best.__class__.__name__
            # the class name of "Lasso" must be modified manually
            if best_name == "GaussianNB":
                best_name = "NaiveBayes" 
            if best_name == "SVC": # must consider rbfsvm, too.
                best_name = "SVM-RadialKernel"

        else:
            train_options = ["Single Model", "Ensemble Model"]
               
        train_option = st.radio("Select a Mode to Train Model", options=train_options)
        
        if train_option == "From Best":        
            select_model = best_name
            state.is_ensemble = False

        if train_option == "Single Model":
            select_model = st.selectbox('Select Another Model to Create', options=select_model_names)
            state.is_ensemble = False

        if train_option == "Ensemble Model":
            state.is_ensemble = True
            ensemble_method = st.selectbox("Select a Method for Ensemble", options=["Ensemble","Blend","Stack"])
            if ensemble_method == "Ensemble":
                select_model = st.selectbox('Select a Base Model for Ensemble', options=select_model_names)
                select_ensemble_method = st.selectbox('Select a Method for Ensemble', options=["Bagging","Boosting"])
            elif ensemble_method == "Blend":
                select_model_blend = st.multiselect('Select One or More Models(s) for Blending', options=select_model_names)
            else:
                select_model_stack_first = st.multiselect('Select One or More Models(s) for First Layer Stacking', options=select_model_names)
                select_model_stack_seconnd = st.selectbox('Select a Base Model for Ensemble', options=select_model_names)
        
        
        st.subheader("Create Model")
        with st.expander("Select Parameters for Creating Model"):
            if not state.is_ensemble:
                fold_text = st.text_input('Control Cross Validation Folds (int or None)', value='None',key=1)
                fold = None if fold_text == 'None' else int(fold_text)
                cross_validation = st.checkbox('Allow Cross Validation or not', value=True)

                button_create = st.button('Training a Single Model')
                try:
                    if button_create:
                        with st.spinner("Training Model..."):
                            state.trained_model = create_model(estimator=all_models[select_model], fold=fold, cross_validation=cross_validation)
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

                is_tuning = st.checkbox("Do You want to Tune the Hyperparemters?", value=False)
                if is_tuning:
                    fold_text_tune = st.text_input('Control CV Folds (int or None)', value='None',key=2)
                    fold_tune = None if fold_text_tune == 'None' else int(fold_text_tune)
                    n_iter = st.number_input("Number of iterations in the Grid Search", min_value = 1, value=10)
                    optimize = st.selectbox('Metric Name to be Evaluated for Hyperparameter Tuning', options=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'] )
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

                    st.markdown('<p style="color:#1386fc">Show All the Metrics Results After Tuning.</p>',unsafe_allow_html=True)       
                    button_tuning = st.button("Show Tuning Model Result")
                    if button_tuning:
                        with st.spinner("Show All the Results..."):
                            AgGrid(state.log_history["tuned_models"])
            
            else:
                fold_ensemble_text = st.text_input('Control Cross Validation Folds (int or None)', value='None',key=3)
                fold_ensemble = None if fold_ensemble_text == 'None' else int(fold_ensemble_text)
                choose_better_ensemble = st.checkbox('When Set to True, the Returned Object is Always Betting Performing', value=False)
                optimize_ensemble = st.selectbox('Metric Name to be Evaluated for Hyperparameter Tuning', options=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'], )

                if ensemble_method == "Ensemble":
                    n_estimators = st.number_input("The number of Base Estimators in the Ensemble", min_value=1, value=10)
                    button_ensemble = st.button("Training an Ensemble Model")
                    if button_ensemble:
                        with st.spinner("Training Ensemble Model"):
                            base = create_model(estimator=all_models[select_model])
                            state.trained_model = ensemble_model(base, method=select_ensemble_method, fold=fold_ensemble,
                                                                 n_estimators=n_estimators, choose_better=choose_better_ensemble,
                                                                 optimize=optimize_ensemble)
                        state.log_history["create_model"] = pull(True)

                elif ensemble_method == "Blend":
                    bases = []
                    button_ensemble = st.button("Training an Ensemble Model")
                    if button_ensemble:
                        with st.spinner("Training Blending Model"):
                            for model in select_model_blend:
                                base = create_model(estimator=all_models[model])
                                bases.append(base)
                            state.trained_model = blend_models(estimator_list=bases, fold=fold_ensemble,
                                                                 choose_better=choose_better_ensemble,
                                                                 optimize=optimize_ensemble)
                        state.log_history["create_model"] = pull(True)

                else:
                    restack = st.checkbox("Restack the Predictions for the Meta Model", value=True)
                    button_ensemble = st.button("Training an Ensemble Model")
                    bases = []
                    if button_ensemble:
                        with st.spinner("Training Stacking Model"):
                            for model in select_model_stack_first:
                                base = create_model(estimator=all_models[model])
                                bases.append(base)
                            if select_model_stack_seconnd in select_model_stack_first:
                                index = select_model_stack_first.index(select_model_stack_seconnd)
                                meta_model = bases[index]
                            else:
                                meta_model = create_model(all_models[select_model_stack_seconnd])
                            state.trained_model = stack_models(estimator_list=bases, meta_model=meta_model,
                                                               fold=fold_ensemble, restack=restack,
                                                                 choose_better=choose_better_ensemble,
                                                                 optimize=optimize_ensemble)
                        state.log_history["create_model"] = pull(True)
                
                st.markdown('<p style="color:#1386fc">Show All the Metrics Results After Tuning.</p>',unsafe_allow_html=True)       
                button_after_create = st.button("Show Model Result")
                if button_after_create:
                    with st.spinner("Show All the Results..."):
                        st.table(convert_dict_to_df(state.log_history["create_model"]))
   
            state.X_before_preprocess = get_config('data_before_preprocess')
            state.y_before_preprocess = get_config('target_param')

        return state
            
    else:
        st.error("Please Set Up Dataset first!")           
    