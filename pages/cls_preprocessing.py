from pycaret.classification import  *
import streamlit as st
from st_aggrid import AgGrid

def write(state):
    # select the target column
    if state.df is not None:
        df = state.df
        
        columns_name = df.columns.tolist()    
        with st.container():
            st.subheader("Select the target variable to make prediction:")
            target_column = st.selectbox('Target column:',options=df.columns,index=len(columns_name)-1)
        
            # select the feature columns
            columns_name.remove(target_column)
            feature_columns = columns_name

            # remove the columns ? 
            remove_column = st.checkbox('Do you have column(s) to remove?', value=state.is_remove)
            
            ignore_columns = state.ignore_columns 
            
            if remove_column:
                remove_columns = st.multiselect('Select one or more column(s) to remove',feature_columns,default=ignore_columns)
                feature_columns = [col for col in feature_columns if col not in remove_columns]
                
                state.is_remove = True
                state.ignore_columns = remove_columns
                
        # training and testing split
        with st.expander("Traning and Testing Split"):
            size = st.number_input('Training Size:', value=0.7)
            data_split_stratify = st.checkbox("Controls Stratification during Split", value=False)
            fold_strategy = st.selectbox('Choice of Cross Validation Strategy',options=['kfold','stratifiedkfold','groupkfold'])
            fold = st.number_input('Number of Folds to be Used in Cross Validation',min_value=2,value=10)
       
        # Preprocessing
        with st.expander("Preprocessing"):
            with st.container():
                st.markdown('<p style="color:#1386fc">Preprocessing for Numeric Columns:</p>',unsafe_allow_html=True)
                numeric_imputation = st.selectbox('Missing Value for Numeric Columns', options=['mean','median'])
                # select numberical features preprocessing
                normalize = st.checkbox('Normalization', value=False)
                normalize_method = 'zscore'
                if normalize:
                    normalize_method = st.selectbox('Method to be used for Normalization',options=['zscore','minmax','maxabs','robust'])
                
                transformation = st.checkbox('Transformation', value=False)
                transformation_method = 'yeo-johnson'
                if transformation:
                    transformation_method = st.selectbox('Method for Transfomation', options=['yeo-johnson','quantile'])
                
                fix_imbalance = st.checkbox('Fix Imbalance of Target Classes',value=False)
                # fix_imbalance_method = None
                # if fix_imbalance:
                #     fix_imbalance_method = st.selectbox('Method to Handle Imbalance', options=['SMOTE','fit_resample'])    
                #     if fix_imbalance_method == 'SMOTE':
                #         fix_imbalance_method = None

                # select categorical features
                categorical_columns = df.select_dtypes(include=['category','object']).columns.tolist()
                categorical_imputation = 'constant'
                unknown_categorical_method = 'least_frequent'
                combine_rare_levels = False
                rare_level_threshold = 0.1
                
                if len(categorical_columns) > 0:
                    st.markdown('<p style="color:#1386fc">Preprocessing for Categorical Columns:</p>',unsafe_allow_html=True)
                    with st.beta_container():
                        categorical_imputation = st.selectbox('Missing Values for Categorical', options=['constant','mode'])
                        unknown_categorical_method = st.selectbox('Handle Unknown Categorical values', options=['least_frequent','most_frequent'])
                        combine_rare_levels = st.checkbox('Combined Rare Levels of Categorical Features as a Single Level',value=False)
                        if combine_rare_levels:
                            rare_level_threshold = st.number_input('Percentile Distribution below Rare Categories are Combined',min_value=0.0,value=0.1)
                    
        # Feature Engineering
        with st.expander("Creating New Features through Features Engineering"):
            with st.container():
                feature_interaction = st.checkbox('Create new Features by Interaction', value=False)
                feature_ratio = st.checkbox('Create new Features by Calculating Ratios', value=False)
                
                polynomial_features = st.checkbox('Create new Features based on Polynomial Combinations', value=False)
                polynomial_degree=2
                polynomial_threshold=0.1
                if polynomial_features:
                    polynomial_degree = st.number_input('Polynomial Degree (int)',min_value=1,step=1,value=2)
                    polynomial_threshold = st.number_input('Polynomial Threshold',min_value=0.0,value=0.1)
                
                trigonometry_features = st.checkbox('Create new Features based on all Trigonometric', value=False)
                group_features = st.multiselect('Select Features that have Related Characteristics',feature_columns)
                group_features = group_features if len(group_features) > 0 else None
                
                bin_numeric_features = st.checkbox('Create new Features based on Bin Combinations', value=False)
                select_bin_numeric_features=None
                if bin_numeric_features:
                    select_bin_numeric_features = st.multiselect('Select Numeric Features Transformed into Categorical Features using K-Means', feature_columns)
                
        # Feature Selection
        with st.expander("Select Features in Dataset Contributes the most in Predicting Target Variable"):
            with st.container():
                feature_selection = st.checkbox('Select a Subset of Features Using a Combination of various Permutation Importance', value=False)
                feature_selection_threshold = 0.8
                if feature_selection:
                    feature_selection_threshold = st.number_input('Threshold for Feature Selection',min_value=0.0,value=0.8)
                    
                remove_multicollinearity = st.checkbox('Remove Highly Linearly Correlated Features', value=False)
                multicollinearity_threshold = 0.9
                if remove_multicollinearity:
                    multicollinearity_threshold = st.number_input('Threshold Used for Dropping the Correlated Features', min_value=0.0, value=0.9)
                
                remove_perfect_collinearity = st.checkbox('Remove Perfect Collinearity (Correaltion=1) Feature', value=False)
                
                pca = st.checkbox('Used PCA to Reduce the Dimensionality of the Dataset', value=False)
                pca_method='linear'
                pca_components = 0.99
                if pca:
                    pca_method = st.selectbox('The Method to Perform Linear Dimensionality Reduction', options=['linear','kernel','incremental'])
                    pca_components = st.number_input('Number of components to keep (float or int)', value=0.99)

                ignore_low_variance = st.checkbox('Remove Categorical Features with Statistically Insignificant Variances', value=False)
                
        # Unsupervised 
        with st.expander("Creating Clusters using the Existing Features from the data with Unsupervised Techniques"):
            with st.container():
                create_clusters = st.checkbox('Create Additioal Features with Clusters', value=False)
                cluster_iter = 20
                if create_clusters:
                    cluster_iter = st.number_input('Number of Iterations used to Create a Cluster', min_value=0, value=20)
                    
                remove_outliers = st.checkbox('Remove Outliers from Training data using PCA')
                outliers_threshold = 0.05
                if remove_outliers:
                    outliers_threshold = st.number_input('The Percentage of Outliers', min_value=0.0, value=0.05)
        
        st.subheader("Start Loading, Preprocessing and Transforma Dataset:")
        with st.container():          
            button_run = st.button("Start Process and Transform")
            if button_run:
                with st.spinner("Preprocessing..."):
                    setup(data=df, target=target_column,train_size=size, preprocess=True,
                    categorical_imputation=categorical_imputation, numeric_imputation=numeric_imputation,
                    normalize=normalize,normalize_method=normalize_method, transformation=transformation,
                    transformation_method=transformation_method, 
                    unknown_categorical_method=unknown_categorical_method,
                    combine_rare_levels=combine_rare_levels,rare_level_threshold=rare_level_threshold,
                    feature_interaction=feature_interaction,ignore_features=ignore_columns,
                    feature_ratio=feature_ratio,polynomial_features=polynomial_features,
                    polynomial_degree=polynomial_degree,polynomial_threshold=polynomial_threshold,
                    trigonometry_features=trigonometry_features,group_features=group_features,
                    bin_numeric_features=select_bin_numeric_features,feature_selection=feature_selection,
                    feature_selection_threshold=feature_selection_threshold,remove_multicollinearity=remove_multicollinearity,
                    multicollinearity_threshold = multicollinearity_threshold,           
                    remove_perfect_collinearity=remove_perfect_collinearity,
                    fix_imbalance = fix_imbalance, 
                    data_split_stratify=data_split_stratify,fold_strategy=fold_strategy,fold=int(fold),
                    pca=pca,pca_method=pca_method,pca_components=pca_components,
                    ignore_low_variance=ignore_low_variance,create_clusters=create_clusters,cluster_iter=cluster_iter,
                    remove_outliers=remove_outliers,outliers_threshold=outliers_threshold,html=False,silent=True)

                state.log_history["set_up"] = pull(True)
                # record the setup procedure
                state.is_set_up = True
                state.classification_task = list(state.log_history["set_up"].data.to_dict()["Value"].values())[2]
                # st.write(state.classification_task[2])
                
                # set best model to None
                state.best = None
                
            st.markdown('<p style="color:#1386fc">Do you want to check Transformed Data?</p>',unsafe_allow_html=True)
            button_transform = st.button("Check Transformed Data")
            try:
                if button_transform:
                    with st.spinner("Loading..."):
                        AgGrid(state.log_history["set_up"].data)
            except:
                st.error("Please Process and Transform Data first!")
                    
            
        st.subheader("Compare All the Machine Learning Model Result:")
        with st.expander("Select Parameters for Comparing Models"):
            with st.container():
                fold_text = st.text_input('Control Cross Validation Folds (int or None)', value='None')
                fold_compare = None if fold_text == 'None' else int(fold_text)

                cross_validation = st.checkbox('Allow Cross Validation or not', value=True)
                sort = st.selectbox('The Sort Order of the Score Grid', options=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'], )

        st.markdown('<p style="color:#1386fc">Compare All the Machine Learning Models based on selected Metrics.</p>',unsafe_allow_html=True)     
        button_compare = st.button("Compare Models")
        try:
            if button_compare:
                with st.spinner('Comparing all Models...'):
                    state.best = compare_models(exclude=['xgboost'],fold=fold_compare, cross_validation=cross_validation, sort=sort)
                    state.log_history["compare_models"] = pull(True)
        except:
            st.error("Please Process and Transform Data first!")

        st.markdown('<p style="color:#1386fc">Show All the Metrics Results.</p>',unsafe_allow_html=True)       
        button_model = st.button("Show All Result")  
        try:  
            if button_model:
                with st.spinner("Show All the Results..."):
                    AgGrid(state.log_history["compare_models"])
        except:
            st.error("Please Compare All Models first!")
        

        return state

    else:
        return st.error("Something went wrong! Please upload dataset first!")
