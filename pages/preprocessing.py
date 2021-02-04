from pycaret.regression import  *
import streamlit as st
import pandas as pd
from utils.convert_dict_to_df import convert_dict_to_df

def write(state):
    # select the target column
    if state.df is not None:
        df = state.df
        columns_name = df.columns.tolist()    
        with st.beta_container():
            st.subheader("Select the target variable to make prediction:")
            target_column = st.selectbox('Target column:',options=columns_name,index=len(columns_name)-1)
        
            # select the feature columns
            columns_name.remove(target_column)
            feature_columns = columns_name

            # remove the columns ? 
            remove_column = st.checkbox('Do you have column(s) to remove?')
            remove_columns = None
            if remove_column:
                remove_columns = st.multiselect('Select one or more column(s) to remove',
                                            feature_columns)
                feature_columns = [col for col in feature_columns if col not in remove_columns]
                state.ignore_columns = remove_columns
        # training and testing split
        with st.beta_expander("Traning and Testing Split"):
            size = st.number_input('Training Size:', value=0.7)
        
        # Preprocessing
        with st.beta_expander("Preprocessing"):
            with st.beta_container():
                st.markdown('<p style="color:#f42756">Preprocessing for Numeric Columns:</p>',unsafe_allow_html=True)
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
                
                transform_target = st.checkbox('Apply Transformation to Target Value',value=False)
                transform_target_method = 'box-cox'
                if transform_target:
                    transform_target_method = st.selectbox('Transformation for Target Value', options=['box-cox','yeo-johnson'])    
                
                state.transform_target = transform_target
                # select categorical features
                categorical_columns = df.select_dtypes(include=['category','object']).columns.tolist()
                categorical_imputation = 'constant'
                unknown_categorical_method = 'least_frequent'
                combine_rare_levels = False
                rare_level_threshold = 0.1
                # fix_imbalance = False
                
                if len(categorical_columns) > 0:
                    st.markdown('<p style="color:#f42756">Preprocessing for Categorical Columns:</p>',unsafe_allow_html=True)
                    with st.beta_container():
                        categorical_imputation = st.selectbox('Missing Values for Categorical', options=['constant','mode'])
                        unknown_categorical_method = st.selectbox('Handle Unknown Categorical values', options=['least_frequent','most_frequent'])
                        combine_rare_levels = st.checkbox('Combined Rare Levels of Categorical Features as a Single Level',value=False)
                        if combine_rare_levels:
                            rare_level_threshold = st.number_input('Percentile Distribution below Rare Categories are Combined',min_value=0.0,value=0.1)
                        # fix_imbalance = st.checkbox('Fix Unequal Distribution of Target class', value=False)
                    
        # Feature Engineering
        with st.beta_expander("Creating New Features through Features Engineering"):
            with st.beta_container():
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
                bin_numeric_features = None
                bin_numeric_features = st.multiselect('Select Numeric Features Transformed into Categorical Features using K-Means',
                                                    feature_columns)
                bin_numeric_features = bin_numeric_features if len(bin_numeric_features) > 0 else None
                
        # Feature Selection
        with st.beta_expander("Select Features in Dataset Contributes the most in Predicting Target Variable"):
            with st.beta_container():
                feature_selection = st.checkbox('Select a Subset of Features Using a Combination of various Permutation Importance', value=False)
                feature_selection_threshold = 0.8
                if feature_selection:
                    feature_selection_threshold = st.number_input('Threshold for Feature Selection',min_value=0.0,value=0.8)
                    
                remove_multicollinearity = st.checkbox('Remove Highly Linearly Correlated Features', value=False)
                multicollinearity_threshold = 0.9
                if remove_multicollinearity:
                    multicollinearity_threshold = st.number_input('Threshold Used for Dropping the Correlated Features', min_value=0.0, value=0.9)
                
                pca = st.checkbox('Used PCA to Reduce the Dimensionality of the Dataset', value=False)
                pca_method='linear'
                pca_components = 0.99
                if pca:
                    pca_method = st.selectbox('The Method to Perform Linear Dimensionality Reduction', options=['linear','kernel','incremental'])
                    pca_components = st.number_input('Number of components to keep (float or int)', value=0.99)

                ignore_low_variance = st.checkbox('Remove Categorical Features with Statistically Insignificant Variances', value=False)
                
        # Unsupervised 
        with st.beta_expander("Creating Clusters using the Existing Features from the data with Unsupervised Techniques"):
            with st.beta_container():
                create_clusters = st.checkbox('Create Additioal Features with Clusters', value=False)
                cluster_iter = 20
                if create_clusters:
                    cluster_iter = st.number_input('Number of Iterations used to Create a Cluster', min_value=0, value=20)
                    
                remove_outliers = st.checkbox('Remove Outliers from Training data using PCA')
                outliers_threshold = 0.05
                if remove_outliers:
                    outliers_threshold = st.number_input('The Percentage of Outliers', min_value=0.0, value=0.05)
        
        st.subheader("Start Loading, Preprocessing and Transforma Dataset:")
        with st.beta_container():          
            button_run = st.button("Start Process and Transform")
            if button_run:
                with st.spinner("Preprocessing..."):
                    setup(data=df, target=target_column,train_size=size, preprocess=True,
                    categorical_imputation=categorical_imputation, numeric_imputation=numeric_imputation,
                    normalize=normalize,normalize_method=normalize_method, transformation=transformation,
                    transformation_method=transformation_method, transform_target=transform_target,
                    transform_target_method=transform_target_method,unknown_categorical_method=unknown_categorical_method,
                    combine_rare_levels=combine_rare_levels,rare_level_threshold=rare_level_threshold,
                    feature_interaction=feature_interaction,ignore_features=remove_columns,
                    feature_ratio=feature_ratio,polynomial_features=polynomial_features,
                    polynomial_degree=polynomial_degree,polynomial_threshold=polynomial_threshold,
                    trigonometry_features=trigonometry_features,group_features=group_features,
                    bin_numeric_features=bin_numeric_features,feature_selection=feature_selection,
                    feature_selection_threshold=feature_selection_threshold,remove_multicollinearity=remove_multicollinearity,
                    multicollinearity_threshold = multicollinearity_threshold,pca=pca,pca_method=pca_method,pca_components=pca_components,
                    ignore_low_variance=ignore_low_variance,create_clusters=create_clusters,cluster_iter=cluster_iter,
                    remove_outliers=remove_outliers,outliers_threshold=outliers_threshold,html=False,silent=True)

                state.log_history = {"setup":pull(True).data.to_dict()} 
                # record the setup procedure
                state.is_set_up = True
                
            st.write("Do you want to check Transformed Data?")
            button_transform = st.button("Check Transformed Data")
            if button_transform:
                with st.spinner("Loading..."):
                    st.write(convert_dict_to_df(state.log_history["setup"]))
        
        st.subheader("Compare All the Machine Learning Model Result:")
        with st.beta_expander("Select Parameters for Comparing Models"):
            with st.beta_container():
                fold_text = st.text_input('Control Cross Validation Folds (int or None)', value='None')
                fold = None if fold_text == 'None' else int(fold_text)

                cross_validation = st.checkbox('Allow Cross Validation or not', value=True)
                sort = st.selectbox('The Sort Order of the Score Grid', options=['R2','MAE','MSE','RMSE','RMSLE','MAPE'], )

        st.write("Compare All the Machine Learning Models based on selected Metrics.")        
        button_compare = st.button("Compare Models")
        if button_compare:
            with st.spinner('Comparing all Models...'):
                state.best = compare_models(fold=fold, cross_validation=cross_validation, sort=sort)
                state.log_history["compare_models"] = pull(True).to_dict()

        st.write("Show All the Metrics Results.")      
        button_model = st.button("Show All Result")    
        if button_model:
            with st.spinner("Show All the Results..."):
                st.write(convert_dict_to_df(state.log_history["compare_models"]))
        

        return state

    else:
        return st.error("Something went wrong! Please upload dataset first!")
