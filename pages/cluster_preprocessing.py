from pycaret.clustering import  *
import streamlit as st
from utils.convert_dict_to_df import convert_dict_to_df

def write(state):
    # select the target column
    if state.df is not None:
        df = state.df
        
        columns_name = df.columns.tolist()
        with st.beta_expander("Select Columns"):    
            with st.beta_container():
                feature_columns = columns_name

                # remove the columns ? 
                remove_column = st.checkbox('Do You Have Column(s) to Remove?', value=False)
                remove_columns = None
                if remove_column:
                    remove_columns = st.multiselect('Select One or More Column(s) to Remove',
                                                feature_columns)
                    feature_columns = [col for col in feature_columns if col not in remove_columns]
                    state.ignore_columns = remove_columns
                supervised = st.checkbox('Is There Supervised Target Column?', value=False)
                supervised_target = None
                if supervised:
                    supervised_target = st.selectbox('Name of Supervised Column',options=feature_columns)
                    state.supervised_target = supervised_target
        # Preprocessing
        with st.beta_expander("Preprocessing"):
            with st.beta_container():
                st.markdown('<p style="color:#1386fc">Preprocessing for Numeric Columns:</p>',unsafe_allow_html=True)
                numeric_imputation = st.selectbox('Missing Value for Numeric Columns', options=['mean','median','zero'])
                # select numberical features preprocessing
                normalize = st.checkbox('Normalization', value=False)
                normalize_method = 'zscore'
                if normalize:
                    normalize_method = st.selectbox('Method to be used for Normalization',options=['zscore','minmax','maxabs','robust'])
                
                transformation = st.checkbox('Transformation', value=False)
                transformation_method = 'yeo-johnson'
                if transformation:
                    transformation_method = st.selectbox('Method for Transfomation', options=['yeo-johnson','quantile'])
                
                categorical_columns = df.select_dtypes(include=['category','object']).columns.tolist()
                categorical_imputation = 'constant'
                unknown_categorical_method = 'least_frequent'
                combine_rare_levels = False
                rare_level_threshold = 0.1
                # fix_imbalance = False
                
                if len(categorical_columns) > 0:
                    st.markdown('<p style="color:#1386fc">Preprocessing for Categorical Columns:</p>',unsafe_allow_html=True)
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

                group_features = st.multiselect('Select Features that have Related Characteristics',feature_columns)
                group_features = group_features if len(group_features) > 0 else None
                bin_numeric_features = st.checkbox('Create new Features based on Bin Combinations', value=False)
                select_bin_numeric_features=None
                if bin_numeric_features:
                    select_bin_numeric_features = st.multiselect('Select Numeric Features Transformed into Categorical Features using K-Means',
                                                    feature_columns)
                
        # Feature Selection
        with st.beta_expander("Select Features in Dataset Contributes the most in Predicting Target Variable"):
            with st.beta_container():       
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
                   
        st.subheader("Start Loading, Preprocessing and Transforma Dataset:")
        with st.beta_container():          
            button_run = st.button("Start Process and Transform")
            if button_run:
                with st.spinner("Preprocessing..."):
                    setup(data=df,preprocess=True,
                    categorical_imputation=categorical_imputation, numeric_imputation=numeric_imputation,
                    normalize=normalize,normalize_method=normalize_method, transformation=transformation,
                    transformation_method=transformation_method,
                    unknown_categorical_method=unknown_categorical_method,
                    combine_rare_levels=combine_rare_levels,rare_level_threshold=rare_level_threshold,
                    ignore_features=remove_columns,
                    group_features=group_features,
                    bin_numeric_features=select_bin_numeric_features,
                    remove_multicollinearity=remove_multicollinearity,
                    multicollinearity_threshold = multicollinearity_threshold,           
                    remove_perfect_collinearity=remove_perfect_collinearity,
                    pca=pca,pca_method=pca_method,pca_components=pca_components,
                    ignore_low_variance=ignore_low_variance,
                    html=False,silent=True)

                state.log_history = {"setup":pull(True).data.to_dict()} 
                # record the setup procedure
                state.is_set_up = True
                
            st.markdown('<p style="color:#1386fc">Do you want to check Transformed Data?</p>',unsafe_allow_html=True)
            button_transform = st.button("Check Transformed Data")
            try:
                if button_transform:
                    with st.spinner("Loading..."):
                        st.write(convert_dict_to_df(state.log_history["setup"]))
            except:
                st.error("Please Process and Transform Data first!")
        
        # st.subheader("Compare All the Machine Learning Model Result:")
        # with st.beta_expander("Select Parameters for Comparing Models"):
        #     with st.beta_container():
        #         fold_text = st.text_input('Control Cross Validation Folds (int or None)', value='None')
        #         fold_compare = None if fold_text == 'None' else int(fold_text)

        #         cross_validation = st.checkbox('Allow Cross Validation or not', value=True)
        #         sort = st.selectbox('The Sort Order of the Score Grid', options=['R2','MAE','MSE','RMSE','RMSLE','MAPE'], )

        # st.markdown('<p style="color:#1386fc">Compare All the Machine Learning Models based on selected Metrics.</p>',unsafe_allow_html=True)     
        # button_compare = st.button("Compare Models")
        # try:
        #     if button_compare:
        #         with st.spinner('Comparing all Models...'):
        #             state.best = compare_models(exclude=['xgboost'],fold=fold_compare, cross_validation=cross_validation, sort=sort)
        #             state.log_history["compare_models"] = pull(True).to_dict()
        # except:
        #     st.error("Please Process and Transform Data first!")

        # st.markdown('<p style="color:#1386fc">Show All the Metrics Results.</p>',unsafe_allow_html=True)       
        # button_model = st.button("Show All Result")    
        # try:  
        #     if button_model:
        #         with st.spinner("Show All the Results..."):
        #             st.write(convert_dict_to_df(state.log_history["compare_models"]))
        # except:
        #     st.error("Please Compare All Models first!")
        

        return state

    else:
        return st.error("Something went wrong! Please upload dataset first!")
