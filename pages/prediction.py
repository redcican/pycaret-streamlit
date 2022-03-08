import streamlit as st
from utils.retrieve_element_for_prediction import retrieve_train_element
from utils.download_button import download_button
import joblib
from pathlib import Path
import pandas as pd


def write(state):
    
    if state.task == "Regression":
        from pycaret.regression import predict_model, finalize_model,save_model
        target_name = state.y_before_preprocess
    elif state.task == "Classification":
        from pycaret.classification import predict_model, finalize_model,save_model
        target_name = state.y_before_preprocess
    else:
        from pycaret.clustering import predict_model,save_model
        target_name = None
        
    def online_predict(model, input_df,target_type):
        """make prediction on online data

        Args:
            model (object): a trained model
            input_df (pd.DataFrame): the input dataframe for predicitons
            target_type (str): the type of training target

        Returns:
            str: predcition
        """
        prediction_df = predict_model(model, data=input_df)
        if target_type == 'Regression' or target_type == 'Classification':
            predictions = prediction_df['Label'][0]
        else:
            predictions = prediction_df['Cluster'][0]
        return predictions        

    if state.trained_model is not None:
        st.header("Make a Prediction on Given Input or Upload a File.")

        add_selectbox = st.sidebar.selectbox(
            "How would you like to predict?",
            ("Online", "Batch", "SaveModel")
        )

        X_before_preprocess = state.X_before_preprocess
        target_name = target_name
        ignore_columns = state.ignore_columns
        trained_model = state.trained_model      
        
        if add_selectbox == "Online":
            with st.spinner("Predicting ..."):
                input_df = retrieve_train_element(X_before_preprocess, target_name, ignore_columns,state.task)
                output = ""
                if st.button("Predict"):
                    output = online_predict(trained_model, input_df,state.task)
                    output = str(output)
                    st.success(f'The Prediction is **{output}**')
        
        if add_selectbox == 'Batch':
            file_upload = st.file_uploader('Upload csv file for prediciton', type=["csv", "xlsx"])
            if file_upload is not None:
                file_extension = file_upload.name.split('.')[1]
                if file_extension == "csv":
                    data = pd.read_csv(file_upload)
                else:
                    data = pd.read_excel(file_upload)
                predictions = predict_model(trained_model, data=data)
                st.write(predictions)  
                
                is_download = st.checkbox("Do You Want to Download the Prediction File?", value=False)
                if is_download:
                    file_extension = st.selectbox("Choose Csv or Excel File to Download", options=[".csv",".xlsx"])
                    file_name = st.text_input("File Name",value="prediction",key=1)
                    if file_name:
                        href = download_button(predictions, file_name, "Download",file_extension)
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error("File Name cannot be empty!") 
        
        if add_selectbox == "SaveModel":
            is_download = st.checkbox("Do You Want to Download the Model?", value=False)
            if is_download:
                file_name = st.text_input("File Name",value="",key=2)
                is_finalize = st.checkbox("Do You Want to Finalize the Model (not for clustering)?", value=False)
                if file_name:
                    if is_finalize:
                        finalized_model = finalize_model(trained_model)
                        _,name = save_model(finalized_model, file_name)
                    else:
                        _,name = save_model(trained_model, file_name)
                    with open(name, "rb") as f:
                        e = joblib.load(f)
                    href = download_button(e, file_name, "Download",".pkl",pickle_it=True)
                    st.markdown(href, unsafe_allow_html=True)
                    
                    remove_cache = st.checkbox("Remove the Cache?", value=False)
                    if remove_cache:
                        p = Path(".").glob("*.pkl")
                        for filename in p:
                            filename.unlink()
                        if len(list(p)) == 0:
                            st.success("Delete the Cache File from Local Filesystem!")
                            st.balloons()
                else:
                    st.error("Please Give a File Name first!")
                

    else:
        st.error("Please Train a Model first!")
    



