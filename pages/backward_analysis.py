from logging import log
import streamlit as st
from sdv.tabular import GaussianCopula,CopulaGAN
from pycaret.regression import predict_model

def write(state):
    if state.trained_model is not None:
        
        X_before_preprocess = state.X_before_preprocess
        target_name = state.y_before_preprocess
        df_X = X_before_preprocess.drop(target_name,axis=1)
        trained_model = state.trained_model

        st.header("Knowledge Generation and Backward Analysis.")
        with st.beta_expander("Knowledge Generation"):
            st.markdown('<p style="color:#1386fc">Please Select a Method to Generate Data.</p>',unsafe_allow_html=True)     
            sdv_method = st.selectbox('Method to Generate Data', options=['GaussianCopula','CTGAN','CopulaGAN'])
            sample = st.number_input('How Many Samples of Data to Generate?', min_value =1, value=df_X.shape[0],key=1)

            if sdv_method == 'GaussianCopula':
                model = GaussianCopula()
            elif sdv_method == 'CopulaGAN':
                is_tune = st.checkbox("Do You Want to Tune Hyperparameters?", value=False)
                if is_tune:
                    epochs = st.number_input("Number of Training Epochs (int)", min_value=1, value=300)
                    batch_size = st.number_input("Number of Data Samples to Process, should be a multiple of 10 (int)", min_value =1, value=500)
                    log_frequency = st.checkbox('Whether to Use Log Frequency', value=True)
                    embedding_dim = st.number_input("Size of the Random Sample Passed to the Generator (int)", min_value=1, value=128)
                    generator_dim  = st.text_input("Size of the Generator Residual Layer (int)", value="(256,256)")
                    discriminator_dim = st.text_input("Size of the Discriminator Residual Layer (int)", value="(256,256)")
                    generator_lr = st.number_input("Learning Rate for the Generator", min_value=0.0, value=2e-4, format="%e")
                    generator_decay  = st.number_input("Generator Weight Decay for the Adam Optimizer", min_value=0.0, value=1e-6, format="%e")
                    discriminator_lr   = st.number_input("Learning Rate for the Discriminator", min_value=0.0, value=2e-4, format="%e")
                    discriminator_decay   = st.number_input("Discriminator  Weight Decay for the Adam Optimizer", min_value=0.0, value=1e-6, format="%e")
                    discriminator_steps  = st.number_input("Number of Discriminator UUpdates to do for Each Generator Update (int)", min_value=1, value=1)
                    model = CopulaGAN(epochs=epochs, batch_size=batch_size,log_frequency=log_frequency,
                                      embedding_dim=embedding_dim,gen_dim=generator_dim,dis_dim=discriminator_dim)
                model = CopulaGAN()
            else:
                pass
            
            button_generate = st.button("Generate")
            if button_generate:
                model.fit(df_X)
                new_data = model.sample(sample)
                new_data_prediction = predict_model(trained_model,new_data)
                st.write(new_data_prediction)
    else:
        st.error("Please Train a Model first!")