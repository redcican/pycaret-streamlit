from logging import log
import streamlit as st
# from sdv.tabular import GaussianCopula,CopulaGAN,CTGAN,TVAE
from sdv.tabular import GaussianCopula
from pycaret.regression import predict_model
from utils.convert_str_to_list import convert_str_to_list

def write(state):
    if state.trained_model is not None:
        
        X_before_preprocess = state.X_before_preprocess
        target_name = state.y_before_preprocess
        df_X = X_before_preprocess.drop(target_name,axis=1)
        trained_model = state.trained_model

        st.header("Knowledge Generation and Backward Analysis.")
        with st.beta_expander("Knowledge Generation"):
            st.markdown('<p style="color:#1386fc">Please Select a Method to Generate Data.</p>',unsafe_allow_html=True)     
            sdv_method = st.selectbox('Method to Generate Data', options=['GaussianCopula','CTGAN','CopulaGAN','TVAE'])
            sample = st.number_input('How Many Samples of Data to Generate?', min_value =1, value=df_X.shape[0],key=1)

            if sdv_method == 'GaussianCopula':
                model = GaussianCopula()
            else:
                is_tune = st.checkbox("Do You Want to Tune Hyperparameters?", value=False)
                if sdv_method == 'CopulaGAN' or sdv_method == 'CTGAN':
                    if is_tune:
                        epochs = st.number_input("Number of Training Epochs (int)", min_value=1, value=300,key=1)
                        batch_size = st.number_input("Number of Data Samples to Process, should be a multiple of 10 (int)", min_value =1, value=500,key=1)
                        log_frequency = st.checkbox('Whether to Use Log Frequency', value=True)
                        embedding_dim = st.number_input("Size of the Random Sample Passed to the Generator (int)", min_value=1, value=128,key=1)
                        generator_dim  = st.text_input("Size of the Generator Residual Layer (int)", value="256,256")
                        discriminator_dim = st.text_input("Size of the Discriminator Residual Layer (int)", value="256,256")
                        generator_lr = st.number_input("Learning Rate for the Generator", min_value=0.0, value=0.0002, format="%e")
                        generator_decay  = st.number_input("Generator Weight Decay for the Adam Optimizer", min_value=0.0, value=1e-6, format="%e")
                        discriminator_lr   = st.number_input("Learning Rate for the Discriminator", min_value=0.0, value=0.0002, format="%e")
                        discriminator_decay   = st.number_input("Discriminator  Weight Decay for the Adam Optimizer", min_value=0.0, value=1e-6, format="%e")
                        discriminator_steps  = st.number_input("Number of Discriminator Updates to do for Each Generator Update (int)", min_value=1, value=1)
                        
                        generator_dim = convert_str_to_list(generator_dim)
                        discriminator_dim = convert_str_to_list(discriminator_dim)
                    if sdv_method == 'CopulaGAN':
                        model = CopulaGAN(epochs=epochs, batch_size=batch_size,log_frequency=log_frequency,
                                        embedding_dim=embedding_dim,generator_dim=generator_dim,ddiscriminator_dim=discriminator_dim,
                                        generator_lr=generator_lr,generator_decay=generator_decay,
                                        discriminator_lr=discriminator_lr,discriminator_decay=discriminator_decay,
                                        discriminator_steps=discriminator_steps)
                    if sdv_method == 'CTGAN':
                        model = CTGAN(epochs=epochs, batch_size=batch_size,log_frequency=log_frequency,
                                        embedding_dim=embedding_dim,generator_dim=generator_dim,ddiscriminator_dim=discriminator_dim,
                                        generator_lr=generator_lr,generator_decay=generator_decay,
                                        discriminator_lr=discriminator_lr,discriminator_decay=discriminator_decay,
                                        discriminator_steps=discriminator_steps)
                else:
                    if is_tune:
                        epochs = st.number_input("Number of Training Epochs (int)", min_value=1, value=300,key=2)
                        batch_size = st.number_input("Number of Data Samples to Process, should be a multiple of 10 (int)", min_value =1, value=500,key=2)
                        embedding_dim = st.number_input("Size of the Random Sample Passed to the Generator (int)", min_value=1, value=128,key=2)
                        compress_dim  = st.text_input("Size of Each Hidden Layer in the Encoder (int)", value="128,128")
                        decompress_dim  = st.text_input("Size of Each Hidden Layer in the Decoder (int)", value="128,128")
                        l2_scale  = st.number_input("Regularization term", min_value=0.0, value=1e-5, format="%e")
                    compress_dim = convert_str_to_list(compress_dim)
                    decompress_dim = convert_str_to_list(decompress_dim)
                    model = TVAE(embedding_dim=embedding_dim, compress_dim=compress_dim, decompress_dim=decompress_dim, 
                                 l2scale=l2_scale, batch_size=batch_size, epochs=epochs)

            button_generate = st.button("Generate")
            if button_generate:
                model.fit(df_X)
                new_data = model.sample(sample)
                new_data_prediction = predict_model(trained_model,new_data)
                st.write(new_data_prediction)
    else:
        st.error("Please Train a Model first!")