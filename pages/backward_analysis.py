from logging import log
import streamlit as st
from sdv.tabular import GaussianCopula,CopulaGAN,CTGAN,TVAE
from pycaret.regression import predict_model
from utils.convert_str_to_list import convert_str_to_list
from utils.plot_regression import gauge_plot,find_top_5_nearest
from utils.download_button import download_button
from st_aggrid import AgGrid

def write(state):
    if state.trained_model is not None:
        
        X_before_preprocess = state.X_before_preprocess
        target_name = state.y_before_preprocess
        df_X = X_before_preprocess.drop(target_name,axis=1)
        trained_model = state.trained_model
        min_value = X_before_preprocess[target_name].min()
        max_value = X_before_preprocess[target_name].max()
        mean_value = X_before_preprocess[target_name].mean()
        original_value = optimal_value = mean_value 
        state.optimal_value = 0

        st.header("Knowledge Generation and Backward Analysis.")
        with st.expander("Knowledge Generation"):
            st.markdown('<p style="color:#1386fc">Please Select a Method to Generate Data.</p>',unsafe_allow_html=True)     
            sdv_method = st.selectbox('Method to Generate Data', options=['GaussianCopula','CTGAN','CopulaGAN','TVAE'])
            sample = st.number_input('How Many Samples of Data to Generate?', min_value =1, value=df_X.shape[0],key=1)

            if sdv_method == 'GaussianCopula':
                model = GaussianCopula()
            else:
                is_tune = st.checkbox("Do You Want to Tune Hyperparameters?", value=False)
                if sdv_method == 'CopulaGAN' or sdv_method == 'CTGAN':
                    epochs = 300
                    batch_size=500
                    log_frequency=True
                    embedding_dim=128
                    generator_dim= (256,256)
                    discriminator_dim = (256,256)
                    generator_lr=0.0002
                    generator_decay=1e-6
                    discriminator_lr = 0.0002
                    discriminator_decay=1e-6
                    discriminator_steps = 1
                    
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
                                        embedding_dim=embedding_dim,generator_dim=generator_dim,discriminator_dim=discriminator_dim,
                                        generator_lr=generator_lr,generator_decay=generator_decay,
                                        discriminator_lr=discriminator_lr,discriminator_decay=discriminator_decay,
                                        discriminator_steps=discriminator_steps)
                    if sdv_method == 'CTGAN':
                        model = CTGAN(epochs=epochs, batch_size=batch_size,log_frequency=log_frequency,
                                        embedding_dim=embedding_dim,generator_dim=generator_dim,discriminator_dim=discriminator_dim,
                                        generator_lr=generator_lr,generator_decay=generator_decay,
                                        discriminator_lr=discriminator_lr,discriminator_decay=discriminator_decay,
                                        discriminator_steps=discriminator_steps)
                else:
                    compress_dims =decompress_dims=(128,128)
                    epochs=300
                    batch_size=500
                    embedding_dim=128
                    l2_scale=1e-5
                    if is_tune:
                        epochs = st.number_input("Number of Training Epochs (int)", min_value=1, value=300,key=2)
                        batch_size = st.number_input("Number of Data Samples to Process, should be a multiple of 10 (int)", min_value =1, value=500,key=2)
                        embedding_dim = st.number_input("Size of the Random Sample Passed to the Generator (int)", min_value=1, value=128,key=2)
                        compress_dims  = st.text_input("Size of Each Hidden Layer in the Encoder (int)", value="128,128")
                        decompress_dims  = st.text_input("Size of Each Hidden Layer in the Decoder (int)", value="128,128")
                        l2_scale  = st.number_input("Regularization term", min_value=0.0, value=1e-5, format="%e")
                        
                        compress_dims = convert_str_to_list(compress_dims)
                        decompress_dims = convert_str_to_list(decompress_dims)
                    model = TVAE(embedding_dim=embedding_dim, compress_dims=compress_dims, decompress_dims=decompress_dims, 
                                 l2scale=l2_scale, batch_size=batch_size, epochs=epochs)

            button_generate = st.button("Generate")
            if button_generate:
                with st.spinner("Generating..."):
                    model.fit(df_X)
                    new_data = model.sample(sample)
                    new_data_prediction = predict_model(trained_model,new_data)
                    AgGrid(new_data_prediction)
                    state.new_data_prediction = new_data_prediction
                    
            button_download = st.button("Download Generated Data")
            if button_download:
                file_extension = st.selectbox("Choose Csv or Excel File to Download", options=[".csv",".xlsx"])
                file_name = st.text_input("File Name",value="prediction",key=1)
                if file_name:
                    href = download_button(state.new_data_prediction, file_name, "Download",file_extension)
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("File Name cannot be empty!") 
                
        st.markdown("---")
        with st.expander("Backward Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Please Select a Index for Data to Optimize")
                index = st.number_input("Index of Data", min_value=0, value=0,max_value=df_X.shape[0]-1,key=1)
                st.write(X_before_preprocess.iloc[index])
                original_value = X_before_preprocess.iloc[index].loc[target_name]

            with col2:
                st.subheader("Optimize")
                lower_bound = st.number_input("The Lower Bound Value to Optimize",value=min_value)
                upper_bound = st.number_input("The Upper Bound Value to Optimize",value=max_value)
                button_optimize = st.button("Optimizer")
                if button_optimize:
                    if state.new_data_prediction is not None:
                        new_prediction = state.new_data_prediction['Label']
                        indices = find_top_5_nearest(new_prediction, original_value)
                        optimal_value = new_prediction[indices[0]]
                        state.suggest_indices = indices
                        state.optimal_value = optimal_value
                    else:
                        st.error("Please Generate New Data first!")
                
        with st.container():
            # state.optimal_value = state.optimal_value if state.optimal_value is not None else 0
            fig = gauge_plot(original_value,state.optimal_value,lower_bound,
                             upper_bound,min_value,max_value)
            st.plotly_chart(fig)
            button_suggest = st.button("Show the Top 5 Suggestions")
            if button_suggest:
                suggestion = state.new_data_prediction.iloc[state.suggest_indices[:5]]
                AgGrid(suggestion)
    else:
        st.error("Please Train a Model first!")
        
