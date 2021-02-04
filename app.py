import streamlit as st
from PIL import Image
from pages import home,data_info, preprocessing,prediction, training, model_analysis
from utils.session import _get_state
    
PAGES = {
    "Home": home,
    "DataInfo": data_info,
    "Preprocessing": preprocessing,
    "Training" : training,
    "ModelAnalysis": model_analysis,
    "Prediction and Save": prediction,
}

def load_image():
    image_eido = Image.open('EIDOlogo.png')
    st.sidebar.image(image_eido, use_column_width=True)

    
def run():
    state = _get_state()
    st.set_page_config(
        page_title="EidoData App",
        page_icon='icons.ico',
        layout="centered",
        initial_sidebar_state='expanded'
    )
    load_image()
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection == "Home":
        state_df = PAGES[selection].write(state)
        state.df = state_df
    if selection == "DataInfo":
        PAGES[selection].write(state.df)
    if selection == "Preprocessing":
        PAGES[selection].write(state)
    if selection == "Training":
        PAGES[selection].write(state)
    if selection == "ModelAnalysis":
        PAGES[selection].write(state)
    if selection == "Prediction and Save":
        PAGES[selection].write(state)
    st.write(state.__dict__)
    state.sync()


if __name__ == '__main__':
    run()