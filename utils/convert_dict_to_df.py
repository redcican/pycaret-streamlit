import pandas as pd

def convert_dict_to_df(dictionary:dict):
    """convert dict to pd.DataFrame and show them in streamlit

    Args:
        dictionary (dict): [description]
    """
    df = pd.DataFrame.from_dict(dictionary)
    return df