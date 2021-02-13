from pycaret.regression import models as regression_models
from pycaret.classification import models as classification_models
from pycaret.clustering import models as clustering_models

def retrieve_models_name(type:str):
    """retrive the abbreviated name for regression models and classification models

    Args:
        pycaret_name (str): the class name from pycaret modules
    """
    if type == "Regression":
        all_models = regression_models() 
    elif type == "Classification":
        all_models = classification_models()
    else:
        all_models = clustering_models()
    dict_models = dict(zip(all_models.Name, all_models.index))
    models = {k.replace(' ','') :v for k, v in dict_models.items()}
    return models
