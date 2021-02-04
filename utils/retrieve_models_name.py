from pycaret.regression import models as regression_models
from pycaret.classification import models as classification_models

def retrieve_models_name(is_regression:bool=True):
    """retrive the abbreviated name for regression models and classification models

    Args:
        pycaret_name (str): the class name from pycaret modules
    """
    all_models = regression_models() if is_regression else classification_models()
    dict_models = dict(zip(all_models.Name, all_models.index))
    models = {k.replace(' ','') :v for k, v in dict_models.items()}
    return models
