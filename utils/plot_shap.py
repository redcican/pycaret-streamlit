import streamlit as st
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import warnings


def st_shap(plot, height=None):
    # plot the shap diagram in stramlit
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def plot_reg_shap_global_and_local(shap_plot_type:str, model:object,  X_train, plot_type:str=None, 
                               max_display:int=None, index_of_explain:int=0):
    """plot the global shap diagram

    Args:
        shap_plot_type: str, the type of shap plot [global | local | scatter]
        model: a trained pycaret model
        plot_type (str): the type of plot   
        max_display (int): the max number rows to display
        X_train (pd.DataFrame): the X training dataset
        index_of_explain (int) : the index of explanation of prediction
    """
    explainer, shap_values, sample_values = get_reg_shap_explainer_global_and_local(model, X_train)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    if model.__class__.__name__ == "CatBoostRegressor":       
        if shap_plot_type == "global":
            if plot_type == "default":
                plot_type=None
            shap.summary_plot(shap_values, X_train, plot_type=plot_type,show=False)
        else:
            plt.clf()
            st_shap(shap.force_plot(explainer.expected_value, shap_values[index_of_explain,:],
                                    X_train.iloc[index_of_explain,:]))
            
    elif model.__class__.__name__ == "RANSACRegressor" \
        or model.__class__.__name__ == "KernelRidge" \
        or model.__class__.__name__ == "SVR" \
        or model.__class__.__name__ == "MLPRegressor" \
        or model.__class__.__name__ == "KNeighborsRegressor" \
        or model.__class__.__name__ == "AdaBoostRegressor":
            if shap_plot_type == "global":
                if plot_type == "default":
                    plot_type=None
                shap.summary_plot(shap_values, sample_values,plot_type=plot_type,show=False)
            else:
                plt.clf()
                st_shap(shap.force_plot(explainer.expected_value, shap_values[index_of_explain],
                                        sample_values.iloc[index_of_explain]))
            
    else:
        if shap_plot_type == "global": 
            if plot_type == 'bar':
                shap.plots.bar(shap_values, max_display=max_display,show=False)
            elif plot_type == 'beeswarm':
                shap.plots.beeswarm(shap_values, max_display=max_display,show=False)
            else:
                shap.plots.heatmap(shap_values, max_display=max_display,show=False)
        elif shap_plot_type == "local":
            shap.plots.waterfall(shap_values[index_of_explain],max_display=max_display,show=False) 

    return st.pyplot(fig)


def get_shap_kernel(estimator:object,X_train):
    """compute the shap value importance for non-tree based model

    Args:
        estimator (a none tree based sklearn estimator): a sklearn non tree based estimator
        x_train ((pd.DataFrame, np.ndarray),): X training data
        x_test ((pd.DataFrame, np.ndarray),): X testing data

    Returns:
        shap plot
    """
    warnings.filterwarnings("ignore")
    # because the kernel explainer for non-tree based model extremly slower
    # so we must use kmeans to extract mainly information from x_train
    # to speed up the calculation
    if X_train.shape[1] > 3:
        x_train_summary = shap.kmeans(X_train,3)
    else:
        x_train_summary = shap.kmeans(X_train,X_train.shape[1])
    explainer = shap.KernelExplainer(estimator.predict,x_train_summary)

    size = len(X_train)
    if size < 50:
        size = size
    elif size * 0.2 > 50:
        size = 50
    else:
        size = int(size * 0.2)
    sample_values = shap.sample(X_train, size)
    shap_values = explainer.shap_values(sample_values, lr_reg='num_features(10)')

    return explainer, shap_values,sample_values
 
 
@st.cache(allow_output_mutation=True)
def get_reg_shap_explainer_global_and_local(model: object, X_train):
    """return the shap explainer object and shap values for
       global and local plot

    Args:
        model (object): a traine pycaret model
        X_train (pd.DataFrame): the X training data
    """
    sample_values = None
    
    if model.__class__.__name__ == "CatBoostRegressor": 
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    elif model.__class__.__name__ == "RANSACRegressor" \
        or model.__class__.__name__ == "KernelRidge" \
        or model.__class__.__name__ == "SVR" \
        or model.__class__.__name__ == "MLPRegressor" \
        or model.__class__.__name__ == "KNeighborsRegressor" \
        or model.__class__.__name__ == "AdaBoostRegressor":
            explainer, shap_values, sample_values = get_shap_kernel(model, X_train)
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        
    return explainer, shap_values, sample_values

    
 
@st.cache(allow_output_mutation=True)
def get_cls_shap_explainer_global_and_local(model: object, X_train, task_type:str):
    """return the shap explainer object and shap values for
       global and local plot classification

    Args:
        model (object): a traine pycaret model
        X_train (pd.DataFrame): the X training data
        task_type: (str): a binary or a multiclasses problem
    """
    sample_values = None
    
    if task_type == 'Binary':
        if model.__class__.__name__ == "ExtraTreesClassifier" \
            or model.__class__.__name__ == "CatBoostClassifier" \
            or model.__class__.__name__ == "RandomForestClassifier" \
            or model.__class__.__name__ == "DecisionTreeClassifier": 
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            
        elif model.__class__.__name__ == "KNeighborsClassifier" \
                or model.__class__.__name__ == "AdaBoostClassifier" \
                or model.__class__.__name__ == "QuadraticDiscriminantAnalysis" \
                or model.__class__.__name__ == "NaiveBayes" \
                or model.__class__.__name__ == "GaussianProcessClassifier" \
                or model.__class__.__name__ == "MLPClassifier":
                explainer, shap_values, sample_values = get_shap_kernel(model, X_train)
        else:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)
    else:
        if model.__class__.__name__ == "ExtraTreesClassifier" \
            or model.__class__.__name__ == "CatBoostClassifier" \
            or model.__class__.__name__ == "RandomForestClassifier" \
            or model.__class__.__name__ == "DecisionTreeClassifier" \
            or model.__class__.__name__ == "ExtremeGradientBoosting" \
            or model.__class__.__name__ == "LightGradientBoostingMachine":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        else:
            explainer, shap_values, sample_values = get_shap_kernel(model, X_train)

        
    return explainer, shap_values, sample_values



def plot_cls_shap_global_and_local(shap_plot_type:str, model:object,  X_train, task_type,
                                   plot_type:str=None, 
                                   max_display:int=None, index_of_explain:int=0):
    """plot the global shap diagram for classification problem

    Args:
        shap_plot_type: str, the type of shap plot [global | local | scatter]
        model: a trained pycaret model
        plot_type (str): the type of plot   
        max_display (int): the max number rows to display
        X_train (pd.DataFrame): the X training dataset
        task_type: (str) a binary or multiclass problem
        index_of_explain (int) : the index of explanation of prediction
    """
    explainer, shap_values, sample_values = get_cls_shap_explainer_global_and_local(model, X_train,task_type)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    if task_type == 'Binary':
        
        if model.__class__.__name__ == "ExtraTreesClassifier" \
            or model.__class__.__name__ == "CatBoostClassifier" \
            or model.__class__.__name__ == "RandomForestClassifier" \
            or model.__class__.__name__ == "DecisionTreeClassifier":     
                
            if shap_plot_type == "global":
                if plot_type == "default":
                    plot_type=None
                # only catboost classifier support 'bar','default' and 'violin'
                if model.__class__.__name__ == "CatBoostClassifier":
                    shap.summary_plot(shap_values, X_train, plot_type=plot_type,show=False)
                else:
                    shap.summary_plot(shap_values, X_train, plot_type=None,show=False)
            else:
                if model.__class__.__name__ == "CatBoostClassifier":
                    st_shap(shap.force_plot(explainer.expected_value, shap_values[index_of_explain,:],
                                            X_train.iloc[index_of_explain,:]))
                else:
                    plt.clf()
                    st_shap(shap.force_plot(explainer.expected_value[index_of_explain], shap_values[index_of_explain],
                                            X_train),height=1000)
            
        elif model.__class__.__name__ == "KNeighborsClassifier" \
            or model.__class__.__name__ == "AdaBoostClassifier" \
            or model.__class__.__name__ == "QuadraticDiscriminantAnalysis" \
            or model.__class__.__name__ == "NaiveBayes" \
            or model.__class__.__name__ == "GaussianProcessClassifier" \
            or model.__class__.__name__ == "MLPClassifier":
            if shap_plot_type == "global":
                if plot_type == "default":
                    plot_type=None
                shap.summary_plot(shap_values, sample_values,plot_type=plot_type,show=False)
            else:
                plt.clf()
                st_shap(shap.force_plot(explainer.expected_value, shap_values[index_of_explain],
                                            sample_values.iloc[index_of_explain]))
                
        else:
            if shap_plot_type == "global": 
                if plot_type == 'bar':
                    shap.plots.bar(shap_values, max_display=max_display,show=False)
                elif plot_type == 'beeswarm':
                    shap.plots.beeswarm(shap_values, max_display=max_display,show=False)
                else:
                    shap.plots.heatmap(shap_values, max_display=max_display,show=False)
            elif shap_plot_type == "local":
                shap.plots.waterfall(shap_values[index_of_explain],max_display=max_display,show=False) 
    else:
        if model.__class__.__name__ == "ExtraTreesClassifier" \
            or model.__class__.__name__ == "CatBoostClassifier" \
            or model.__class__.__name__ == "RandomForestClassifier" \
            or model.__class__.__name__ == "DecisionTreeClassifier" \
            or model.__class__.__name__ == "ExtremeGradientBoosting" \
            or model.__class__.__name__ == "LightGradientBoostingMachine":
                if shap_plot_type == "global":
                    shap.summary_plot(shap_values, X_train,plot_type=None,show=False)
                else:
                    plt.clf()
                    st_shap(shap.force_plot(explainer.expected_value[index_of_explain], shap_values[index_of_explain],
                                            X_train),height=1000)    
        else:
            if shap_plot_type == "global":
                if plot_type == "default":
                    plot_type=None
                shap.summary_plot(shap_values, sample_values,plot_type=plot_type,show=False)
            else:
                plt.clf()
                st_shap(shap.force_plot(explainer.expected_value, shap_values[index_of_explain],
                                            sample_values.iloc[index_of_explain]))
                            
                   
    return st.pyplot(fig)

