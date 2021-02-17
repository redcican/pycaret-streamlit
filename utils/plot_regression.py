import plotly.graph_objects as go
import numpy as np

def get_plotly_act_vs_predict(estimator,X_train,X_test,y_train,y_test):
    """draw the actual vs predction for regression plot

    Args:
        estimator: (object) trained pycaret ml model
        y_train ((pd.DataFrame, np.ndarray)): y training data
        y_train_pred ((pd.DataFrame, np.ndarray)): prediction on y training data
        y_test ((pd.DataFrame, np.ndarray)): y testing data
        y_test_pred ((pd.DataFrame, np.ndarray)): prediction on y testing data

    Returns:
        str: plotly figure object
    """

    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(y_train),y=np.array(y_train_pred),mode='markers',name="Train"))
    fig.add_trace(go.Scatter(x=np.array(y_test), y=np.array(y_test_pred),mode='markers',name="Test"))

    fig.add_shape(type="line",
        x0=0, y0=0, x1=1, y1=1,xref="paper",yref="paper",name="Identity",
        line=dict(
            color="black",width=2,dash="dot"))
    fig.update_layout(
        title={
            'text':"Actual Value vs. Predicted Value",
            'xanchor':'center',
            'yanchor': 'top',
            'x': 0.5},
        xaxis_title="Actual",
        yaxis_title="Predicted",
        margin=dict(l=40, r=40, t=40, b=40),
        width=1000)
    
    return fig


def gauge_plot(original_value,optimal_value, lower_bound, 
               upper_bound, min_value, max_value):
    """plot the gauge plot for backwards Analysis regression problem

    Args:
        original_value (float or int): the original Y value to optimize
        optimal_value (float or int): the optimal value found in generated data
        lower_bound (float or int): the lower bound value to optimize 
        upper_bound (float or int): the upper bound value to optimize
        min_value (float or int): the minimum value of target column
        max_value (float or int): the maximum value of target column    
    Returns:
        [object]: plotly gauge object to show
    """
    if original_value > optimal_value:
        delta = {'reference': original_value, 'increasing': {'color': "RebeccaPurple"}}
    else:
        delta = {'reference': original_value, 'decreasing': {'color': "RebeccaPurple"}}   
    
    
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = optimal_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Optimal", 'font': {'size': 24}},
        
        delta = delta,
        gauge = {
            'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                # {'range': [min_value, mean_value], 'color': 'cyan'},royalblue
                {'range': [lower_bound, upper_bound], 'color': 'cyan'}]}))
            # 'threshold': {
            #     'line': {'color': "red", 'width': 4},
            #     'thickness': 0.75,
            #     'value': max_value-reference_value}}))
    fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
    return fig


def find_top_5_nearest(array, value):
    """Find the top 5 closest neighbors of given optimal value

    Args:
        array (np.array): the generated data with prediction
        value (int or float): optimal value to find

    Returns:
        list: the top 5 indices of suggested value
    """
    array = np.asarray(array)
    diff = np.abs(array - value)
    indices = np.argsort(diff)
    return indices