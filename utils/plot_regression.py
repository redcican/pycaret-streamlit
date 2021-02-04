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