from sktime.utils.plotting import plot_series
import pandas as pd
import numpy as np

def pred_rmse(y_test, preds):
    """ This function generates the root mean squared error for the 2022Q2 predictions against the actual sales and all models.
    """
    return mean_squared_error(y_test.iloc[0,:], preds.iloc[0,:],squared=False)


def plot_pred(y_train, y_test, preds):
    """ This function generates 10 plots for random models in the dataset.
    """
    for i in [random.randint(0,y_train.shape[1]) for _ in range(10)]:
        plot_series(y_train.iloc[:,i], y_test.iloc[:,i], preds.iloc[:,i], labels = ['train','test','preds']);
        
        
