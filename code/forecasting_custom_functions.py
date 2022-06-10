from sktime.utils.plotting import plot_series
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random
from sktime.forecasting.naive import NaiveForecaster

def pred_rmse(y_test, preds):
    """ This function generates the root mean squared error for the 2022Q2 predictions against the actual sales and all models.
    """
    return mean_squared_error(y_test.iloc[0,:], preds.iloc[0,:],squared=False)


def plot_pred(y_train, y_test, preds):
    """ This function generates 10 plots for random models in the dataset.
    """
    for i in [random.randint(0,y_train.shape[1]) for _ in range(10)]:
        plot_series(y_train.iloc[:,i], y_test.iloc[:,i], preds.iloc[:,i], labels = ['train','test','preds']);
        
        
def forecast(y):
    """ This function generates the next month's predictions for a dataframe containing monthly sales.
    Method: uses a naive forecasting method using the average of the last 6 months of sales. If sales 
    had not begun until months 1-5 prior, it will only uses the non-zero values.
    
    args:
        y (Pandas DataFrame): a dataframe containing past sales of products, where the columns
        are the product names, and the rows are monthly sales in datetime format withly monthly frequency.
    
    returns:
        preds (Pandas DataFrame): a dataframe containing predictions for the next month for each model
    """
    # Create a copy of y with nan to fill with predictions 
    preds = pd.DataFrame(data = y.iloc[[-1],:], columns = y.columns)
    preds.index = preds.index + 1
    preds.iloc[0,:] = np.nan

    rest = y.copy()
    x = -7
    window_length = 6
    while window_length > 0 :
        # assemble a list of models which had sales in the month (starting at 6 months prior and increasing to 2 months prior)
        collection = rest.iloc[x,:][rest.iloc[x,:] > 0].index
        if len(collection) > 0:
            model = NaiveForecaster(strategy='mean',window_length = window_length)
            model.fit(rest[collection])
            model_preds = model.predict(fh=1)
            model_preds.columns = collection
            # fill the pred with the prediction
            for model in collection:
                preds[model] = model_preds[model].values[0]
            rest = rest.drop(columns=collection)
        x += 1
        window_length -= 1
    # Fill the models which had no sales in this period with 0.
    preds.fillna(0,inplace=True)

    return preds
    