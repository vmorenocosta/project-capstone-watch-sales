# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.impute import KNNImputer

from custom_functions import *

def calc_norm_sales(df_models):
    """ This function calculates a target value - Normalized Sales, which is the overall sales divided by the number of years that the model was sold. 
    The function modifies the dataframe in place
    
    args:
        df_models (Pandas dataframe): a dataframe which contains monthly sales
        
    return:
        df_models (Pandas dataframe): the dataframe, now containing overall sales and norm_sales
    """
    # Calculate overall sales
    df_models = df_models.merge(pd.DataFrame(return_date_col(df_models.set_index('style_id')).T.sum(),columns=['overall_sales']),left_on = 'style_id',right_index=True)

    # Create new dataframe with yearly sales
    yearly_sales = return_sales(df_models,'style_id').resample('Y').sum().T
    norm_sales = []
    for i,x in enumerate(yearly_sales.index):
        years_of_sales = sum([1 for sales in yearly_sales.loc[x,yearly_sales.iloc[:,0:-1].columns] if sales > 0])
        if yearly_sales.iloc[i,-1] > 0:
            years_of_sales += 4/12
        if years_of_sales == 0:
            norm_sales.append(0)
        else:
            norm_sales.append(df_models.loc[i,'overall_sales']/years_of_sales)
    yearly_sales['norm_sales'] = norm_sales

    plt.title('Normalized Sales')
    plt.hist(norm_sales)
    plt.xlabel('Sales/years sold')
    plt.ylabel('Num models');
    
    return df_models.merge(yearly_sales[['norm_sales']],left_on = 'style_id',right_index=True)

    

def plot_preds(y_test, preds, model_name):
    """ This function creates two plots: the predictions against the actual values, and the predictions against the residuals.
    
    Args:
        y_test (Pandas series): actual values
        preds (Pandas series): predictions
    """    
    residuals = y_test - preds
    plt1 = plt.figure()
    plt.scatter(preds,y_test)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=True, scaley=True)
    plt.xlabel('Predicted sales')
    plt.ylabel('Actual sales')
    plt.title(f'Predicted vs Actual Sales: {model_name}');

    plt2 = plt.figure()
    plt2.suptitle(f'Residuals vs predicted sales: {model_name}')
    sns.residplot(x = y_test,
                  y = residuals,
                  lowess = True,
                  line_kws = {'color':'red'})
    plt.xlabel('Predicted sales')
    plt.ylabel('Residuals');
    
def create_evaluate_model(model, X_train, X_test, y_train, y_test):
    """ This function accepts X_train, X_test, y_train, y_test, and a regression model instance to preprocess data and create, fit, and evaluate the model.
    
    args:
        model (function): an instance of a regression model
        X_train (Pandas dataframe): a dataframe that contains training variables to fit a model
        X_test (Pandas dataframe): a dataframe that contains testing variables to evaluate a model
        y_train (Pandas series): a dataframe that contains the training variable to be predicted by the model for evaluation
        y_test (Pandas series): a dataframe that contains the testing variable to be predicted by the model for evaluation
    """
    # Print model name
    print(model[1])
    
    # Create a column transformer to one hot encode categorical variables
    categorical_attributes = []
    for attribute in X_train.columns:
        try:
            int(attribute)
        except:
            categorical_attributes.append(attribute)
            
    ct = ColumnTransformer([('ohe',OneHotEncoder(drop='first',sparse=False,handle_unknown='ignore'),
                         categorical_attributes)],
                      remainder='passthrough',
                      verbose_feature_names_out=False)
    
    # Create a pipeline to transform, scale, and model
    pipe = Pipeline([
        ('ct',ct),
        ('ss',StandardScaler()),
        model
    ])

    # Fit model with training data
    pipe.fit(X_train,y_train)
    
    # Print traw and test score
    print('Train score:', pipe.score(X_train,y_train))
    print('Test raw score:', pipe.score(X_test,y_test))

    # Create predictions with test data
    preds = pipe.predict(X_test)

    # Zero out any negative predictions and predictions above upper bound
    upper_bound = 300
    preds = [x if x > 0 else 0 for x in preds]
    preds = [x if x < upper_bound else 0 for x in preds]

    # Evalute altered predictions
    test_cleaned_rmse = mean_squared_error(y_test, preds,squared=False)
    print('Test cleaned score:', r2_score(y_test, preds))
    print('Test cleaned RMSE:',test_cleaned_rmse)
    print('Baseline RMSE:', mean_squared_error(y_test,np.full_like(y_test, y_train.mean()),squared=False))
    
    # Plot predictions using custom function
    plot_preds(y_test, preds, model[0])
    
    print()
    return preds, test_cleaned_rmse