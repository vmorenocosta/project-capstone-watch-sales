# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

import warnings

from custom_functions import *

def calc_norm_yearly_sales(df_models, last_month_of_sales, year=2019):
    """ This function calculates a target value - normalized yearly sales, which is calculated by:
    1. Calculate yearly sales average
    2. Divide yearly sales by yearly average
    3. Sum the normalized sales
    4. Divide the sum by the total # years for which there were sales
    
    args:
        df_models (Pandas dataframe): a dataframe which contains monthly sales
        last_month_of_sales (int): the last month there were sales in the dataset
        year (int): the year to use to multiply normalized sales to get target value (default is 2019)
        
    return:
        df_models (Pandas dataframe): the dataframe, now containing overall sales and norm_sales
    """
    # Calculate overall sales
    df_models = df_models.merge(pd.DataFrame(return_date_col(df_models.set_index('style_id')).T.sum(),columns=['overall_sales']),left_on = 'style_id',right_index=True)

    # Create new dataframe with yearly sales
    yearly_sales = return_sales(df_models,'style_id').resample('Y',kind='period').sum().T

    # Calculate the yearly average sales
    avg_yearly_sales = yearly_sales[yearly_sales > 0].mean()

    # Calculate the total normalized yearly sales (note this number is the total sales divided by the yearly average)
    total_norm_sales = (yearly_sales/avg_yearly_sales).T.sum()

    # Add this to the dataframe
    df_models = df_models.merge(pd.DataFrame(total_norm_sales,columns=['total_norm_sales']),left_on = 'style_id',right_index=True)

    # Specify the last month of sales
    last_month_num = last_month_of_sales # 4 = april

    norm_sales = []
    for i,x in enumerate(yearly_sales.index):
        # Calculate number of years of sales for all except the last year
        years_of_sales = sum([1 for sales in yearly_sales.loc[x] if sales > 0])
        if yearly_sales.iloc[i,-1] > 0:
            years_of_sales += last_month_num/12
        if years_of_sales == 0:
            norm_sales.append(0)
        else:
            # Use 2019 yearly average to convert normalized sales to yearly sales
            norm_sales.append(avg_yearly_sales[str(year)]*df_models.loc[i,'total_norm_sales']/years_of_sales)
    yearly_sales['norm_yearly_sales'] = norm_sales
    
    plt.title('Histogram of normalized yearly sales by model')
    plt.hist(norm_sales)
    plt.xlabel('Normalized yearly sales')
    plt.ylabel('Count');
    
    return df_models.merge(yearly_sales[['norm_yearly_sales']],left_on = 'style_id',right_index=True)


def make_classes(target, class_upper):
    """ This function assigns a class to items in the dataframe based on a target variable and where it falls in the class upper bounds.
    
    args:
        target (Pandas Series): a series containing the target variable
        class_upper (dict): a dictionary where the keys are the classes and the values are the upper bounds for the target to the be in the class (inclusive)
    
    return:
        class_sales (list): a list of classes for each target
    """
    class_sales = []
    for i, sales in target.items():
        c = 0
        for c in class_upper.keys():
            if sales <= class_upper[c]:
                class_sales.append(c)
                break
            elif c == max(class_upper.keys()):
                class_sales.append(c+1)

    plt.hist(class_sales)
    plt.title('Number of models in each classes (excl. test data)')
    plt.xlabel('Classes')
    plt.ylabel('Count');

    return class_sales


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


def create_evaluate_model_class(model, X_train, X_test, y_train, y_test):
    """ This function accepts X_train, X_test, y_train, y_test, and a regression model instance to preprocess data and create, fit, and evaluate the model.
    
    args:
        model (function): an instance of a classfication model
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
    test_accuracy = pipe.score(X_test,y_test)
    print('Train accuracy:', pipe.score(X_train,y_train))
    print('Test accuracy:', test_accuracy)

    # Create predictions with test data
    preds = pipe.predict(X_test)

    # Evaluate predictions
    print('Test f1 score:', f1_score(y_test, preds, average='weighted'))
    
    # Generate confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test,preds)
    plt.title(model[0])
    
    print()
    return pipe, preds, test_accuracy


def get_Cs_scores(pipe,model_name):
    """ This function obtains the score for each C explored in the Logistic Regression CV
        args:
            pipe (a Sklearn pipeline): a pipeline that contains a fitted logistic regression CV model
        return:
            Cs_scores: a dictionary of Cs and the average score across all classes
    """
    Cs_scores = []
    pipe_scores = pipe.named_steps[model_name].scores_
    for i in range(len(pipe_scores[0][0])):
        scores_sum = 0
        for j in range(len(pipe_scores[0])):
            scores_sum += (pipe_scores[0][j][i])
        Cs_scores.append(scores_sum/(j+1))
    Cs_scores = dict(zip(pipe.named_steps[model_name].Cs_,Cs_scores))
    return Cs_scores


def get_coefs(pipe, model_name):
    df_coefs = pd.DataFrame(index=pipe.named_steps.ct.named_transformers_.ohe.get_feature_names_out())
    coefs = pipe.named_steps[model_name].coef_
    for i in range(len(coefs)):
        df_coefs[f'coefs_class_{i}'] = coefs[i]
    return df_coefs

