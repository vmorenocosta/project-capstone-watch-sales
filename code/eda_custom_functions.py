import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from custom_functions import *

def eda_by_column(df_models, group_col):
    """ This function generates various plots by grouping a dataframe by the "group_col" and plotting against a variable.
    
    args:
        df_models (Pandas Dataframe): a dataframe that contains the columns names for group_col and variable
        group_col (str): the name of a categorical column to group df_models
    """
    data = df_models.groupby(by=group_col).mean()['norm_yearly_sales']
    plt1 = px.histogram(data, title = f'Histogram of average normalized yearly sales grouped by {group_col}', 
                        color = data.keys())
    plt1.show()

    data1 = df_models.groupby(by=group_col).mean()['norm_yearly_sales'].sort_values(ascending=False).head(10)
    plt2 = px.bar(data1, title = f'Top {len(data1)} average normalized yearly sales grouped by {group_col}', color = data1.keys())
    plt2.show()
    
    data2 = df_models.groupby(by=group_col).count()['norm_yearly_sales'].sort_values(ascending=False).head(10)
    plt3 = px.bar(data2, title = f'Top {len(data2)} most common {group_col}', color = data2.keys())
    plt3.show()

    sales = return_sales(df_models,group_col)
    top_collections_2019 = dict(sales.loc['2019',:].sum().sort_values(ascending=False).head(20))
    top_collections_2020 = dict(sales.loc['2020',:].sum().sort_values(ascending=False).head(20))
    top_collections_2021 = dict(sales.loc['2021',:].sum().sort_values(ascending=False).head(20))

    fig = go.Figure(data=[
        go.Bar(name='2019', x=list(top_collections_2019.keys()), y=list(top_collections_2019.values())),
        go.Bar(name='2020', x=list(top_collections_2020.keys()), y=list(top_collections_2020.values())),
        go.Bar(name='2021', x=list(top_collections_2021.keys()), y=list(top_collections_2021.values()))
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', title=f'Top {len(top_collections_2019)} {group_col} in each year')
    fig.show()
    