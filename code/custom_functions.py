# Functions used throughout the project
import pandas as pd
import numpy as np
import datetime
import warnings
import math


def snake_case(df):
    """ This function accepts a dataframe and modifies the column names by converting to snake case.
    
    args:
        df (Pandas dataframe): a dataframe
    """
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ','_').str.replace('u_','').str.replace('ws_','')


current_year = datetime.datetime.now().year
def month_to_datetime(month, year=current_year):
    """ This function accepts month name and returns the month in datetime form. Default year is the current year, but you can specify other years
    
    args:
        month (str): a month name
        optional (year): the year, default is the current year
    """
    return datetime.datetime.strptime(f'{month[0:3]} {year}', '%b %Y').strftime('%Y-%m')

def replace_months_from_df(df, year=current_year):
    """ This function accepts a dataframe where some columns have month names with no year, and renames the columns in place so that the months are in datetime format.
    
    args:
        df (Pandas dataframe): a dataframe where some columns have month names with no year
        optional (year): the year to add to the months, default is the current year
        
    """
    months_to_rename = {}
    for c in df:
        try:
            month_to_datetime(c)
            months_to_rename[c] = month_to_datetime(c)
        except:
            pass
    df.rename(months_to_rename,axis='columns',inplace=True)
    

def clean_values(df):
    """ This function cleans values in a series by lowercasing, stripping, and converting back to null (as applicable). This modifies the original dataframe.
    
    args:
        df (Pandas dataframe): a dataframe where columns of dtype "object" will be cleaned
    """
    for c in df:
        if df[c].dtype == 'O':
            df[c] = df[c].astype(str).apply(str.lower).str.strip().replace('Nan',np.nan)
            

def yearly_to_monthly(df):
    """ This function takes in a dataframe, and estimates the monthly sales for year columns (modifies the original dataframe).
    The estimate is by the rule of thumb by the business: "Q4 sales are usually 1/3 of the yearly sales".
    This function assumes that months in Q1, Q2, and Q3 sales are equally distributed amongst the rest of the 2/3 of sales, 
    and the months in Q4 are also equally distributed.
    Note: the function drops the yearly sales and replaces them with the monthly estimates
    
    args:
        df (Pandas dataframe): a dataframe that contains sales organized by year in columns
    """
    for c in df:
        try:
            Q4_monthly_sales = [(yearly_sales/3)/3 for yearly_sales in df[c]]
            other_monthly_sales = [((2*yearly_sales/3)/3)/3 for yearly_sales in df[c]]
            for i in range(1,10):
                df[datetime.datetime.strptime(f'{c} {i}','%Y %m').strftime('%Y-%m')] = other_monthly_sales
            for i in range(10,13):
                df[datetime.datetime.strptime(f'{c} {i}','%Y %m').strftime('%Y-%m')] = Q4_monthly_sales
            df.drop(columns=c,inplace=True)
        except:
            pass
        
def sort_columns(df):
    """ This function finds and sorts all the month/year columns in a dataframe. The columns must in in '%Y-%m' format.
    
    args:
        df (Pandas dataframe): a dataframe that contains columns that are datetime format
        
    return:
        df (Pandas dataframe): a dataframe that contains columns that are datetime format, now sorted
    """
    dates = []
    other = []
    for c in df:
        try:
            datetime.datetime.strptime(c,'%Y-%m').strftime('%Y-%m')
            dates.append(c)
        except:
            other.append(c)
    dates.sort()
    return df[other + dates]


def return_nondate_col(df):
    """ This function returns a dataframe without the columns that can be converted to a date.
    
    args:
         df (Pandas dataframe): a dataframe that contains columns that are datetime format
         
    return:
         df (Pandas dataframe): a dataframe that without the columns that are datetime format
    """
    nondate_cols = []
    for c in df:
        try:
            datetime.datetime.strptime(c,'%Y-%m').strftime('%Y-%m')
        except:
            nondate_cols.append(c)
    return df[nondate_cols]


def return_date_col(df):
    """ This function returns a dataframe with only the columns that can be converted to a date, and returns these columns sorted.
    
    args:
         df (Pandas dataframe): a dataframe that contains columns that are datetime format
         
    return:
         df (Pandas dataframe): a dataframe that with only the columns that are datetime format
    """
    date_cols = []
    for c in df:
        try:
            datetime.datetime.strptime(c,'%Y-%m').strftime('%Y-%m')
            date_cols.append(c)
        except:
            pass
    return sort_columns(df[date_cols])


def consolidate_columns(df, priority_col, secondary_col, new_col):
    """ This function consolidates two conflicting columns which have the same category of information.
    The function first uses the priority_col to fill values. If the priority_col is null, it uses the secondary_col.
    Finally, the function modifies the dataframe: drops both the priority_col and the secondary_col, and adds the new col.
    The function will raise a warning if new_col has any null values.
    
    args:
        df (Pandas dataframe): a dataframe that contains conflicting columns
        priority_col (str): the name of a column in df that serves as the first choice for filling values in new_col
        secondary_col (str): the name of a column in df that serves as the second choice for filling values in new_col
        new_col (str): the name of a new columns that consolidates priority_col & secondary_col
    """
    df[new_col] = [x if str(y) == 'nan' else y for x, y in zip(df[secondary_col],df[priority_col])]
    df.drop(columns=[priority_col,secondary_col],inplace=True)
    if df[new_col].isna().sum() != 0:
        warnings.warn(f'There are null values in {new_col}')
     
    
def simplify_collections(collections_old):
    """ This function simplifies collection names by:
        - removing case sizes (anything in this format: '22mm')
        - removing the
        - standardizing variations of ' - ' to '-'
        - replacing automatic with auto, chronograph with chrono, multifunction with multi
    
    args:
        collections_old (Pandas series): a series that contains a collection names
        
    return:
        collections (list): a list of collection names that have been standardized
    """
    print('Num of collections, before:', len(set(collections_old)))
    
    collections = []
    phrase_replacements = {'commuter':'comuter', 'mm':'', 'automatic':'auto', 'chronograph':'chrono', 'multifunction':'multi',
                        'rutherford':'ruterford','other':'misc','the':'','3hand':'3h','hand':'h'}
    name_replacements = {'classicminutewgli':'classicminuteglitz','3hdate':'3h', 'obf':'originalboyfriend','modernpersuit':'modernpursuit','deam':'dean',
                         'mensmisc':'misc-menswatch','XXX':np.nan,'fossilblue':'blue'}
    for name in collections_old.fillna('XXX'):
        name_split = name.split(' ')
        new_name = []
        for x in name_split:
            flag = True
            for item in phrase_replacements.keys():
                if item in x:
                    new_name.append(phrase_replacements[item])
                    flag = False
            if flag:
                new_name.append(x)
        new_name = ''.join(new_name)
        if new_name in name_replacements.keys():
            new_name = name_replacements[new_name]
        collections.append(new_name)

    print('Num of collections, after:', len(set(collections)))
    return collections

def fill_num_col_zero(df):
    """ This function fills null values in numeric columns in a dataframe with 0. 
    
    args:
        df (Pandas dataframe): a dataframe where columns that are not "object" will have their null values filled with 0
    """
    for c in df:
        if df[c].dtype != 'O':
            df[c] = df[c].fillna(0)
            
def consolidate_identical_columns(df, priority_col, secondary_col, new_col):
    """ This function consolidates two columns which should have identical information (for non-null columns) using consolidate_columns.
    The function will raise a warning if values are not identical
    
    args:
        df (Pandas dataframe): a dataframe that contains conflicting columns
        priority_col (str): the name of a column in df that serves as the first choice for filling values in new_col
        secondary_col (str): the name of a column in df that serves as the second choice for filling values in new_col
        new_col (str): the name of a new columns that consolidates priority_col & secondary_col
    """
    if df[df[priority_col] != df[secondary_col]][[priority_col, secondary_col]].dropna().sum().sum() != 0:
        warnings.warn(f'{priority_col} and {secondary_col} do not match')
    consolidate_columns(df, priority_col, secondary_col, new_col)

    
def simplify_color(color_old):
    """ This function simplifies color names by replacing inconsistencies
    
    args:
        color_old (Pandas series): a series that contains a color names
        
    return:
        color (list): a list of color names that have been standardized
    """
    print('Num of colors, before:', len(set(color_old)))
    
    color = color_old.replace({'2-tone':'two-tone','2t silver/rose':'rose gold','multi':'multicolor','gray':'grey',
                              'mop':'mother of pearl','yg / silver':'gold','rg / silver':'rose gold','yellow gold':'gold',
                               'silver stainless':'silver','smoke ip':'smoke','black ip':'black','brown ip':'brown',
                              })

    print('Num of colors, after:', len(set(color)))
    return color


def return_sales(df, col):
    """ This function returns sales from a dataframe that contains columns that can be 
    converted to datetime using return_date_col custom function. The returned function contains the 
    given as columns and monthly sales as index. This function also drops columns where all monthly sales are null.
    
    args:
        df (Pandas dataframe): a dataframe that contains columns that are datetime format
        col (str): the name of a column to be set as the new columns names
         
    return:
        sales (Pandas dataframe): a dataframe that where given col are the columns and monthly sales are the index.
    """
    sales = return_date_col(df.groupby(col).sum()).T
    sales.index = pd.to_datetime(sales.index)
    return sales.dropna(axis='columns')


def year_to_months(year):
    """ This function returns a list of the months of the year in '2022-01' format.
    
    args:
        year (str): the year of months to be returned
    
    return:
        months (list): list of strings (months in that year)
    """
    months = []
    for i in range(1,10):
        months.append(f'{year}-0{i}')
    for i in range(10,13):
        months.append(f'{year}-{i}')
    return months


def monthly_to_yearly(df, year):
    """ This function calculates the yearly sales from the monthly sales of that year in the dataframe. 
    
    args:
        df (Pandas dataframe): a dataframe that contains sales organized by year in columns
        year (str): the yearly sales to be returned
        
    return:
        yearly_sales (pandas series): a series containing the yearly sales of the specific year
    """
    months = year_to_months(year)
    yearly_sales = 0
    count = 0
    for col in months:
        try:
            yearly_sales += df[col]
            count += 1
        except:
            warnings.warn(f'Missing month of {col}')
    return yearly_sales

    
def preds_to_order(order_sheet, order_multiple, class_multiple = 1, ):
    """ This function generates orders from the forecasts from old models and the predictions for new models.
    The function detects whether it is Q4, and if so, increases the order by a multiplier given by the following rule of thumb:
    "Q4 sales are usually 1/3 of the yearly sales".
    
    args:
        order_sheet (Pandas dataframe): a dataframe that contains a new_model_preds_columns and proposal_preds_column (both variables)
        and a qty_total_inv column
        order_multiple (int): an integer that defines the quantity to which a proposal prediction is rounded up
        class_multiple (int), optional: the value to multiply the class value by to generate an order (only applicable for classification).
                                        Default is 1 to handle regression cases.
    """
    order_sheet['quarterly_forecast'] = 0
    order_sheet['order_quantity'] = 0
    proposal_preds_column = [x for x in order_sheet.columns if 'monthly_forecast' in x][0]
    new_model_preds_column = [x for x in order_sheet.columns if ('monthly_classifications' in x) | ('monthly_regressions' in x)][0]
    Q4_multiplier = 1
    if proposal_preds_column[-1] == str(4):
        Q4_multiplier = (1/3) / ((2/3) * (1/3)) # Q4 divided by another Q sales 
    
    for i in order_sheet.index:
        base = order_multiple
        if str(order_sheet.loc[i,proposal_preds_column]) == "nan":
            quarterly = class_multiple * order_sheet.loc[i,new_model_preds_column]
            order = quarterly * Q4_multiplier
        else:
            quarterly = 3 * order_sheet.loc[i,proposal_preds_column] * Q4_multiplier
            order = quarterly - order_sheet.loc[i,'qty_total_inv']
            if order < 0:
                order = 0
        order_sheet.loc[i,'quarterly_forecast'] = round(quarterly)
        order_sheet.loc[i,'order_quantity'] = base * math.ceil(order/base)