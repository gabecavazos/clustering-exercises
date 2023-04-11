'''Wrangles data from Zillow Database'''

##################################################Wrangle.py###################################################

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

from env import user, password, host
import warnings
warnings.filterwarnings('ignore')

#**************************************************Acquire*******************************************************

def acquire_zillow():
    if os.path.exists('zillow_2017.csv'):
        print('local version found!')
        return pd.read_csv('zillow_2017.csv', index_col=0)
    else:
        ''' Acquire data from Zillow using env imports and rename columns'''

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

        query = """

        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        FROM properties_2017

        LEFT JOIN propertylandusetype USING(propertylandusetypeid)

        WHERE propertylandusedesc IN ("Single Family Residential",                       
                                      "Inferred Single Family Residential")"""

        # get dataframe of data
        df = pd.read_sql(query, url)


        # renaming column names to one's I like better
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                                  'bathroomcnt':'bathrooms', 
                                  'calculatedfinishedsquarefeet':'area',
                                  'taxvaluedollarcnt':'tax_value', 
                                  'yearbuilt':'year_built',})
        return df

#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


#**************************************************check it out*******************************************************

def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)


def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)



def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
        
#**************************************************Prepare*******************************************************

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value'])
    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')
    # fit the imputer once on train
    imputer.fit(train[['year_built']])
    #impute (transform) the values learned from train on train, val, test
    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])   
    return train, validate, test    


def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df



def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))



def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df





#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test

#**************************************** probably too much and redundant because I put most of this in evaluate*********************************************************


def convert_to_object(df, columns):
    """Converts the data types of a list of DataFrame columns to object.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): A list of column names to convert.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    for col in columns:
        df[col] = df[col].astype('object')
    return df



def run_lazy_classifier(data, target_column):
    """
    Runs LazyClassifier on a given DataFrame.

    Args:
    data (pandas.DataFrame): The input DataFrame containing the features and target.
    target_column (str): The name of the target column in the DataFrame.

    Returns:
    None
    """
    # Split the data into training and test sets
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a LazyClassifier object and fit it to the data
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Print the model performance summary
    print(models)

    
def run_lazy_regressor(data, target_column):
    """
    Runs LazyRegressor on a given DataFrame.

    Args:
    data (pandas.DataFrame): The input DataFrame containing the features and target.
    target_column (str): The name of the target column in the DataFrame.

    Returns:
    None
    """
    # Split the data into training and test sets
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a LazyRegressor object and fit it to the data
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Print the model performance summary
    print(models)
    
    
def visualize_corr(df, sig_level=0.05, figsize=(10,8)):
    """
    Takes a Pandas dataframe and a significance level, and creates a heatmap of 
    statistically significant correlations between the variables.
    """
    # Create correlation matrix
    corr = df.corr()

    # Mask upper triangle of matrix (redundant information)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Get statistically significant correlations (p-value < sig_level)
    pvals = df.apply(lambda x: df.apply(lambda y: stats.pearsonr(x, y)[1]))
    sig = (pvals < sig_level).values
    corr_sig = corr.mask(~sig)

    # Set up plot
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.set_style("white")

    # Create heatmap with statistically significant correlations
    sns.heatmap(corr_sig, cmap='Purples', annot=True, fmt=".2f", mask=mask, vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(f"Statistically Significant Correlations (p<{sig_level})")
    plt.show()
    
    
def regression_errors(y, yhat):
    """
    Calculates regression errors and returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    """
    # Calculate SSE, ESS, and TSS
    SSE = np.sum((y - yhat) ** 2)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = SSE + ESS

    # Calculate MSE and RMSE
    n = len(y)
    MSE = SSE / n
    RMSE = np.sqrt(MSE)

    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y):
    """
    Calculates the errors for the baseline model and returns the following values:
    sum of squared errors (SSE)
    mean squared error (MSE)
    root mean squared error (RMSE)
    """
    # Calculate baseline prediction
    y_mean = np.mean(y)
    yhat_baseline = np.full_like(y, y_mean)

    # Calculate SSE, MSE, and RMSE
    SSE_bl = np.sum((y - yhat_baseline) ** 2)
    n = len(y)
    MSE_bl = SSE / n
    RMSE_bl = np.sqrt(MSE)

    return SSE_bl, MSE_bl, RMSE_bl


def better_than_baseline(y, yhat):
    """
    Checks if your model performs better than the baseline and returns a boolean value.
    """
    # Calculate errors for model and baseline
    SSE_model = np.sum((y - yhat) ** 2)
    SSE_baseline = np.sum((y - np.mean(y)) ** 2)

    # Check if model SSE is less than baseline SSE
    if SSE_model < SSE_baseline:
        return True
    else:
        return False


def plot_residuals(y, yhat):
    """
    Creates a residual plot using matplotlib.
    """
    # Calculate residuals
    residuals = y - yhat

    # Create residual plot
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    
    
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'tax_value'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values, 
                                                  index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
    
    