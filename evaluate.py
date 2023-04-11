import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


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

    # Calculate R-squared and RMSE for model and baseline
    r2_model = r2_score(y, yhat)
    r2_baseline = 0.0  # since baseline prediction is always the mean value
    rmse_model = mean_squared_error(y, yhat, squared=False)
    rmse_baseline = mean_squared_error(y, np.full_like(y, np.mean(y)), squared=False)

    # Check if model SSE is less than baseline SSE
    if SSE_model < SSE_baseline:
        print("Model outperformed the baseline.")
    else:
        print("Model did not outperform the baseline.")
    
    # Print R-squared and RMSE for model and baseline
    print(f"R-squared (model): {r2_model:.4f}")
    print(f"R-squared (baseline): {r2_baseline:.4f}")
    print(f"RMSE (model): {rmse_model:.4f}")
    print(f"RMSE (baseline): {rmse_baseline:.4f}")    
    
    
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
    
    
def select_kbest(X, y, k):
    """
    Select the top k features from X based on their correlation with y using the f_regression method.
    """
    selector = SelectKBest(f_regression, k=k)  # Create a SelectKBest object with the f_regression method and k as input
    selector.fit(X, y)  # Fit the selector to the data
    mask = selector.get_support()  # Get a mask of the selected features
    selected_features = []  # Create an empty list to store the names of the selected features
    for bool, feature in zip(mask, X.columns):  # Loop through the mask and the columns of X
        if bool:  # If the feature is selected
            selected_features.append(feature)  # Add the name of the feature to the selected_features list
    return selected_features  # Return the list of selected features


def rfe(X, y, k):
    """
    Perform Recursive Feature Elimination to select the top k features from X based on their correlation with y.
    """
    estimator = LinearRegression()  # Create a LinearRegression object as the estimator
    selector = RFE(estimator, n_features_to_select=k)  # Create an RFE object with k as the number of features to select
    selector.fit(X, y)  # Fit the selector to the data
    mask = selector.support_  # Get a mask of the selected features
    selected_features = []  # Create an empty list to store the names of the selected features
    for bool, feature in zip(mask, X.columns):  # Loop through the mask and the columns of X
        if bool:  # If the feature is selected
            selected_features.append(feature)  # Add the name of the feature to the selected_features list
    return selected_features  # Return the list of selected features


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
    
    
    
def print_corr_table(df, target):
    """
    Takes a Pandas dataframe and a target variable name, and prints a table of 
    correlation coefficients and p-values ordered by highest to lowest correlation coefficient.
    """
    # Create correlation matrix and extract correlation coefficients for target variable
    corr = df.corr()
    corr_target = corr[target]

    # Calculate P-values for all correlations
    pvals = df.apply(lambda x: df.apply(lambda y: stats.pearsonr(x, y)[1]))

    # Combine correlation coefficients and P-values into a single dataframe
    corr_table = pd.concat([corr_target, pvals[target]], axis=1)
    corr_table.columns = ["corr_coef", "p_value"]
    
    # Sort table by absolute correlation coefficient (in descending order)
    corr_table["abs_corr_coef"] = corr_table["corr_coef"].abs()
    corr_table = corr_table.sort_values("abs_corr_coef", ascending=False)
    corr_table = corr_table.drop(columns=["abs_corr_coef"])

    # Print table
    print("Correlation Coefficients and P-Values:")
    print(corr_table)
    
    
def show_distribution(dataframe, column_name):
    """
    Display a histogram showing the distribution of values in a dataframe column.
    
    Args:
    dataframe: pandas DataFrame containing the data
    column_name: string representing the name of the column whose distribution is to be displayed
    
    Returns:
    None
    """
    plt.hist(dataframe[column_name])
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title('Distribution of ' + column_name)
    plt.show()