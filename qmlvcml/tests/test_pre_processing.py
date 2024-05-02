
import numpy as np
import pandas as pd

# crete some random data for classification
def random_data(n=10, col=2):
    """
    Create a random dataset for classification. 

    Parameters
    ----------
    n: int
        Number of observations
    col: int
        Number of columns of different observations. 

    Returns
    -------
    df: pd.DataFrame
        A dataframe with n observations and col columns, and a target variable.
    """
    np.random.seed(0)
    data = np.random.rand(n, col)
    # create a binary target variable
    target = np.random.randint(0, 2, n)
    
    # create a dataframe
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['target'] = target

    return df

def split_X_y(df: pd.DataFrame):
    """
    Split the dataframe into X and y.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with n observations and col columns, and a target variable.

    Returns
    -------
    X: pd.DataFrame
        A dataframe with n observations and col columns.
    y: pd.Series
        A series with the target variable.
    """
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y


