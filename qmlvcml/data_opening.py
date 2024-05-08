import pandas as pd
import os


def read_data(file_path: str, y_col_name: str) -> tuple:
    """
    Read the data and return the features and target variables.

    :param file_path: the path to the data file.
    :type file_path: str
    :param y_col_name: the name of the target column.
    :type y_col_name: str
    :return: the features and target variables.
    :rtype: tuple of pandas.DataFrame
    """    
    df = pd.read_csv(file_path)
    if y_col_name not in df.columns:
        raise ValueError(f"Column {y_col_name} not found in the dataframe")
    y = df[y_col_name]
    X = df.drop(columns=[y_col_name])
    return X, y

def read_banana_data() -> tuple:
    """
    Read the banana quality data that is saved in this repository.
    
    Returns
    -------
    tuple: X, y
        The features: 7 columns of different banana properties/observations.
        The target: The quality of the banana.

    :return: The features of 7 different banana properties of the banana -> X; The quality of the banana -> y
    :rtype: tuple of pandas.DataFrame
    """
    # get the current directory path
    current_dir = os.path.dirname(__file__)
    # get the file path
    file_path = os.path.join(current_dir, 'data', 'banana_quality.csv')
    return read_data(file_path, 'Quality')