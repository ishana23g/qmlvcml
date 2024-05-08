import pandas as pd
import os

from typing import Tuple


def read_data(file_path: str, y_col_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Read the data and return the features and target variables.

    Args:
        - file_path (str): The path to the data file.
        - y_col_name (str): The name of the column that contains the target variable.

    Raises:
        - ValueError: If the target column is not found in the dataframe.

    Returns:
        tuple[pd.DataFrame, pd.Series]:
            - X (pandas.DataFrame): The features columns.
            - Y (pandas.Series): The target column.
    """
    df = pd.read_csv(file_path)
    if y_col_name not in df.columns:
        raise ValueError(f"Column {y_col_name} not found in the dataframe")
    y = df[y_col_name]
    X = df.drop(columns=[y_col_name])
    return X, y


def read_banana_data() -> tuple[pd.DataFrame, pd.Series]:
    """Read the banana quality data that is saved in this repository.

    Returns:
        tuple[pd.DataFrame, pd.Series]:
            - X (pandas.DataFrame): The 7 features columns of different banana properties, can also be thought of the observations.
            - y (pandas.Series): The target column that contains the quality of the banana.
    """
    # get the current directory path
    current_dir = os.path.dirname(__file__)
    # get the file path
    file_path = os.path.join(current_dir, "data", "banana_quality.csv")
    return read_data(file_path, "Quality")
