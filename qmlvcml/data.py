import pandas as pd

def read_data(file_path: str, y_col_name: str) -> tuple:
    """
    Read the data and return the features and target variables.

    Parameters
    ----------
    file_path : str
        The path to the data file.

    Returns
    -------
    tuple
        The features and target variables.
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
    """
    return read_data('data/banana_quality.csv', 'Quality')