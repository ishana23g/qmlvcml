"""
This module contains the pre-processing functions that are used in the QML and CML pipelines. There are functions that are specific to QML and CML, and some that are general and can be used in both pipelines.
"""

from matplotlib import scale
import pandas as pd
# import numpy as np
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from pandas.api.types import is_numeric_dtype

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import trimap
import pacmap


from typing import Tuple, Union

# for transfrom_X the type is optional, and also takes in None or a string
def transform_X(X: pd.DataFrame, type: Union[str, None] = None) -> np.array:
    """
    This function takes in a pandas dataframe or a regular numpy array and returns a pennylane numpy array.
    The function can also transform the data using a specified method to reduce the dimensions: 'trimap', 'pacmap', 'tsne', 'pca'.

    Args:
        - X (pd.DataFrame): The input data (observations) in a pandas dataframe.
        - type (str | None, optional): The type of transformation to apply to the data. The possible options are: `'trimap', 'pacmap', 'tsne', 'pca', 'none', None.` Defaults to None.

    Raises:
        - ValueError: If the input data is not a pandas dataframe.
        - ValueError: If the type is not one of the specified options.

    Returns:
        - np.array: The transformed data which is also scaled. Note that the data is now a pennylane numpy array.
    """    
    if not isinstance(X, (pd.DataFrame)):
        raise ValueError("X must be a pandas dataframe")
    
    types = ['trimap', 'pacmap', 'tsne', 'pca', 'none', None]
    if type == 'trimap':
        X = trimap.TRIMAP().fit_transform(X.to_numpy())
    elif type == 'pacmap':
        X = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(X.to_numpy())
    elif type == 'tsne':
        X = TSNE(n_components=2, random_state=42).fit_transform(X.to_numpy())
    elif type == 'pca':
        X = PCA(n_components=2).fit_transform(X.to_numpy())
    elif type == 'none' or type is None:
        X = X.to_numpy()
    else:
        raise ValueError(f"Type must be one of {types}")
    
    X = np.array(X, requires_grad=False)
    # for each do a min-max scaling for each column
    X = scale_data(X)
    return X


def train_test_split_custom(df:pd.DataFrame, 
                            y_col: Union[str, pd.DataFrame, pd.Series],
                            test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[np.array, np.array, np.array, np.array]:
    """Split the data into training and testing sets.
    The X data (observations) are all min-maxed scaled and the target variable is converted to a binary class.
    The min-max scaling is done column wise to ensure that the one feature does not dominate the other.
    NOTE: The data will be stored using pennylane's numpy that uses tensors. 

    Args:
        - df (pd.DataFrame): The input data.
        - y_col (str | pd.DataFrame | pd.Series): The column name of the target variable.
        - test_size (float, optional): The size of the testing set. Defaults to 0.2.
        - random_state (int, optional): The random state (seed) for splitting the data. Defaults to 42. (there is no real reason for using 42; if it is not changed it will remain consistent)

    Raises:
        - ValueError: The test_size has to be between 0.0 and 1.0, non-inclusive.
        - KeyError: If the target column is not found in the dataframe.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: 
                - train_X (np.array): The training data.
                - test_X (np.array): The testing data.
                - train_y (np.array): The training target.
                - test_y (np.array): The testing target.
        """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size has to be between 0.0 and 1.0, non-inclusive")
    if isinstance(y_col, str):
        try:
            X = df.drop(columns=[y_col])    
            y = df[y_col]
        except KeyError:
            raise KeyError(f"Column {y_col} not found in the dataframe")
    else:
        X = df
        y = y_col

    X = np.array(X, requires_grad=False)
    X = scale_data(X)
    y, _ = binary_classifier(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def accuracy(confusion: pd.DataFrame) -> float:
    """
    Calculate the accuracy of the model from a confusion matrix.
    This function only works for binary classification/binary confusion matrices

    Args:
        - confusion (pd.DataFrame): The confusion matrix that is a 2x2 matrix.

    Returns:
        - float: The accuracy of the model that will be a value between 0 and 1 indicating the percentage of correct predictions.
    """
    return np.trace(confusion)/np.sum(np.sum(confusion))

# def combine_data(feats_train, Y_train, predictions_train, feats_val, Y_val, predictions_val, num_layers):
#     """
#     This function combines the features, target variable, and predictions into a single dataframe.
#     Features are also known as the input data or the observations.

#     Parameters:
#     -----------
#     feats_train: np.array
#         The training features
#     Y_train: np.array
#         The training target variable
#     predictions_train: np.array
#         The training predictions
#     feats_val: np.array
#         The validation features
#     Y_val: np.array
#         The validation target variable
#     predictions_val: np.array
#         The validation predictions
#     num_layers: int
#         The number of layers in the model (number of features)
    
#     Returns:
#     --------
#     pd.DataFrame
#         The combined data
#     """
#     train_dat = np.c_[feats_train,
#                       Y_train, predictions_train]
#     test_dat = np.c_[feats_val, 
#                         Y_val, predictions_val]
#     data = np.r_[train_dat, test_dat]
#     df = pd.DataFrame(data)
#     col_names = [f"Feature_{i}" for i in range(num_layers)] + ["Y", "Prediction"]
#     df.columns = col_names
#     return df
    

def binary_classifier (Y: np.array) -> Tuple[np.array, dict]:
    """This function takes in the target variable and returns a binary class.

    Args:
        - Y (np.array): The target variable that is already a pennylane numpy array.

    Raises:
        - ValueError: If we are not given a binary class. This is checked by ensuring that the target variable has two unique values found in the given array.

    Returns:
        Tuple[np.array, dict]:
            - Y (np.array): The converted array that contains now -1, 1 values.
            - mapping (dict): The mapping of the original classes. It will have structure like {original_class_0: -1, original_class_1: 1}
    """
    if len(np.unique(Y)) != 2:
        raise ValueError("Y must be a binary class")
    else:
        y_classes = np.unique(Y).tolist()
        mapping = {y_classes[0]: -1, y_classes[1]: 1}
        Y = np.array([mapping[y] for y in Y], requires_grad=False)
    return Y, mapping

def back_transform (Y: np.array, mapping: dict) -> np.array:
    """This function takes in the target variable and returns the original classes.
    Basically, it is the inverse of the binary_classifier function.

    Args:
        - Y (np.array): The target variable that is already a pennylane numpy array.
        - mapping (dict): The mapping of the original classes. This can either be constructed manually or can come from the binary_classifier function.

    Raises:
        - ValueError: If the target variable does not have exactly two classes.

    Returns:
        - np.array: The {-1, 1} classes are converted back to the original classes using the mapping.
    """
    if len(np.unique(Y)) != 2:
        raise ValueError("Y must be a binary class")
    else:
        mapping = {v: k for k, v in mapping.items()}
        Y = np.array([mapping[y] for y in Y], requires_grad=False)
    return Y


# CML Specific Pre-Processing Functions


def scale_data(X: np.array) -> np.array:
    """This function scales the data using min-max scaling.

    Args:
        - X (np.array): The observational data that is only has numerical values. NOTE: that the data is a pennylane numpy array. Also I do not check if the data is numerical or not. We are assuming the user has already checked this.

    Returns:
        - np.array: The [0, 1] scaled data.
    """
    for i in range(X.shape[1]):
        col_min = min(X[:, i])
        col_max = max(X[:, i])
        X[:, i] = (X[:, i] - col_min) / (col_max - col_min)
    return X

# QML Specific Pre-Processing Functions

def get_angles(x: np.array) -> np.array:
    """This function takes in a numpy array and returns the angles for the state preparation circuit.

    Args:
        - x (np.array): The input observational data. We are assuming that the data is already row wise normalized and padded.

    Returns:
        - np.array: The angles for which will be used in the state preparation circuit.
    """    
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def padding_and_normalization(X: np.array, c=0.1) -> np.array:
    """This function pads the data with a constant value and normalizes it.
    Pads out the input data with a constant value of c. 
    The latent dimensions (padded values) ensure that the normalization does not erase any information on the length of the vectors,
    and keep the features distinguishable.
    Then we normalize the data by dividing by the norm of each row entry vectors.

    Args:
        - X (np.array): The raw observational data that is not normalized. Before using this function, we might have the data use dimensionality reduction techniques. NOTE: This is not optimized/generalized for any multi/high dimensional data. It works for 2D data currently. 
        - c (float, optional): The constant value to pad the data with. Defaults to 0.1 as this is a small non-zero value.

    Returns:
        - np.array: The padded and normalized data.
    """    
    padding = np.ones((len(X), 2)) * c
    X_pad = np.c_[X, padding]
    normalization = np.sqrt(np.sum(X_pad**2, -1))
    return (X_pad.T / normalization).T

def feature_map(X: np.array) -> np.array:
    """This function takes in the input data and returns the new features.

    Args:
        - X (np.array): The padded and normalized data.

    Returns:
        - np.array: The new features.
    """    
    return np.array([get_angles(x) for x in X])

