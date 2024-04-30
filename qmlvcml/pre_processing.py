import pandas as pd
# import numpy as np
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from pandas.api.types import is_numeric_dtype

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import trimap
import pacmap


def transform_X(X: pd.DataFrame, type=None) -> np.array:
    """
    This function takes in a pandas dataframe or a regular numpy array and returns a pennylane numpy array.
    The function can also transform the data using a specified method to reduce the dimensions: 'trimap', 'pacmap', 'tsne', 'pca'.
    
    Parameters:
    -----------
    X: pd.DataFrame
        The input data
    type: str or None
        The type of transformation to apply to the data. 
        The options are: 'trimap', 'pacmap', 'tsne', 'pca', 'none', None

    Returns:
    --------
    np.array
        The transformed data
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
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
    return X


def train_test_split_custom(df, y_col, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    y_col : str, or pd.DataFrame/np.array
        The column name of the target variable. Default is 'target'.
    test_size : float, optional
        The size of the testing set. Default is 0.2.
        Has to be between 0.0 and 1.0.
    random_state : int, optional
        The random state for splitting the data. Default is 42.

    Returns
    -------
    train_X : pandas.DataFrame
        The training data.
    test_X : pandas.DataFrame
        The testing data.
    train_y : pandas.Series
        The training target.
    test_y : pandas.Series
        The testing target.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size has to be between 0.0 and 1.0, given test_size={test_size}")
    if isinstance(y_col, str):
        try:
            X = df.drop(columns=[y_col])
            X = scale_data(X)
            y = df[y_col]
        except KeyError:
            raise KeyError(f"Column {y_col} not found in the dataframe")
    else:
        X = df
        y = y_col
    y, _ = binary_classifier(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def accuracy(confusion: pd.DataFrame):
    """
    Calculate the accuracy of the model.

    Parameters
    ----------
    confusion : pandas.DataFrame
        The confusion matrix.

    Returns
    -------
    accuracy : float
        The accuracy of the model.
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
    

def binary_classifier (Y: np.array) -> tuple:
    """
    This function takes in the target variable and returns a binary class.

    Parameters:
    -----------
    Y: np.array
        The target variable
        
    Returns:
    --------
    tuple: (np.array, dict)
        The binary class and the mapping of the original classes

    Raises:
    -------
    ValueError
        If the target variable does not have exactly two classes

    """
    if len(np.unique(Y)) != 2:
        raise ValueError("Y must be a binary class")
    else:
        y_classes = np.unique(Y).tolist()
        mapping = {y_classes[0]: -1, y_classes[1]: 1}
        Y = np.array([mapping[y] for y in Y], requires_grad=False)
    return Y, mapping

def back_trainsform (Y: np.array, mapping: dict) -> np.array:
    """
    This function takes in the target variable and returns the original classes.

    Parameters:
    -----------
    Y: np.array
        The target variable
    mapping: dict
        The mapping of the original classes
        
    Returns:
    --------
    np.array
        The original classes

    Raises:
    -------
    ValueError
        If the target variable does not have exactly two classes

    """
    if len(np.unique(Y)) != 2:
        raise ValueError("Y must be a binary class")
    else:
        mapping = {v: k for k, v in mapping.items()}
        Y = np.array([mapping[y] for y in Y], requires_grad=False)
    return Y


# CML Specific Pre-Processing Functions


def scale_data(df: pd.DataFrame):
    """
    Scale the data.
    We are only working with numeric data, so we will use StandardScaler.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.

    Returns
    -------
    scaled_data : pandas.DataFrame
        The scaled data.
    """
    scaler = StandardScaler()
    try:
        scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure that the data contains only numeric columns.")
        scaled_data = df
    return scaled_data


# QML Specific Pre-Processing Functions

def get_angles(x: np.array) -> np.array:
    """
    This function takes in a numpy array and returns the angles for the state preparation circuit.

    Parameters:
    -----------
    x: np.array
        The input data

    Returns:
    --------
    np.array
        The angles for the state preparation circuit
    """
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def padding_and_normalization(X: np.array, c=0.1):
    """
    Pads out the input data with a constant value of c. The latent dimensions (padded values) ensure that the normalization does not erase any information on the length of the vectors, and keep the features distinguishable. Then we normalize the data by dividing by the norm of each row entry vectors.

    Parameters:
    -----------
    X: np.array
        The input data

    Returns:
    --------
    np.array
        The padded and normalized data
    """
    padding = np.ones((len(X), 2)) * c
    X_pad = np.c_[X, padding]
    normalization = np.sqrt(np.sum(X_pad**2, -1))
    return (X_pad.T / normalization).T

def feature_map(X: np.array) -> np.array:
    """
    This function takes in the input data and returns the new features.

    Parameters:
    -----------
    X: np.array
        The input data

    Returns:
    --------
    np.array
        The new features
    """
    return np.array([get_angles(x) for x in X])

