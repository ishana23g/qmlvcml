import pandas as pd
# import numpy as np
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# do SVM classification
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


from .pre_processing import *

def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float = "scale",
) -> SVC:
    """Train the Support Vector Machine (SVM) model for binary classification.

    Args:
        - X_train (pd.DataFrame): The training observations which are all numeric.
        - y_train (pd.Series): The training target. We need to make sure that we have only two classes.
        - kernel (str, optional): The kernel for the SVM model. Defaults to "rbf" which is the Radial Basis Function.
        - C (float, optional): The regularization parameter. Defaults to 1.0.
        - gamma (float, optional): The kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Defaults to "scale" which is 1 / (n_features * X.var()).

    Raises:
        - ValueError: If the target variable has more than two classes.
        - ValueError: If the target variable is not a pandas Series or DataFrame.

    Returns:
        - SVC: The trained SVM model.
    """    
    if isinstance(y_train, (pd.Series, pd.DataFrame)):
        y_train = pd.DataFrame(y_train)
        if len(np.unique(y_train)) != 2:
            raise ValueError("We need exactly two classes for binary classification")
    else: 
        raise ValueError("y_train must be a pandas Series or DataFrame")
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: SVC, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series,
                   isPlot: bool = False) -> pd.DataFrame:
    """Evaluate the SVM model.

    Args:
        - model (SVC): The trained SVM model.
        - X_test (pd.DataFrame): The observations for testing that are all numeric.
        - y_test (pd.Series): The target variable for testing that are linked to the observations' indexes labels.
        - isPlot (bool, optional): Whether to plot the confusion matrix and the observations in 2D space (with the color representing the target variable). Defaults to False.

    Returns:
        - pd.DataFrame: The confusion matrix.
    """    
    y_pred = model.predict(X_test)
    confusion = pd.DataFrame(confusion_matrix(y_test, y_pred))
    if isPlot:
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        visualize_data(X_test, y_test, model)
    return confusion


def visualize_data(X: pd.DataFrame, 
                   y: pd.Series, 
                   type: str = "tsne",
                   figure_size=(10, 10)) -> None:
    """Visualize the data using a dimensionality reduction technique
    Possible techniques: 'trimap', 'pacmap', 'tsne', 'pca'.

    More stuff that could be done in the future is try and visualize the decision boundary of the SVM model and a gradient of the decision function.

    Args:
        - X (pd.DataFrame): The observational data without the target variable.
        - y (pd.Series): Just the target variable (for coloring the data).
        - type (str, optional): The type of visualization. Defaults to "tsne".
        - figure_size (tuple, optional): The size of the figure. Defaults to (10, 10).

    Raises:
        - ValueError: If the type of visualization is not supported.
    """    
    if type not in ["trimap", "pacmap", "tsne", "pca"]:
        raise ValueError(f"Unknown type: {type}")
    X = transform_X(X, type)

    plt.figure(figsize=figure_size)

    # plot data
    for color, label in zip(["b", "r"], [1, -1]):
        plot_x = (X[:, 0][y == label],)
        plot_y = (X[:, 1][y == label],)
        plt.scatter(
            plot_x,
            plot_y,
            c=color,
            marker="^",
            ec="k",
            label=f"class {label} validation",
        )
    plt.legend(loc="center left", bbox_to_anchor=(1.1, 0.05))
    plt.tight_layout()
    plt.show()


def apply_svm(df: pd.DataFrame, y_col: str, 
              test_size: float = 0.2, random_state: int = 42,
              isPlot: bool = False) -> tuple[SVC, pd.DataFrame, float]:
    """Apply SVM to the data.

    Possible future improvements are more hyperparameter tuning and k-fold or n-fold cross-validation. 

    Args:
        - df (pd.DataFrame): All of the data with both the features and target variable.
        - y_col (str): The name of the target variable column in the dataframe.
        - test_size (float, optional): The size of the testing set. Defaults to 0.2.
        - random_state (int, optional): The random state for splitting the data. Defaults to 42.
        - isPlot (bool, optional): Whether to plot the confusion matrix and the observations in 2D space (with the color representing the target variable). Defaults to False.

    Returns:
        tuple[SVC, pd.DataFrame, float]: 
        - SVC: The trained SVM model.
        - pd.DataFrame: The confusion matrix.
        - float: The accuracy of the model.
    """    


    """
    

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    y_col : str
        The column name of the target variable.
    test_size : float, optional
        The size of the testing set. Default is 0.2.
        Has to be between 0.0 and 1.0.
    random_state : int, optional
        The random state for splitting the data. Default is 42.
    isPlot : bool, optional
        Whether to plot the confusion matrix. Default is False.

    Returns
    -------
    model : sklearn.svm.SVC
        The trained SVM model.
    confusion : pandas.DataFrame
        The confusion matrix.
    """
    X_train, X_test, y_train, y_test = train_test_split_custom(
        df, y_col=y_col, test_size=test_size, random_state=random_state
    )
    # convert to pandas dataframes
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    model = train_svm(X_train, y_train)
    confusion = evaluate_model(model, X_test, y_test, isPlot=isPlot)
    acc = accuracy(confusion)
    print(f"Accuracy: {acc}")
    return model, confusion, acc