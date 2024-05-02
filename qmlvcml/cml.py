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
):
    """
    Train the Support Vector Machine (SVM) model for binary classification.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data.
    y_train : pandas.Series
        The training target. We need to make sure that we have only two classes.
    kernel : str, optional
        The kernel for the SVM model. Default is 'rbf' (Radial Basis Function).
    C : float, optional
        The regularization parameter. Default is 1.0.
    gamma : float, optional
        The kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Default is 'scale'.

    Returns
    -------
    model : sklearn.svm.SVC
        The trained SVM model.
    """
    try:
        y_train = pd.DataFrame(y_train)
        if len(np.unique(y_train)) != 2:
            raise ValueError("We need exactly two classes for binary classification.")
    except Exception as e:
        print(e)
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: SVC, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series,
                   isPlot: bool = False):
    """
    Evaluate the SVM model.

    Parameters
    ----------
    model : sklearn.svm.SVC
        The trained SVM model.
    X_test : pandas.DataFrame
        The testing data.
    y_test : pandas.Series
        The testing target.

    Returns
    -------
    confusion : pandas.DataFrame
        The confusion matrix.
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
                   model: SVC, 
                   type: str = "tsne",
                   figure_size=(10, 10)):
    """
    Visualize the data using a dimensionality reduction technique
    Possible techniques: 'trimap', 'pacmap', 'tsne', 'pca'.

    Parameters
    ----------
    X : pandas.DataFrame
        The observational data without the target variable.
    y : pandas.Series
        Just the target variable (for coloring the data).
    type : str, optional
        The type of visualization. Default is 'tsne'.
        Other options are 'trimap', 'pacmap', 'pca'.
    figure_size : tuple, optional
        The size of the figure. Default is (10, 10).
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
              isPlot: bool = False):
    """
    Apply SVM to the data.

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
    model = train_svm(X_train, y_train)
    confusion = evaluate_model(model, X_test, y_test, isPlot=isPlot)
    print(f"Accuracy: {accuracy(confusion)}")
    return model, confusion