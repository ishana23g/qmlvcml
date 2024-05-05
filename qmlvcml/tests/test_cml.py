import pandas as pd

import pytest

from qmlvcml import *
from .test_helper_funcs import *


def test_svms(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    banana_df_X, banana_df_y = read_banana_data()
    banana_df = pd.concat([banana_df_X, banana_df_y], axis=1)
    # applying with plots
    model, confusion = apply_svm(banana_df, 'Quality', isPlot=True)

    # applying without plots
    model, confusion = apply_svm(banana_df, 'Quality', isPlot=False)
    
    # this is consistent so we can much more easily check for the accuracy and confusion matrix
    assert accuracy(confusion) == 0.98125, "Did not get the correct known accuracy of this model"

def test_train_svm_fails():
    # create random X and y data for testing
    df = random_data(n=10, col=3)
    X, y = split_X_y(df)
    y[0] = 2

    # test if the target variable is binary
    with pytest.raises(ValueError, match="We need exactly two classes for binary classification"):
        train_svm(X, y)
        
    with pytest.raises(ValueError, match="y_train must be a pandas Series or DataFrame"):
        train_svm(df, 'target')


def test_visualize_data_fails(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    df = random_data(n=10, col=3)
    X, y = split_X_y(df)

    with pytest.raises(ValueError, match="Unknown type: invalid"):
        visualize_data(X, y, SVC(), type='invalid')