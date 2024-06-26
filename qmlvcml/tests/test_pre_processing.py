import numpy as np
# from pennylane import numpy as np
import pandas as pd

from qmlvcml.pre_processing import *
from qmlvcml.vqml import state_preparation
from .test_helper_funcs import *

import pytest

# testing transform_X
def test_transform_X():

    # testing if the passed X is valid
    orig_col = 7
    X_df = random_data(col=orig_col, n=100)
    X_str = "This is our data"

    X, y = split_X_y(X_df)

    # make sure that the type is a pandas dataframe
    with pytest.raises(ValueError, match="X must be a pandas dataframe"):
        transform_X(X_str)

    # testing if the type is valid
    true_types = ['trimap', 'pacmap', 'tsne', 'pca', 'none', None]
    for type in true_types:
        trans_X = transform_X(X, type=type)
        n_cols = trans_X.shape[1]
        # make sure that there is the correct number of columns
        if type == 'none' or type is None:
            assert n_cols == orig_col, "We have a different dim. which was not was expected"
        else:
            assert n_cols == 2, "Did not apply dim. We wanted to get back only 2 columns (2D data)"
        # check the min and max of trans_X is between 0 and 1 for all columns
        scaling(trans_X)

    invalid_types = ['invalid', 'trimap1', 'pca1', 'tsne1', 'pacmap1']
    for type in invalid_types:
        with pytest.raises(ValueError, match="Type must be one of"):
            transform_X(X, type=type)


def test_train_test():
    # will be testing train_test_split_custom; scale_data; binary_classifier
    # test the parameters if test_size is between 0 and 1
    df = random_data(n=10, col=3)
    X, y = split_X_y(df)
    y_col_str = 'target'

    X_train, X_test, y_train, y_test = train_test_split_custom(df, y_col_str, 0.5)
    # check that all the column numbers and row numbers are right
    # since 0.5 is the only valid test size we can just hard code some of these stuff
    assert X_train.shape == (5, 3), "X_train has the wrong shape"
    assert X_test.shape == (5, 3), "X_test has the wrong shape"
    assert y_train.shape == (5,), "y_train has the wrong shape"
    assert y_test.shape == (5,), "y_test has the wrong shape"
    
    # for testing purposes we will convert the data to pandas dataframes
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    # check that X_train and X_test together give the range of 0 to 1
    X = pd.concat([X_train, X_test], axis=0)
    X = np.array(X)
    scaling(X)

    # check invalid test sizes
    invalid_test_sizes = [-0.1, 0, 1, 1.1]
    for test_size in invalid_test_sizes:
        with pytest.raises(ValueError, match="test_size has to be between 0.0 and 1.0, non-inclusive"):
            train_test_split_custom(df, y_col_str, test_size)
        
    # try using an invalid y_col
    invalid_y_col = 'invalid'
    with pytest.raises(KeyError, match=f"Column {invalid_y_col} not found in the dataframe"):
        train_test_split_custom(df, invalid_y_col)


def test_accuracy():   
    conf_100 = pd.DataFrame([[8, 0], [0, 2]])
    conf_0 = pd.DataFrame([[0, 2], [1, 0]])
    conf_50 = pd.DataFrame([[2, 2], [2, 2]])
    assert abs(accuracy(conf_100) -  1) < 1e-6,  "Did not compute the accuracy correctly"
    assert abs(accuracy(conf_0)   -  0) < 1e-6,  "Did not compute the accuracy correctly"
    assert abs(accuracy(conf_50)  - .5) < 1e-6, "Did not compute the accuracy correctly"


def test_y_transformations() :
    y_og = pd.Series([0, 0, 0, 1, 1])
    # expect y to be 0 and 1
    # check we do have that
    y_unique = np.unique(y_og)# .numpy()
    assert np.all(y_unique == [0, 1]), "Did not get the correct unique values"
    assert len(y_unique) == 2, "Did not get the correct number of unique values"

    y, mapping = binary_classifier(y_og)
    y = pd.Series(y)
    y_trans_unique = np.unique(y)# .numpy()
    assert np.all(y_trans_unique == [-1, 1]), "Did not get the correct unique values"
    assert np.all(y_og == back_transform(y, mapping)), "Did not get back the same values as we put in"

    # try having more than 2 unique values
    y_og = pd.Series([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="Y must be a binary class"):
        binary_classifier(y_og)
    with pytest.raises(ValueError, match="Y must be a binary class"):
        back_transform(y_og, mapping)

# CML Specific Pre-Processing Functions

def test_get_angles():
    x = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])# , requires_grad=False)
    ang = get_angles(x)
    assert np.allclose(ang, [0.563975, -0., 0., -0.975046, 0.975046]), "Did not get the correct angles"

def test_q_encoding():
    df = random_data(n=10, col=2)
    X, y = split_X_y(df)
    X = np.array(X)# , requires_grad=False)
    c = 0.1
    X_pad = padding_and_normalization(X, c)

    # testing padding is correct
    # make sure that all the rows sum up to 1
    for i in range(X_pad.shape[0]):
        row_squared = float(np.sum(X_pad[i, :] ** 2))
        assert np.allclose(row_squared, 1), "Did not normalize the observations correctly"
    # check if we have now 4 columns, and 10 rows
    assert X_pad.shape == (10, 4), "Did not pad the data correctly"

    # testing feature map
    X = feature_map(X_pad)
    # just testing we get the same amount of observations and the correct number of features
    # 2 original features and 3 new features in total. 5 features
    assert X.shape == (10, 5), "Did not get the correct shape"





