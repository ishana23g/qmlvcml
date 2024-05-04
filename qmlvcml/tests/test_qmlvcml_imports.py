"""
Unit and regression test for the qmlvcml package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest

from qmlvcml import *


def test_qmlvcml_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qmlvcml" in sys.modules
    
def test_vqml_imported():
    """
    Test if all the functions from vqml module are imported directly into the QMLvCML package
    """
    assert "state_preparation" in dir(sys.modules["qmlvcml"])
    assert "layer" in dir(sys.modules["qmlvcml"])
    assert "circuit" in dir(sys.modules["qmlvcml"])
    assert "variational_classifier" in dir(sys.modules["qmlvcml"])
    assert "square_loss" in dir(sys.modules["qmlvcml"])
    assert "cost" in dir(sys.modules["qmlvcml"])
    assert "apply_model" in dir(sys.modules["qmlvcml"])

def test_cml_imported():
    """
    Test if all the functions from cml module are imported directly into the QMLvCML package
    """
    assert "train_svm" in dir(sys.modules["qmlvcml"])
    assert "evaluate_model" in dir(sys.modules["qmlvcml"])
    assert "visualize_data" in dir(sys.modules["qmlvcml"])
    assert "apply_svm" in dir(sys.modules["qmlvcml"])

def test_pre_processing_imported():
    """
    Test if all the functions from pre_processing module are imported directly into the QMLvCML package
    """
    assert "transform_X" in dir(sys.modules["qmlvcml"])
    assert "train_test_split_custom" in dir(sys.modules["qmlvcml"])
    assert "accuracy" in dir(sys.modules["qmlvcml"])
    assert "binary_classifier" in dir(sys.modules["qmlvcml"])
    assert "back_transform" in dir(sys.modules["qmlvcml"])
    assert "scale_data" in dir(sys.modules["qmlvcml"])
    assert "get_angles" in dir(sys.modules["qmlvcml"])
    assert "padding_and_normalization" in dir(sys.modules["qmlvcml"])
    assert "feature_map" in dir(sys.modules["qmlvcml"])

def test_data_opening_imported():
    """
    Test if all the functions from data_opening module are imported directly into the QMLvCML package
    """
    assert "read_data" in dir(sys.modules["qmlvcml"])
    assert "read_banana_data" in dir(sys.modules["qmlvcml"])
