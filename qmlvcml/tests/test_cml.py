import pandas as pd
from unittest.mock import patch 
import pytest

from qmlvcml import *

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