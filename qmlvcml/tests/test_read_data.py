# raise ValueError(f"Column {y_col_name} not found in the dataframe")

import pandas as pd
from qmlvcml import *
import os 
import pytest

def test_read_data():
    # read in the banana data 
    # but have the wrong col name 
    current_dir = os.path.dirname(__file__)
    # go out one directory
    current_dir = os.path.dirname(current_dir)
    
    # get the file path
    file_path = os.path.join(current_dir, 'data', 'banana_quality.csv')

    with pytest.raises(ValueError, match="Column quality not found in the dataframe"):
        read_data(file_path, 'quality')
    