
import numpy as np
import pandas as pd


#### ⌄⌄⌄⌄⌄⌄⌄ HELPER FUNCTIONS  ⌄⌄⌄⌄⌄⌄⌄  ####

# crete some random data for classification
def random_data(n=10, col=2):
    np.random.seed(0)
    data = np.random.rand(n, col)
    # create a binary target variable
    target = np.random.randint(0, 2, n)
    
    # create a dataframe
    df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(col)])
    df['target'] = target

    return df

def split_X_y(df: pd.DataFrame):
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y

def scaling(X: np.array):
    for i in range(X.shape[1]):
        # print(f"Column {i}")
        # print(X[:, i])
        # print(f"Min: {min(X[:, i])}")
        # print(f"Max: {max(X[:, i])}")
        assert abs(min(X[:, i]) - 0) < 1e-6, "Did not scale the column correctly to get min = 0"
        assert abs(max(X[:, i]) - 1) < 1e-6, "Did not scale the column correctly to get max = 1"
 
#### ^^^^^^^ HELPER FUNCTIONS  ^^^^^^^  ####


