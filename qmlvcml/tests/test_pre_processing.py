
import numpy as np
import pandas as pd

# crete some random data for classification


np.random.seed(0)
data = np.random.rand(100, 5)
target = np.random.randint(0, 2, 100)

# create a dataframe
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
df['target'] = target

# create a Data object
data = Data(df)
print(data)

data.split_train_test()
print(data)
