QMLvCML
==============================
[//]: # (Badges)
[![Documentation Status](https://readthedocs.org/projects/qmlvcml/badge/?version=latest)](https://qmlvcml.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Build Status](https://github.com/ishana23g/qmlvcml/workflows/CI/badge.svg)](https://github.com/ishana23g/qmlvcml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/ishana23g/QMLvCML/branch/main/graph/badge.svg?token=QGz0MSi2xw)](https://codecov.io/gh/ishana23g/QMLvCML)

This package is to use a dataset and compare a Quantum based Classifier that is a Machine Learning Algorithm vs a Classical Machine Learning Algorithm.

The dataset by default that this package has access to is from [Kaggle Banana Dataset](https://www.kaggle.com/datasets/l3llff/banana)

Installation
============

To install this package, you first need to clone this repository.
And once you are inside the repository, you can install the package using the following command:

```
        pip install -e .
```


Stuff you can do:
===============

A simple example of how to use the package is shown below.

First we want to load some data to be classified. This can be done using the following command:

```
        import qmlvcml 
        banana_df_X, banana_df_y = qmlvcml.load_banana_data()
```

Then we can apply the Variational Classifier using the following command:

```
        qmlvcml.apply_model(banana_df_X, banana_df_y, steps=50,
                     batch_size_percent=.8, isPlot=True, isDebug=False,
                     dim_reduce_type='trimap')
```

For the the classical algorithm (SVMs) we can use the following command:

```
        import pandas as pd
        banana_df = pd.concat([banana_df_X, banana_df_y], axis=1)
        model, confusion = qmlvcml.apply_svm(banana_df, 'Quality', isPlot=True)
```

### Copyright

Copyright (c) 2024, Ishana Garuda


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
