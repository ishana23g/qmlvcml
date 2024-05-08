Getting Started
===============

This page details how to get started with QMLvCML. 

To install this package, you first need to clone this repository.
And once you are inside the repository, you can install the package using the following command:

.. code-block:: bash
        pip install -e .

A simple example of how to use the package is shown below.

First we want to load some data to be classified. This can be done using the following command:
.. code-block:: python
        import qmlvcml 
        banana_df_X, banana_df_y = qmlvcml.load_banana_data()

Then we can apply the Variational Classifier using the following command:
.. code-block:: python
        qmlvcml.apply_model(banana_df_X, banana_df_y, steps=50,
                     batch_size_percent=.8, isPlot=True, isDebug=False,
                     dim_reduce_type='trimap')

For the the classical algorithm (SVMs) we can use the following command:
.. code-block:: python
        import pandas as pd
        banana_df = pd.concat([banana_df_X, banana_df_y], axis=1)
        model, confusion = qmlvcml.apply_svm(banana_df, 'Quality', isPlot=True)




