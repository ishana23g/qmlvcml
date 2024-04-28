# Training & Validation
# Training phase varies according to quantum classifier type. For example, in the case of quantum-inspired classifier,
# we train model on classical machine. In the case of quantum kernel classifier, we estimate kernel on a quantum
# device) for instance, IBM Q computer or quantum simulator). 

# Validation
# We evaluate and update parameters to decrease cost function and improve performance of model



def train_test_split(df):
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.

    Returns
    -------
    train_df : pandas.DataFrame
        The training data.
    test_df : pandas.DataFrame
        The testing data.
    """
    pass

models = ['SVM', 'QNN', 'QKC']

def train_model(train_df, method='q'):
    """
    Train the model.

    Parameters
    ----------
    train_df : pandas.DataFrame
        The training data.
    method : str, optional
        The method to use for training. Default is 'q' or 'qiskit'.

    Returns
    -------
    model : object
        The trained model.
    """
    pass