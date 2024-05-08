"""
This module contains the quantum variational classifier and the quantum variational circuit machine learning model.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import trimap
import pacmap

import seaborn as sns   
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd

from .pre_processing import *

from typing import Tuple

# NOTE the numpy array used in this quantum process is from pennylane, not default from numpy. 

def state_preparation(a: np.array) -> None:
    """This function prepares the quantum state according to the angles provided. It will be adding the gates to the qubit wires. 

    Args:
        - a (np.array): The angles for the state preparation circuit in a pennylane numpy array.
    """
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)


def layer(layer_weights: np.array) -> None:
    """This function applies a layer of operations to all qubits in the circuit. In this case, it applies a rotation gate to each qubit and a CNOT gate between the qubits.
    NOTE: This currently only works for 2 qubits. To generalise, we need to add a loop for the number of qubits. More information can be found here: https://arxiv.org/pdf/quant-ph/0407010
    
    Args:
        - layer_weights (np.array): The weights for the layer
    """    
    for wire in range(2):
        qml.Rot(*layer_weights[wire], wires=wire)
    qml.CNOT(wires=[0, 1])

# create a quantum device
dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit(weights: np.array, x: np.array = np.array([0, 0, 0, 0])) -> qml.expval:
    """Applies a quantum circuit to the given input.

    Args:
        - weights (np.array): The weights for the circuit
        - x (np.array, optional): The input data which are pre-processed to get the angles for the state preparation circuit. Defaults to np.array([0, 0, 0, 0]).

    Returns:
        - qml.expval: The expectation value of the Pauli-Z operator on the first qubit.
    """    
    state_preparation(x)
    for layer_weights in weights:
        layer(layer_weights)
    _circuit = qml.expval(qml.PauliZ(0))

    return _circuit


def variational_classifier(weights: np.array, bias: np.array, x: np.array, isPlot=False) -> float:
    """This function applies the quantum circuit to the given input data.

    Args:
        - weights (np.array): The weights for the circuit (The quantum part of the model)
        - bias (np.array): The bias for the circuit (The classical part of the model)
        - x (np.array): The observational data to apply the circuit to; this will be the angles we got from the the state preparation circuit.
        - isPlot (bool, optional): Whether to plot the circuit or not. Defaults to False. (NOTE: Might not be working as expected, need to check this.)

    Returns:
        - float: The output of the circuit
    """    
    if isPlot:
        qml.draw(circuit)(weights, x)
    return circuit(weights, x) + bias


def square_loss(labels: np.array, predictions: np.array) -> float:
    """This function calculates the square loss between the labels and the predictions.    

    Args:
        - labels (np.array): The true labels
        - predictions (np.array): The predicted labels

    Returns:
        - float: The square loss between the labels and the predictions
    """
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def cost(weights: np.array, bias: np.array, X: np.array, Y: np.array) -> float:
    """This function calculates the cost function for the quantum circuit.

    Args:
        - weights (np.array): The weights for the circuit (The quantum part of the model)
        - bias (np.array): The bias for the circuit (The classical part of the model)
        - X (np.array): The observational data to apply the circuit to. This will be the angles we got from the the state preparation circuit.
        - Y (np.array): The true labels

    Returns:
        - float: The cost function for the quantum circuit
    """    
    # Transpose the batch of input data in order to make the indexing
    # in state_preparation work
    predictions = variational_classifier(weights, bias, X.T)
    return square_loss(Y, predictions)



def apply_model(
    X: Union[np.array, pd.DataFrame],
    Y: Union[np.array, pd.DataFrame],
    weights_init: np.array =None,
    bias_init: np.array =None,
    steps: int =100,
    batch_size_percent: float =0.5,
    split_per: float = 0.4,
    isPlot: bool =False,
    isDebug: bool =False,
    dim_reduce_type: Union[str, None] =None,
    smoothness: int = 100,
    seed: int =42
) -> Tuple[np.array, np.array, np.array, np.array]:
    """This function applies the quantum model to the given data. The model is trained using the given data and tries to optimize the weights and bias to get the best possible model.

    Args:
        - X (np.array | pd.Dataframe): The observational data without the target variable.
        - Y (np.array | pd.Dataframe): The target variable (for classification).
        - weights_init (np.array, optional): The initial weights for the circuit. Defaults to None, it will be randomly generated.
        - bias_init (np.array, optional): The initial bias for the circuit. Defaults to None, it will be randomly generated.
        - steps (int, optional): The number of steps to train the model. Defaults to 100.
        - batch_size_percent (float, optional): The percentage of the training data to use in each batch for the QML model steps. Defaults to 0.5.
        - split_per (float, optional): The percentage of the data to use for the training set (the training-testing split percentage). Defaults to 0.4.
        - isPlot (bool, optional): Whether to plot the dimensionality reduction data, the normalized+padded data, the feature vectors, the circuit, the confusion matrix, and the decision boundary of the final model with the predicted labels. Defaults to False.
        - isDebug (bool, optional): Whether to print out debug information or not. Defaults to False. Can be useful not for debugging but for understanding the process too. 
        - dim_reduce_type (str | None, optional) : The type of dimension reduction to apply to the data. The options are: 'trimap', 'pacmap', 'tsne', 'pca', 'none', None. Defaults to None.
        - smoothness (int, optional): The smoothness of the final grid plot which will be used in the decision boundary plot. Defaults to 100.
        - seed (int, optional): The seed for the random number generator. Defaults to 42.

    Raises:
        - ValueError: If the batch_size_percent or split_per are not between 0 and 1.
        - ValueError: If the dim_reduce_type is not one of the supported types.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]:
            - np.array: The final weights for the circuit
            - np.array: The final bias for the circuit
            - np.array: The costs for each step of the training
            - np.array: The accuracies for each step of the training        
    """    

    # make sure that batch_size_percent and split_per are between 0 and 1
    if not 0 < batch_size_percent < 1:
        raise ValueError("batch_size_percent must be between 0 and 1")
    if not 0 < split_per < 1:
        raise ValueError("split_per must be between 0 and 1")

    figure_size = (10, 10)
    X = transform_X(X, type=dim_reduce_type)
    X_norm = padding_and_normalization(X, c=0.1)
    # the angles for state preparation are the features
    features = feature_map(X_norm)
    
    if isDebug:
        # print out the dimentions, and also the first top 5 samples /head
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"X_norm shape: {X_norm.shape}")
        print(f"Features shape: {features.shape}")

        print(f"X_norm[0]: {X_norm[0]}")
        print(f"Features[0]: {features[0]}")

    # create a train-test split (using X -> features now)
    feats_train, feats_val, Y_train, Y_val = train_test_split_custom(
        features, y_col=Y, test_size=split_per, random_state=seed
    )

    # get the indecies of the train and test data, to split the X dataset itself for plotting later
    train_index, test_index = train_test_split(
        np.arange(len(X)), test_size=split_per, random_state=seed
    )

    num_train = len(Y_train)

    Y, _ = binary_classifier(Y)

    if isPlot:
        print("Plotting the data")
        plt.figure(figsize=figure_size)
        plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="b", marker="o", ec="k")
        plt.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c="r", marker="o", ec="k")
        plt.title("Original data")
        plt.tight_layout()
        plt.show()

        print("Plotting the padded and normalised data")
        plt.figure(figsize=figure_size)
        dim1 = 0
        dim2 = 1
        plt.scatter(
            X_norm[:, dim1][Y == 1], X_norm[:, dim2][Y == 1], c="b", marker="o", ec="k"
        )
        plt.scatter(
            X_norm[:, dim1][Y == -1],
            X_norm[:, dim2][Y == -1],
            c="r",
            marker="o",
            ec="k",
        )
        plt.title(f"Padded and normalised data (dims {dim1} and {dim2})")
        plt.tight_layout()
        plt.show()

        print("Plotting the feature vectors")
        plt.figure(figsize=figure_size)
        dim1 = 0
        dim2 = 3
        plt.scatter(
            features[:, dim1][Y == 1],
            features[:, dim2][Y == 1],
            c="b",
            marker="o",
            ec="k",
        )
        plt.scatter(
            features[:, dim1][Y == -1],
            features[:, dim2][Y == -1],
            c="r",
            marker="o",
            ec="k",
        )
        plt.title(f"Feature vectors (dims {dim1} and {dim2})")
        plt.tight_layout()
        plt.show()

    opt = NesterovMomentumOptimizer(0.5)

    num_qubits = len(np.unique(Y))  # number of classes
    num_layers = len(features[0])  # number of features

    if weights_init is None:
        weights = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    else:
        weights = weights_init

    if bias_init is None:
        bias = np.array(0.0, requires_grad=True)
    else:
        bias = bias_init

    costs = []
    accs = []
    weightses = [weights]
    biases = [bias]

    preds = []
    
    batch_size = int(batch_size_percent * num_train)
    for it in range(steps):
        # Update the weights by one optimizer step, using only a limited batch of data
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
            
        # try and find the optimal weights and bias
        weights, bias, _, _ = opt.step(
            cost, weights, bias, feats_train_batch, Y_train_batch
        )

        # Append the weights and bias to the list
        weightses.append(weights)
        biases.append(bias)

        # Compute predictions on train and validation set
        predictions_train = np.sign(variational_classifier(weights, bias, feats_train.T))
        predictions_val = np.sign(variational_classifier(weights, bias, feats_val.T))

        # Compute accuracy on train and validation set
        conf_train = confusion_matrix(Y_train, predictions_train)
        conf_val = confusion_matrix(Y_val, predictions_val)
        acc_train = accuracy(conf_train)
        acc_val = accuracy(conf_val)

        # Compute cost on all samples
        _cost = cost(weights, bias, features, Y)
        costs.append(_cost)
        accs.append(acc_val)
        
        if isDebug:
            print(
                f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
                f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
            )

    # check to see if min cost and max accuracy are the same 
    if np.argmin(costs) != np.argmax(accs):
        print("Warning: Minimum cost and maximum accuracy are not at the same iteration")

    best_acc_index = np.argmax(accs)
    weights = weightses[best_acc_index]
    bias = biases[best_acc_index]
    # final model
    predictions_val = np.sign(variational_classifier(weights, bias, feats_val.T))

    print(f"Best accuracy: {accs[best_acc_index]}")
    print(f"Final weights: {weights}")
    print(f"Final bias: {bias}")
    print(f"Final predictions: {predictions_val}")
    
    if isPlot:
        # do a subplot of the costs and accuracies
        plt.figure(figsize=figure_size)
        plt.subplot(2, 1, 1)
        plt.plot(costs, "r")
        # add a point to show the minimum cost
        plt.plot(np.argmin(costs), np.min(costs), "ro")
        plt.title("Cost")
        plt.subplot(2, 1, 2)
        plt.plot(accs, "b")
        # add a point to show the maximum accuracy
        plt.plot(np.argmax(accs), np.max(accs), "bo")
        plt.title("Accuracy")
        plt.tight_layout()
        plt.show()

    # df = combine_data(feats_train, Y_train, predictions_train, feats_val, Y_val, predictions_val, num_layers)

    if isPlot:
        plt.figure(figsize=figure_size)
        cm = plt.cm.RdBu

        # make data for decision regions
        xx, yy = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), smoothness), 
                             np.linspace(min(X[:, 1]), max(X[:, 1]), smoothness) )
        X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

        # preprocess grid points like data inputs above
        X_grid = padding_and_normalization(X_grid, c=0.1)

        features_grid = feature_map(X_grid)
        predictions_grid = variational_classifier(weights, bias, features_grid.T, isPlot=isPlot)
        Z = np.reshape(predictions_grid, xx.shape)

        # plot decision regions
        levels = np.arange(-1, 1.1, 0.1)
        cnt = plt.contourf(xx, yy, Z, levels=levels, cmap=cm, alpha=0.8, extend="both")
        plt.contour(
            xx,
            yy,
            Z,
            levels=[0.0],
            colors=("black",),
            linestyles=("--",),
            linewidths=(0.8,),
        )
        plt.colorbar(cnt, ticks=[-1, 0, 1])
        X_val = X[test_index]
        # plot data
        for color, label in zip(["b", "r"], [1, -1]):
            plot_x = (X_val[:, 0][predictions_val == label],)
            plot_y = (X_val[:, 1][predictions_val == label],)
            plt.scatter(
                plot_x,
                plot_y,
                c=color,
                marker="^",
                ec="k",
                label=f"class {label} validation",
            )
        plt.legend(loc="center left", bbox_to_anchor=(1.1, 0.05))
        plt.tight_layout()
        plt.show()

        # plot out a confusion matrix too
        print("Plotting the confusion matrix")
        plt.figure(figsize=figure_size)
        cm = confusion_matrix(Y_val, predictions_val)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


    return weights, bias, costs, accs


