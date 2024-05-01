"""A short description of the project (less than one line)."""

# Add imports here
from .vqml import * # state_preparation, layer, circuit, variational_classifier, square_loss, cost, apply_model
from .cml import * # train_svm, evaluate_model, visualize_data, apply_svm
from .pre_processing import * # transform_X, train_test_split_custom, accuracy, binary_classifier, back_trainsform, scale_data, get_angles, padding_and_normalization, feature_map
from .data_opening import * # read_data, read_banana_data

from ._version import __version__