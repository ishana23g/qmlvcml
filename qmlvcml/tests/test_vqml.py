
from unittest.mock import patch 

import pennylane as qml
from pennylane import numpy as np

from qmlvcml import *

import pytest

@qml.qnode(dev)
def apply_state_prep(angles):
    state_preparation(angles)
    return qml.state()

def test_state_preparation():
    x = np.array([0.53896774, 0.79503606, 0.27826503, 0.0], requires_grad=False)
    ang = get_angles(x)
    assert np.allclose(ang, [0.563975, -0., 0., -0.975046, 0.975046]), "Did not get the correct angles"
    # same as test_get_angles
    ang_amplituides = np.real(apply_state_prep(ang))
    assert np.allclose(ang_amplituides, [0.53896774, 0.79503606, 0.27826503, 0.0]), "Did not get the correct amplitudes"

def test_with_banana(monkeypatch):
    banana_df_X, banana_df_y = read_banana_data()
    monkeypatch.setattr(plt, 'show', lambda: None)
    apply_model(banana_df_X, banana_df_y, steps=1,
                     batch_size_percent=.8, isPlot=True, isDebug=True,
                     dim_reduce_type='trimap')
    # no debug
    apply_model(banana_df_X, banana_df_y, steps=1,
                     batch_size_percent=.8, isPlot=True, isDebug=False,
                     dim_reduce_type='trimap')
    # no plot
    apply_model(banana_df_X, banana_df_y, steps=1,
                     batch_size_percent=.8, isPlot=False, isDebug=False,
                     dim_reduce_type='trimap')
