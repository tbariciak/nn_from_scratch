import numpy as np


def sigmoid(Z):
    """Implements the sigmoid activation function

    Args:
        Z (np.ndarray): Output of the linear layer, with any shape

    Returns:
        A (np.ndarray): Output of sigmoid(Z), same shape as Z
    """
    A = 1 / (1 + np.exp(-Z))
    assert A.shape == Z.shape

    return A


def sigmoid_backprop(dA, Z):
    """Implements backpropagation for single sigmoid unit

    Args:
        dA (np.ndarray):
            Post-activation gradient, with any shape
        Z (np.ndarray):
            Pre-activation parameter 'Z' calculated in forward propagation

    Returns:
        dZ (np.ndarray): gradient of the cost with respect to 'Z'
    """
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ


def relu(Z):
    """Implements the ReLU function

    Args:
        Z (np.ndarray): Output of the linear layer, with any shape

    Returns:
        A (np.ndarray): Output of relu(Z), same shape as Z
    """
    A = np.maximum(0, Z)
    assert A.shape == Z.shape

    return A


def relu_backprop(dA, Z):
    """Implements backpropagation for single ReLU unit

    Args:
        dA (np.ndarray):
            Post-activation gradient, with any shape
        Z (np.ndarray):
            Pre-activation parameter 'Z' calculated in forward propagation

    Returns:
        dZ (np.ndarray): gradient of the cost with respect to 'Z'
    """
    # copying post-activation gradient
    dZ = np.array(dA, copy=True)

    # derivative of Z is 0 where Z <= 0
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape

    return dZ
