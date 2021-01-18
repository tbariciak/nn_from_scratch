import numpy as np


def binary_crossentropy_cost(AL, y):
    """Computes the binary cross-entropy cost

    Args:
        AL (np.ndarray):
            Vector containing probabilities for positive class, with shape
            (1, number of instances)
        y (np.ndarray):
            Ground truth labels with shape (1, number of instances)

    Returns:
        float: cross-entropy cost
    """
    # number of instances
    m = y.shape[1]

    cost = (-1/m) * np.sum(
        np.multiply(y, np.log(AL)) + (1 - y) * np.log(1 - AL))

    return cost
