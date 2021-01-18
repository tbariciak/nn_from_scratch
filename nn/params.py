import numpy as np


def init_params(layer_dims):
    """
    Initializes parameters of L-layer neural network

    The weight matrices are initialized using random initialization, whereas
    the biases are initialized with zeros.

    Args:
        layer_dims (list):
            List specifying the dimensions (number of nodes) for each layer in
            the network. The first element should be equal to the size of the
            inputs and the final element should be the expected number of
            outputs (must be 1 for binary classification).

            For instance, to create a 2-layer neural network with 10 inputs, a
            single hidden layer with 5 neurons, and a single output, specify:

            layer_dims = [10, 5, 1]

    Returns:
        params (dict):
            Python dictionary containing the weight matrix and bias vector for
            each layer in the neural network. The dictionary keys are formatted
            as follows:

                W{L}: weight matrix of layer L
                b{L}: bias vector of layer L

            To access the parameters for layer 1 you would use:

                W = params['W1']
                b = params['b1']
    """
    params = {}
    L = len(layer_dims)  # number of layers

    for l in range(1, L):
        # use random initialization for weight matrix
        params[f'W{l}'] = np.random.randn(
            layer_dims[l], layer_dims[l-1]) * 0.01

        # use zero initialization for the bias vector
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))

        assert params[f'W{l}'].shape == (layer_dims[l], layer_dims[l-1])
        assert params[f'b{l}'].shape == (layer_dims[l], 1)

    return params


def update_params(params, grads, learning_rate):
    """Update neural network parameters using gradient descent

    Args:
        params (dict):
            Python dictionary containing the weight matrix and bias vector
            for each layer
        grads (dict):
            Python dictionary containing the gradient of the cost with respect
            to the weight matrix, bias vector and activations for each layer
        learning_rate (float):
            Learning rate for the model

    Returns:
        params (dict):
            Python dictionary with updated parameters. The dictionary keys are
            formatted as follows:

                W{L}: weight matrix of layer L
                b{L}: bias vector of layer L
    """
    L = len(params) // 2  # number of layers

    for l in range(1, L+1):
        params[f'W{l}'] = params[f'W{l}'] - learning_rate * grads[f'dW{l}']
        params[f'b{l}'] = params[f'b{l}'] - learning_rate * grads[f'db{l}']

    return params
