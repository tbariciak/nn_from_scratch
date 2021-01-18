import numpy as np
from .activation_fn import sigmoid, relu
from .exceptions import InvalidActivationFunctionError


def fwd_prop_layer(A_prev, W, b, activation_fn):
    """Implements the forward propagation for the LINEAR -> ACTIVATION layer

    Args:
        A_prev (np.ndarray):
            Activations from previous layer with shape (size of previous layer,
            number of instances)
        W (np.ndarray):
            Weight matrix for current layer, with shape (size of current layer,
            size of previous layer)
        b (np.ndarray):
            Bias vector for current layer, with shape
            (size of current layer, 1)
        activation_fn (str):
            String indicating activation function to be used
            (one of 'sigmoid' or 'relu')

    Returns:
        A (np.ndarray):
            Activations for current layer
        cache (Tuple[np.ndarray, ...]):
            Tuple containing the following parameters (for backpropagation):
                Index   Parameter
                    0   Activations from previous layer
                    1   Weight matrix for current layer
                    2   Bias vector for current layer
                    3   Pre-activation parameter for current layer

    Raises:
        InvalidActivationFunctionError:
            Activation function string must be one of 'sigmoid', 'relu'
    """
    # pre-activation parameter
    Z = np.dot(W, A_prev) + b
    assert Z.shape == (W.shape[0], A_prev.shape[1])

    if (not isinstance(activation_fn, str) or
            activation_fn not in ['sigmoid', 'relu']):
        raise InvalidActivationFunctionError(
            "Activation function must be one of 'sigmoid', 'relu'")

    # apply activation function
    if activation_fn == 'sigmoid':
        A = sigmoid(Z)
    else:
        A = relu(Z)

    # activation function should not change array shape
    assert A.shape == Z.shape
    cache = (A_prev, W, b, Z)

    return A, cache


def fwd_prop_model(X, params):
    """
    Implements forward propagation for neural network with architecture
    "[LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID" where L is the number of
    layers

    Args:
        X (np.ndarray):
            Input data with shape (input size, number of instances)
        params (dict):
            Weight matrix and bias vector values for each layer in the network.
            Can be initialized using initialize_params()

    Returns:
        AL (np.ndarray):
            Activation output of final layer (layer L)
        caches (List[Tuple[np.ndarray, ...], ...]):
            List of caches containing the following parameters for each layer:
                Index   Parameter
                    0   Activations from previous layer
                    1   Weight matrix for current layer
                    2   Bias vector for current layer
                    3   Pre-activation parameter for current layer
    """
    caches = []
    L = len(params) // 2  # number of layers

    # activations for first layer are the inputs
    A = X

    # [LINEAR -> RELU] * (L - 1) (hidden layers)
    for l in range(1, L):
        A_prev = A
        A, cache = fwd_prop_layer(
            A_prev, params[f'W{l}'], params[f'b{l}'], 'relu')
        caches.append(cache)

    # LINEAR -> SIGMOID (output layer)
    AL, cache = fwd_prop_layer(
        A, params[f'W{L}'], params[f'b{L}'], 'sigmoid')
    caches.append(cache)

    # activations of final layer should have shape (1, number of instances)
    assert AL.shape == (1, X.shape[1])

    return AL, caches
