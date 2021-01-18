import numpy as np
from .exceptions import InvalidActivationFunctionError
from .activation_fn import sigmoid_backprop, relu_backprop


def back_prop_layer(dA, cache, activation_fn):
    """Implements the backward propagation for the LINEAR -> ACTIVATION layer

    Args:
        dA (np.ndarray):
            Post-activation gradient for current layer
        cache (Tuple[np.ndarray, ...]):
            Tuple containing the following parameters for each layer:
                Index   Parameter
                    0   Activations from previous layer
                    1   Weight matrix for current layer
                    2   Bias vector for current layer
                    3   Pre-activation parameter for current layer
        activation_fn (str):
            String indicating activation function that was used in forward
            propagation (one of 'sigmoid' or 'relu')

    Returns:
        dA_prev (np.ndarray):
            Gradient of the cost with respect to the activations of the
            previous layer
        dW (np.ndarray):
            Gradient of the cost with respect to the weight matrix 'W' for the
            current layer
        db (np.ndarray):
            Gradient of the cost with respect to the bias vector 'b' for the
            current layer

    Raises:
        InvalidActivationFunctionError:
            Activation function string must be one of 'sigmoid', 'relu'
    """
    A_prev, W, b, Z = cache

    # number of instances
    m = A_prev.shape[1]

    if (not isinstance(activation_fn, str) or
            activation_fn not in ['sigmoid', 'relu']):
        raise InvalidActivationFunctionError(
            "Activation function must be one of 'sigmoid', 'relu'")

    # gradient of cost with respect to pre-activation parameter
    if activation_fn == 'sigmoid':
        dZ = sigmoid_backprop(dA, Z)
    else:
        dZ = relu_backprop(dA, Z)

    # gradient of cost with respect to weight matrix and bias vector
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    # gradient of cost with respect to activations of previous layer
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def back_prop_model(AL, y, caches):
    """
    Implements backward propagation for neural network with architecture
    "[LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID" where L is the number of
    layers

    Args:
        AL (np.ndarray):
            Vector containing probabilities for positive class, with shape
            (1, number of instances)
        y (np.ndarray):
            Ground truth labels with shape (1, number of instances)
        caches (List[Tuple[np.ndarray, ...], ...]):
            List of caches containing the following parameters for each layer:
                Index   Parameter
                    0   Activations from previous layer
                    1   Weight matrix for current layer
                    2   Bias vector for current layer
                    3   Pre-activation parameter for current layer

    Returns:
        grads (dict):
            Python dictionary containing the gradient of the cost with respect
            to the weight matrix, bias vector and layer activations. The
            dictionary keys are formatted as follows:

                Key String      Gradient with Respect To
                     dA{L}      activations of layer L
                     dW{L}      weight matrix of layer L
                     db{L}      bias vector of layer L

            To access the parameters for layer 1 you would use:

                dA = params['dA1']
                dW = params['dW1']
                db = params['db1']
    """
    grads = {}
    L = len(caches)  # number of layers
    y = y.reshape(AL.shape)

    # partial derivative of log loss with respect to final layer activation
    dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))

    # backpropagation of Lth (last) layer with SIGMOID activation
    current_cache = caches[-1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = back_prop_layer(
        dAL, current_cache, 'sigmoid')

    # backpropagation of hidden layers with RELU activation
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = back_prop_layer(
            grads[f'dA{l+1}'], current_cache, 'relu')

        grads[f'dA{l}'] = dA_prev
        grads[f'dW{l+1}'] = dW
        grads[f'db{l+1}'] = db

    return grads
