import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from .params import init_params, update_params
from .fwd_prop import fwd_prop_model
from .back_prop import back_prop_model
from .cost import binary_crossentropy_cost


class NN:
    """
    Class for L-layer binary classification neural network with the following
    layers: [LINEAR -> RELU] * (L - 1) -> [LINEAR -> SIGMOID]
    """

    def __init__(self, layer_dims):
        """
        Args:
            layer_dims (list):
                List specifying the dimensions (number of nodes) for each layer
                in the network. The first element should be equal to the size
                of the input and the final element should be the expected
                number of outputs (must be 1 for binary classification).

                For instance, to create a 2-layer neural network with 10
                inputs, a single hidden layer with 5 neurons, and a single
                output, specify:

                layer_dims = [10, 5, 1]
        """
        self.params = init_params(layer_dims)

    def fit(self, X, y, n_epochs, learning_rate, print_cost_interval=1,
            plot_cost=True, val_size=None):
        """Trains the neural network on the provided instances

        Args:
            X (np.ndarray):
                2-D array with shape (input size, number of instances)
            y (np.ndarray):
                Ground truth labels with shape (1, number of instances)
            n_epochs (int):
                Number of epochs to train for
            learning_rate (float):
                Learning rate for the gradient descent algorithm
            print_cost_interval (int):
                Interval that cost should be printed to the console in epochs
                (e.g. print_cost_interval=10 will print the cost every 10
                epochs)
            plot_cost (bool, optional):
                If True, a plot of the cost versus epoch will be shown after
                training completes
            val_size (float, optional):
                Specifies the ratio of the training data that should be used
                as the validation set. If val_size = None, a validation set
                will not be created
        """
        costs = []

        # partition validation set if necessary
        if val_size is not None:
            idx = int(X.shape[1] * val_size)
            X_val, y_val = X[:, :idx], y[:, :idx]
            X, y = X[:, idx:], y[:, idx:]

        for i in range(1, n_epochs + 1):
            # forward propagation
            AL, caches = fwd_prop_model(X, self.params)
            cost = binary_crossentropy_cost(AL, y)

            if val_size is not None:
                # compute cost on validation set
                AL_val, _ = fwd_prop_model(X_val, self.params)
                cost_val = binary_crossentropy_cost(AL_val, y_val)

            # backward propagation
            grads = back_prop_model(AL, y, caches)

            # update parameters
            self.params = update_params(self.params, grads, learning_rate)

            # classifier accuracy on training set
            acc = accuracy_score(y.T, self.predict(X).T)

            if val_size is not None:
                acc_val = accuracy_score(y_val.T, self.predict(X_val).T)
                costs.append((cost, cost_val))

                if i % print_cost_interval == 0 or i == 1:
                    print(f'Epoch {i}, Cost {cost: .2f}, Acc {acc: .2f}, ',
                          f'Val Cost {cost_val: .2f}, Val Acc {acc_val: .2f}')
            else:
                costs.append(cost)
                if i % print_cost_interval == 0 or i == 1:
                    print(f'Epoch {i}, Cost {cost: .2f}, Acc {acc: .2f}')

        # plot cost versus epoch
        if plot_cost:
            if val_size is not None:
                plt.plot(np.array(costs)[:, 0], label='Train')
                plt.plot(np.array(costs)[:, 1], label='Val')
            else:
                plt.plot(np.squeeze(costs), label='Train')

            plt.xlabel('Epoch')
            plt.ylabel('Binary Crossentropy Cost')
            plt.title(f'Learning rate: {learning_rate}')
            plt.legend()
            plt.show()

    def predict(self, X):
        """Calculate prediction for new instance(s)

        Args:
            X (np.ndarray):
                2-D array with shape (input size, number of instances)

        Returns:
            y (np.ndarray):
                Array with shape (1, number of instances) containing the
                predicted label for each instance
        """
        probas = self.predict_probas(X)
        y = (probas > 0.5).astype(np.uint8)
        return y

    def predict_probas(self, X):
        """
        Calculate prediction for new instances(s) and return as probabilities

        Args:
            X (np.ndarray):
                2-D array with shape (input size, number of instances)

        Returns:
            probas (np.ndarray):
                Array with shape (1, number of instances) containing the
                probability that each instance belongs to the positive class
        """
        probas, _ = fwd_prop_model(X, self.params)
        return probas
