#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module that holds the logic for making and using neural networks"""

import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_deriv(X):
    sig_X = sigmoid(X)
    return sig_X * (1 - sig_X)


class NeuralNetwork(object):
    def __init__(self, *, layers=None, activation_func=sigmoid, derivative_func=sigmoid_deriv):
        self.activation_func = activation_func
        self.derivative_func = derivative_func

        def make_random_matrix(*dims):
            return 2*np.random.rand(dims) - 1

        # Randomly initialize the first l-1 layers with an extra bias node
        self.weights = [make_random_matrix(
            layers[i - 1] + 1,
            layers[i] + 1
        ) for i in range(1, len(layers) - 1)]

        self.num_outputs = layers[-1]

    def fit(self, X, y, learning_rate=0.2, learning_rate_update=lambda lr: lr,
        max_iters=123456, error_threshold=float('-inf'), stochastic_descent=False):

        # Create a target matrix with a row for each sample and a column for each output
        y = np.eye(self.num_outputs)[y]

        for _ in range(max_iters):
            if stochastic_descent:
                # Choose a random sample to be used in this iteration
                rand_sample_idx = np.random.randint(0, X.shape[0])
                X_iter = np.array([X[rand_sample_idx]])
                y_iter = np.array([y[rand_sample_idx]])

            else:
                X_iter = X
                y_iter = y

            # Forward propagation
            a = self.predict(X_iter, return_layer_values=True)

            # Calculate errors and deltas

            # The output layer's error is calculated directly by determining the
            # difference between the predicted output and the true output
            error = y_iter - a[-1]

            if np.abs(np.mean(error)) < error_threshold:
                break

            # The required change is calculated by multiplying the 'error' of
            # the nodes by the derivative of the activation function at those
            # points. This way, if the points are far from the desired output
            # they will change more, and if they are close then they will change
            # less
            deltas = [error * self.derivative_func(a[-1])]

            # len - 1 because arr[len] is out of bounds
            # len - 2 because we calculated the delta of the output layer above
            # End at 1 because there are no weights associated with the input layer
            for l in range(len(a) - 2, 0 , -1):
                # The l'th layer's 'error' is a measure of how 'responsible'
                # each node was for the error of the final prediction.
                # This value is calculated by multiplying the amount of change needed
                # by the l+1'th layer by the weights which connect the l'th and
                # the l+1'th layers
                layer_error = np.dot(deltas[0], self.weights[l].T)

                # The deltas for the hidden layers are calculated the same as
                # for the output layer, multiply the layer's 'error' by the
                # derivative of the activation function at that point
                #
                # NOTE: Insert the new value at the front of the list so that
                #       the order of the deltas matches the order of the weights
                deltas.insert(0, layer_error * self.derivative_func(a[l]))

            # Backpropagation
            for j, (layer, delta) in enumerate(zip(a, deltas)):
                # Adjust the weights by a fraction (learning_rate) of the delta
                # from the originally calculated weight
                self.weights[j] += learning_rate * np.dot(layer.T, delta)

            learning_rate = learning_rate_update(learning_rate)

    def predict(self, X, return_layer_values=False):
        # Add the bias feature to each sample (i.e. ones column)
        X = np.insert(
            X,
            0,
            np.ones(X.shape[0]),
            axis=1
        )

        # Forward propagation
        a = [X] # Layer 0 is the input layer (which is just the raw feature vectors)

        for weight in self.weights:
            a.append(self.activation_func(np.dot(a[-1], weight)))

        if return_layer_values:
            return a

        else:
            return np.array([target.argmax() for target in a[-1])
