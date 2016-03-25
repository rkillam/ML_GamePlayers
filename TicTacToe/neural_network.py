#!/usr/bin/python3
# -*- coding: utf-8 -*-

import abc
import numpy as np


class AbstractActivationFunc(object, metaclass=abc.ABCMeta):
    """Activation functions, and their derivatives, for neural networks"""

    @staticmethod
    @abc.abstractmethod
    def f(x):
        """The activation function"""
        pass

    @staticmethod
    @abc.abstractmethod
    def df(x):
        """The derivative of the activation function"""
        pass


class SigmoidFunc(AbstractActivationFunc):

    def f(x):
        """Sigmoid function

        @type x: float or numpy ndarray
        @param x: The value of a numpy array with the values to be put through
        the sigmoid function

        @rtype: float or numpy ndarray (depends on input)
        @return: An float or a numpy array where the sigmoid function is
        applied to each element

        >>> SigmoidFunc.f(np.array([0.5, 0.5, 0.5]))
        array([ 0.62245933,  0.62245933,  0.62245933])
        >>> SigmoidFunc.f(np.array([0.75, 0.15, 0.22, 0.36]))
        array([ 0.6791787 ,  0.53742985,  0.55477924,  0.58904043])
        >>> SigmoidFunc.f(np.array([1]))
        array([ 0.73105858])
        >>> SigmoidFunc.f(np.array([0]))
        array([ 0.5])
        >>> SigmoidFunc.f(np.array([10]))
        array([ 0.9999546])
        >>> SigmoidFunc.f(np.array([-10]))
        array([  4.53978687e-05])
        >>> SigmoidFunc.f(1)
        0.7310585786300049
        >>> SigmoidFunc.f(0)
        0.5
        >>> SigmoidFunc.f(10)
        0.99995460213129761
        >>> SigmoidFunc.f(-10)
        4.5397868702434395e-05
        """

        return 1 / (1 + np.exp(-x))

    def df(x):
        """Derivative of the sigmoid function

        @type x: float or numpy ndarray
        @param x: The value of a numpy array with the values to be put through
        the sigmoid function

        @rtype: float or numpy ndarray (depends on input)
        @return: An float or a numpy array where the derivative of the sigmoid
        function is applied to each element

        >>> SigmoidFunc.df(np.array([0.5, 0.5, 0.5]))
        array([ 0.23500371,  0.23500371,  0.23500371])
        >>> SigmoidFunc.df(np.array([0.75, 0.15, 0.22, 0.36]))
        array([ 0.21789499,  0.24859901,  0.24699924,  0.2420718 ])
        >>> SigmoidFunc.df(np.array([1]))
        array([ 0.19661193])
        >>> SigmoidFunc.df(np.array([0]))
        array([ 0.25])
        >>> SigmoidFunc.df(np.array([10]))
        array([  4.53958077e-05])
        >>> SigmoidFunc.df(np.array([-10]))
        array([  4.53958077e-05])
        >>> SigmoidFunc.df(1)
        0.19661193324148185
        >>> SigmoidFunc.df(0)
        0.25
        >>> SigmoidFunc.df(10)
        4.5395807735907655e-05
        >>> SigmoidFunc.df(-10)
        4.5395807735951673e-05
        """

        fx = SigmoidFunc.f(x)
        return fx * (1 - fx)


class NeuralNetwork(object):
    def __init__(self, **kwargs):
        self.dims = kwargs['dims']
        self.activation_func = kwargs.get('activation_func', SigmoidFunc)

        def make_weights(d1, d2):
            return 2*np.random.random((d1, d2)) - 1

        self.weights = []

        # Randomly initialize the first l - 1 layers with an extra node for the bias unit
        for i in range(len(self.dims) - 2):
            self.weights.append(make_weights(self.dims[i]+1, self.dims[i+1] + 1))

        # Add the last layer without a bias unit on the output layer
        self.weights.append(make_weights(self.dims[-2] + 1, self.dims[-1]))

    @property
    def num_inputs(self):
        return self.dims[0]

    @property
    def num_output(self):
        return self.dims[-1]

    def feed_forward(self, X):
        """Feed the input X into the neural network and return the output

        @type X: numpy ndarray
        @param X: The inputs to the neural network

        @rtype: numpy ndarray
        @return: The outputs of each layer of the network

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> nn = NeuralNetwork(dims=(3, 2, 3), activation_func=SigmoidFunc)
        >>> nn.feed_forward(np.arange(3))[-1]
        array([[ 0.43241998,  0.50483867,  0.73643123]])
        >>> len(nn.feed_forward(np.arange(3)))
        3
        """

        X = np.atleast_2d(X)

        # Add the bias feature to each sample (i.e. ones column)
        X = np.insert(
            X,
            0,
            np.ones(X.shape[0]),
            axis=1
        )

        outputs = [X]
        for weight in self.weights:
            outputs.append(self.activation_func.f(np.dot(outputs[-1], weight)))

        return outputs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
