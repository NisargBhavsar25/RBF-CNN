import numpy as np
from layer import Layer

class RBF(Layer):
    def __init__(self, input_size, output_size, centres):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.centers = centres

    def forward(self, input):
        self.input = input
        self.r2 = (input - self.centers)*(input - self.centers).T
        return np.dot(self.weights, self.r2) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.r2.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient