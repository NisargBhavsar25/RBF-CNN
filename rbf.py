import numpy as np
from layer import Layer

class RBF(Layer):
    def __init__(self, input_size, output_size, centers):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.centers = centers

    def forward(self, input):
        self.input = input
        self.output = 1 - (input-self.centers)*((input-self.centers).T)
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.output.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient