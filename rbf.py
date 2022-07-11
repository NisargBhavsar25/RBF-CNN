import numpy as np
from layer import Layer

class RBF(Layer):
    def __init__(self, input_size, output_size, center):
        self.center = center

    def forward(self, input):
        self.input = np.dot((input - self.center), (input + self.center))
        return self.input

    # def backward(self, output_gradient, learning_rate):
    #     weights_gradient = np.dot(output_gradient, self.input.T)
    #     input_gradient = np.dot(self.weights.T, output_gradient)
    #     self.weights -= learning_rate * weights_gradient
    #     self.bias -= learning_rate * output_gradient
    #     return input_gradient