import numpy as np
from sqlalchemy import outparam
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class ReLu(Activation):
    def __init__(self):
        def relu(x):
            opt=[]
            for element in x:
                opt.append(element if element.any() > 0 else 0)
            return np.array(opt).resize(x.shape)
        def relu_prime(x):
            opt=[]
            for element in x:
                opt.append(1 if element.any() > 0 else 0)
            return np.array(opt).resize(x.shape)

        super().__init__(relu, relu_prime)

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
            
        def softmax_prime(x):
            n = np.size(x)
            return (np.identity(n) - x.T) * x

        super().__init__(softmax, softmax_prime)
        

class H(Activation):
    def __init__(self):
        def h(x):
            return (1 - np.pow(x, 2))
            
        def h_prime(x):
            return (-2*x)

        super().__init__(h, h_prime)