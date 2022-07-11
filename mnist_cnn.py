import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from maxpool import Maxpooling
from reshape import Reshape
from activations import ReLu, Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 224, 224)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 224, 224), 3, 16),
    Maxpooling(),
    ReLu(),
    Convolutional((1, 111, 111), 3, 32),
    Maxpooling(),
    ReLu(),
    Convolutional((1, 54, 54), 3, 128),
    Maxpooling(),
    ReLu(),
    Convolutional((1, 26, 26), 3, 256),
    Maxpooling(),
    ReLu(),
    Reshape((256, 26, 26), (256 * 26 * 26, 1)),
    Dense(256 * 26 * 26, 20),
    ReLu(),
    Dense(20, 10),
    Softmax(),
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")