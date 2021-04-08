import math
import numpy as np


def generator(num, size, inputs, correct):
    """Generate random input and correct output for it"""
    for _ in range(num):
        array = np.random.rand(size)
        array = array / np.linalg.norm(array)	# normalisation - mgc
        inputs.append(array)
        c = [1 if elem == max(array) else 0 for elem in array]
        correct.append(c)


def activation(z):
    return sigmoid(z)
    # return tanh(z)
    # return ReLU(z)


def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def tanh(z):
    return np.tanh(z)


def ReLU(z):
    vf = np.vectorize(insideReLU)
    return vf(z)


def insideReLU(x):
    return x if x >= 0 else 0
