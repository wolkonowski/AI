import math
import numpy as np


def generator(num, size, inputs, correct):
    """Clear arrays and generate random input and correct output for it"""
    inputs.clear()
    correct.clear()
    for _ in range(num):
        array = np.random.rand(size)
        array = normalize(array)
        inputs.append(array)
        c = [1 if elem == max(array) else 0 for elem in array]
        correct.append(c)


def evilGenerate(num, size, inputs):
    """Generate evil input"""
    inputs.clear()
    for _ in range(num):
        array = np.random.rand(size)
        m = np.where(array == max(array))[0][0]
        r = np.random.randint(size)
        while m == r:
            r = np.random.randint(size)
        array[r] = array[m]
        array = normalize(array)
        inputs.append(array)


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


def normalize(array):
    return (array / np.linalg.norm(array))
