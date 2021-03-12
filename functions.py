import math
import numpy as np


def sigmoid(z):
    return 1.0/(1+np.exp(-z))
