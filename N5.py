from typing import Generator

from numpy.core.fromnumeric import size
from f5 import *
import numpy as np
import random


class Network(object):
    def __init__(self, neutrons, batchSize=None, epochs=None, lr=None):
        self.lr = lr if lr else 1.0
        self.batchSize = batchSize if batchSize else 10
        self.epochs = epochs if epochs else 2000
        self.totalEpochs = 0
        self.neutrons = neutrons
        self.layers = len(neutrons)
        gen = np.random.default_rng()
        """An array of vertical vectors with biases
        for every layer except for first one"""
        self.biases = [(gen.random((neutron, 1)))*2-1
                       for neutron in neutrons[1:]]

        """An array of matrixes with weights
        between each two adjacent layers
        left from 0 to last-1, rigth from 1 to last"""
        self.weights = [(gen.random(size=(right, left)))*2-1 for left, right
                        in zip(neutrons[:-1], neutrons[1:])]

    def forward(self, a):
        """ "a" is input vertical vector"""
        for w, b in zip(self.weights, self.biases):
            """Apply activation function to each layer
            multiplication of matrix and vector + vector = vector"""
            a = activation(np.dot(w, a)+b)
        return a

    def start(self, a):
        """"Make an matrix out of array, transpose it (you get vertical vector)
        and send it to "forward"""
        return self.forward(np.transpose([a]))
        """Return is vertical vector"""

    def totalCost(self, array, correct):
        """Equation 6 from book"""
        sum = 0
        for a, c in zip(array, correct):
            """ "c" is an array, "a" is a vertical vector so we
            have to transpose "a" (we get matrix) and select first row ([0])"""
            sum += np.linalg.norm(c-np.transpose(self.start(a))[0])**2
        return sum/(2*len(correct))

    def cost(self, array, correct):
        """Calculate cost just for one input"""
        c = np.linalg.norm(correct-np.transpose(self.start(array))[0])
        return c**2

    def trainBatch(self, array, correct, batchSize):
        """Train some of the given inputs"""
        """Generate list of possible indexes"""
        indexes = [i for i in range(0, len(correct))]
        """Create matrixes for partial derivatives"""
        deltaw = [np.zeros(w.shape) for w in self.weights]
        deltab = [np.zeros(b.shape) for b in self.biases]

        for _ in range(batchSize):
            if(len(indexes) == 0):
                """If we are out of inputs"""
                break
            """Pick a random element in array, save it,
            pop it and train it"""
            r = random.randint(0, len(indexes)-1)
            index = indexes[r]
            inputA = array[index]
            correctA = correct[index]
            indexes.pop(r)
            miniw, minib = self.train(
                np.transpose([inputA]), np.transpose([correctA]))
            """ "deltaw" and "deltab" are arrays of matrixes and we
            have to add results to them respectively"""
            for dw, miw in zip(deltaw, miniw):
                dw += miw
            for db, mib in zip(deltab, minib):
                db += mib
        """ Then we have to change out weights and biases respectively,
        modyfying it by learning rate"""
        for w, dw in zip(self.weights, deltaw):
            w -= (self.lr/batchSize) * dw
        for b, db in zip(self.biases, deltab):
            b -= (self.lr/batchSize) * db

    def train(self, inp, out):
        a = inp
        activations = []
        zs = []
        nablaw = [np.zeros(w.shape) for w in self.weights]
        nablab = [np.zeros(b.shape) for b in self.biases]
        activations.append(a)
        for w, b in zip(self.weights, self.biases):
            """Apply activation function to each layer
            multiplication of matrix and vector + vector = vector"""
            z = np.dot(w, a)+b
            zs.append(z)
            a = activation(z)
            activations.append(a)
        nablab[-1] = (a-out)*activation_prime(zs[-1])
        nablaw[-1] = np.dot(nablab[-1], np.transpose(activations[-2]))
        for i in range(2, self.layers):
            nablab[-i] = \
                np.dot(np.transpose(self.weights[-i+1]), nablab[-i+1]) * \
                activation_prime(zs[-i])
            nablaw[-i] = np.dot(nablab[-i], np.transpose(activations[-i-1]))
        return nablaw, nablab

    def SGD(self, array, correct):
        """Train in epochs and show partial results"""
        print("Start cost:", self.totalCost(array, correct))
        for i in range(self.epochs):
            self.trainBatch(array, correct, self.batchSize)
            print(f"cost ({self.totalEpochs+i+1})\
                : {self.totalCost(array, correct)}")
            if(i % 10 == 0):
                self.test(array, correct)

    def show(self, array, correct):
        """Show input, output and correct (desirable) output for each input"""
        for a, c in zip(array, correct):
            print(
                f"in: {a} out: {np.transpose(self.start(a))[0]} correct: {c}")

    def test(self, array, correct):
        """Show input, output and correct (desirable) output for
        each incorrect input and show number of positive cases"""
        num = 0
        for a, c in zip(array, correct):
            r = np.transpose(self.start(a))[0]
            if(np.where(r == max(r))[0][0] == np.where(c == max(c))[0][0]):
                num += 1
        print(f"Poprawne: {num} ogółem: {len(correct)}")

    def evilTest(self, array):
        """Show input, output and correct (desirable) output for each input"""
        print("Eviltest")
        for a in array:
            r = np.transpose(self.start(a))[0]
            print(f"in: {a} out: {r}")
