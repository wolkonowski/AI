from functions import *
import numpy as np
import random
import copy


class Network(object):
    def __init__(self, neutrons):
        self.lr = 0.03
        self.deltax = 0.001
        self.batchSize = 2
        self.epochs = 2000
        self.neutrons = neutrons
        self.layers = len(neutrons)
        # generate biases for all layers except for input one
        self.biases = [np.random.rand(neutron, 1) for neutron in neutrons[1:]]
        # generate weights for all connections between i-th and (i+1)-th layer
        self.weights = [np.random.rand(right, left) for left, right
                        in zip(neutrons[:-1], neutrons[1:])]

    def forward(self, a):
        # a <- input vector
        for w, b in zip(self.weights, self.biases):
            # apply sigmoid to each layer
            a = sigmoid(np.dot(w, a)+b)
            # multiplication of matrix and vector + vector = vector
        return a

    def diffForward(self, a, weights, biases):
        for w, b in zip(weights, biases):
            # apply sigmoid to each layer
            a = sigmoid(np.dot(w, a)+b)
            # multiplication of matrix and vector + vector = vector
        return a

    def start(self, a):
        return self.forward(np.transpose([a]))

    def totalCost(self, array, correct):
        sum = 0
        for a, c in zip(array, correct):
            sum += np.linalg.norm(c-np.transpose(self.start(a))[0])
        return sum/(2*len(correct))

    def cost(self, array, correct):
        # print(correct,np.transpose(self.start(array))[0])
        c = np.linalg.norm(correct-np.transpose(self.start(array))[0])
        return c

    def diffCost(self, array, correct, w, b):
        cost2 = np.linalg.norm(correct-np.transpose(
            self.diffForward(np.transpose([array]), w, b))[0])
        return cost2 - self.cost(array, correct)

    def trainBatch(self, array, correct, batchSize):
        correctC = copy.deepcopy(correct)
        arrayC = copy.deepcopy(array)
        while(len(correctC) > 0):
            deltaw = [np.zeros((right, left)) for left, right in
                      zip(self.neutrons[:-1], self.neutrons[1:])]
            deltab = [np.zeros((neutron, 1)) for neutron in self.neutrons[1:]]

            for _ in range(batchSize):
                if(len(correctC) == 0):
                    break
                r = random.randint(0, len(correctC)-1)
                inputA = arrayC[r]
                correctA = correctC[r]
                arrayC.pop(r)
                correctC.pop(r)
                result = self.train(inputA, correctA)
                for x in range(len(self.weights)):
                    for y in range(len(self.weights[x])):
                        deltaw[x][y] += result[0][x][y]
                for x in range(len(self.biases)):
                    for y in range(len(self.biases[x])):
                        deltab[x][y] += result[1][x][y]
            for x in range(len(self.weights)):
                for y in range(len(self.weights[x])):
                    self.weights[x][y] -= self.lr*deltaw[x][y]/batchSize
            for x in range(len(self.biases)):
                for y in range(len(self.biases[x])):
                    self.biases[x][y] -= self.lr*deltab[x][y]/batchSize

    def train(self, inp, out):
        biasesC = copy.deepcopy(self.biases)
        weightsC = copy.deepcopy(self.weights)
        weightsD = [np.zeros((right, left)) for left, right in
                    zip(self.neutrons[:-1], self.neutrons[1:])]
        biasesD = [np.zeros((neutron, 1)) for neutron in self.neutrons[1:]]
        for j in range(len(self.biases)):
            for i in range(len(self.biases[j])):
                biasesC[j][i] += self.deltax
                biasesD[j][i] = self.diffCost(inp, out, weightsC, biasesC) / \
                    self.deltax
                biasesC[j][i] -= self.deltax
        for j in range(len(self.weights)):
            for i in range(len(self.weights[j])):
                weightsC[j][i] += self.deltax
                weightsD[j][i] = self.diffCost(inp, out, weightsC, biasesC) / \
                    self.deltax
                weightsC[j][i] -= self.deltax
        x = [weightsD, biasesD]
        return x

    def SGD(self, array, correct):
        print(self.totalCost(array, correct))
        for _ in range(self.epochs):
            self.trainBatch(array, correct, self.batchSize)
        print(self.totalCost(array, correct))
        print(self.start(array[0]))
        print(self.start(array[1]))


p = Network([4, 10, 10, 4])
# score = p.totalCost([[-19, -29, -39, -46]], [[1, 1,1,1,1,1,1,1,1,1]])
p.SGD([[0.2, 0.1, 0.7, 0.1], [0.8, 0.9, 0.1, 0.1]],
      [[0, 0, 1, 0], [0, 1, 0, 0]])
# print(p.cost([-19, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
