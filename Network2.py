from functions import *
import numpy as np
import random
import copy


class Network(object):
    def __init__(self, neutrons):
        self.lr = 4
        self.deltax = 0.001
        self.batchSize = 2
        self.epochs = 50
        self.neutrons = neutrons
        self.layers = len(neutrons)
        # generate biases for all layers except for input one
        self.biases = [np.random.rand(neutron, 1) for neutron in neutrons[1:]]
        # generate weights for all connections between i-th and (i+1)-th layer
        self.weights = [np.random.rand(right, left) for left, right
                        in zip(neutrons[:-1], neutrons[1:])]

    def setupData(self, data, correct):
        self.data = data
        self.correct = correct

    def forward(self, a):
        # a <- input vector
        for w, b in zip(self.weights, self.biases):
            # apply sigmoid to each layer
            a = sigmoid(np.dot(w, a)-b)
            # multiplication of matrix and vector + vector = vector
        return a

    def diffForward(self, a, weights, biases):
        for w, b in zip(weights, biases):
            # apply sigmoid to each layer
            a = sigmoid(np.dot(w, a)-b)
            # multiplication of matrix and vector + vector = vector
        return a

    def start(self, a):
        return self.forward(np.transpose([a]))

    def totalCost(self, array, correct):
        sum = 0
        for a, c in zip(array, correct):
            sum += np.linalg.norm(c-np.transpose(self.start(a))[0])
        return sum/(len(correct))

    def cost(self, array, correct):
        # print(correct,np.transpose(self.start(array))[0])
        c = np.linalg.norm(correct-np.transpose(self.start(array))[0])
        return c

    def diffCost(self, w, b):
        cost2 = 0
        for a, c in zip(self.data, self.correct):
            cost2 += np.linalg.norm(c - np.transpose(self.diffForward(np.transpose([a]), w, b)[0]))
        cost2 /= len(self.correct)
        diff = cost2 - self.totalCost(self.data, self.correct)
        return diff

    def trainBatch(self, batchSize):
        correctC = copy.deepcopy(self.correct)
        arrayC = copy.deepcopy(self.data)
        while(len(correctC) > 0):
            deltaw = [np.zeros(w.shape) for w in self.weights]
            deltab = [np.zeros(b.shape) for b in self.biases]

            for _ in range(batchSize):
                if(len(correctC) == 0):
                    break
                r = random.randint(0, len(correctC)-1)
                inputA = arrayC[r]
                correctA = correctC[r]
                arrayC.pop(r)
                correctC.pop(r)
                result = self.train(inputA, correctA)
                for w in range(len(self.weights)):
                    for x in range(len(self.weights)):
                        for y in range(len(self.weights[x])):
                            for z in range(len(self.weights[x][y])):
                                deltaw[x][y][z] += result[0][x][y][z]
                                print(deltaw[x][y][z])
                for x in range(len(self.biases)):
                    for y in range(len(self.biases[x])):
                        for z in range(len(self.biases[x][y])):
                            deltab[x][y][z] += result[1][x][y][z]
            for x in range(len(self.weights)):
                for y in range(len(self.weights[x])):
                    for z in range(len(self.weights[x][y])):
                        self.weights[x][y][z] -= deltaw[x][y][z]/batchSize*self.lr
            for x in range(len(self.biases)):
                for y in range(len(self.biases[x])):
                    for z in range(len(self.biases[x][y])):
                        self.biases[x][y][z] -= deltab[x][y][z]/batchSize*self.lr

    def train(self, inp, out):
        biasesC = copy.deepcopy(self.biases)
        weightsC = copy.deepcopy(self.weights)
        weightsD = [np.zeros(w.shape) for w in self.weights]
        biasesD = [np.zeros(b.shape) for b in self.biases]
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                for k in range(len(self.biases[i][j])):
                    biasesC[i][j][k] += self.deltax
                    biasesD[i][j][k] = self.diffCost(weightsC, biasesC)
                    biasesC[i][j][k] -= self.deltax
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.biases[i][j])):
                    weightsC[i][j][k] += self.deltax
                    weightsD[i][j][k] = self.diffCost(weightsC, biasesC)
                    weightsC[i][j][k] -= self.deltax
        x = [weightsD, biasesD]
        return x

    def SGD(self):
        print(self.totalCost(self.data, self.correct))
        print(self.start(self.data[0]))
        print(self.start(self.data[1]))
        for _ in range(self.epochs):
            self.trainBatch(self.batchSize)
        print(self.totalCost(self.data, self.correct))
        print(self.start(self.data[0]))
        print(self.start(self.data[1]))


p = Network([4, 10, 10, 4])
# score = p.totalCost([[-19, -29, -39, -46]], [[1, 1,1,1,1,1,1,1,1,1]])
p.setupData([[0.2, 0.1, 0.7, 0.1], [0.8, 0.9, 0.1, 0.1]],
            [[0, 0, 1, 0], [0, 1, 0, 0]])
p.SGD()
# print(p.cost([-19, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
