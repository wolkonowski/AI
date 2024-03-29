from functions import *
import numpy as np
import random
import copy


class Network(object):
    def __init__(self, neutrons):
        self.lr = 1
        self.deltax = 0.001
        self.batchSize = 5
        self.epochs = 30
        self.neutrons = neutrons
        self.layers = len(neutrons)

        """An array of vertical vectors with biases
        for every layer except for first one"""
        self.biases = [np.random.rand(neutron, 1) for neutron in neutrons[1:]]

        """An array of matrixes with weights
        between each two adjacent layers
        left from 0 to last-1, rigth from 1 to last"""
        self.weights = [np.random.rand(right, left) for left, right
                        in zip(neutrons[:-1], neutrons[1:])]

    def forward(self, a):
        """ "a" is input vertical vector"""
        for w, b in zip(self.weights, self.biases):
            """Apply activation function to each layer
            multiplication of matrix and vector + vector = vector"""
            a = activation(np.dot(w, a)+b)
        return a

    def diffForward(self, a, weights, biases):
        """Same as "forward" but with custom weights and biases"""
        for w, b in zip(weights, biases):
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
            sum += np.linalg.norm(c-np.transpose(self.start(a))[0])
        return sum/(2*len(correct))

    def cost(self, array, correct):
        """Calculate cost just for one input"""
        c = np.linalg.norm(correct-np.transpose(self.start(array))[0])
        return c

    def diffCost(self, array, correct, w, b):
        """Calculate differences of costs for the single input between
        "cost2" with custom weights and biases and
        original cost for those input"""
        cost2 = np.linalg.norm(correct-np.transpose(
            self.diffForward(np.transpose([array]), w, b))[0])
        return cost2 - self.cost(array, correct)

    def trainBatch(self, array, correct, batchSize):
        """Train the whole given inputs"""

        """Copy all arrays of inputs and corrects to randomly pick them
        (and then delete) to train"""
        correctC = copy.deepcopy(correct)
        arrayC = copy.deepcopy(array)
        while(len(correctC) > 0):
            """Create matrixes for partial derivatives"""
            deltaw = [np.zeros(w.shape) for w in self.weights]
            deltab = [np.zeros(b.shape) for b in self.biases]

            for _ in range(batchSize):
                if(len(correctC) == 0):
                    """If we are out of inputs"""
                    break
                """Pick a random element in array, save it,
                pop it and train it"""
                r = random.randint(0, len(correctC)-1)
                inputA = arrayC[r]
                correctA = correctC[r]
                arrayC.pop(r)
                correctC.pop(r)
                result = self.train(inputA, correctA)
                """ "result[0]" <- nablaw, "result[1]" <- nablab
                "deltaw" and "deltab" are arrays of matrixes and we
                have to add results to them respectively"""
                for x in range(len(self.weights)):
                    for y in range(len(self.weights[x])):
                        for z in range(len(self.weights[x][y])):
                            deltaw[x][y][z] += result[0][x][y][z]
                for x in range(len(self.biases)):
                    for y in range(len(self.biases[x])):
                        for z in range(len(self.biases[x][y])):
                            deltab[x][y][z] += result[1][x][y][z]
            """ Then we have to change out weights and biases respectively,
            modyfying it by learning rate"""
            for x in range(len(self.weights)):
                for y in range(len(self.weights[x])):
                    for z in range(len(self.weights[x][y])):
                        self.weights[x][y][z] -= self.lr*deltaw[x][y][z] / \
                            batchSize
            for x in range(len(self.biases)):
                for y in range(len(self.biases[x])):
                    for z in range(len(self.biases[x][y])):
                        self.biases[x][y][z] -= self.lr*deltab[x][y][z] / \
                            batchSize

    def train(self, inp, out):
        """Copy all biases and weights and generate proper arrays of
        matrixes for partial derivatives"""
        biasesC = copy.deepcopy(self.biases)
        weightsC = copy.deepcopy(self.weights)
        weightsD = [np.zeros(w.shape) for w in self.weights]
        biasesD = [np.zeros(b.shape) for b in self.biases]
        """For each bias and weight:
        add "deltax", calculate derivative for that change,
        substract "deltax" """
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                for k in range(len(self.biases[i][j])):
                    biasesC[i][j][k] += self.deltax
                    biasesD[i][j][k] = self.diffCost(
                        inp, out, weightsC, biasesC) / self.deltax
                    biasesC[i][j][k] -= self.deltax
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    weightsC[i][j][k] += self.deltax
                    weightsD[i][j][k] = self.diffCost(
                        inp, out, weightsC, biasesC) / self.deltax
                    weightsC[i][j][k] -= self.deltax
        """Pack output into one array"""
        x = [weightsD, biasesD]
        return x

    def SGD(self, array, correct):
        """For each epoch train the whole input"""
        print("Start cost:", self.totalCost(array, correct))
        for _ in range(self.epochs):
            self.trainBatch(array, correct, self.batchSize)
        print("End cost:", self.totalCost(array, correct))

    def show(self, array, correct):
        """Show input, output and correct (desirable) output for each input"""
        for a, c in zip(array, correct):
            print("in:", a,
                  "out:", np.transpose(self.start(a))[0], "correct:", c)


"""Initialize our Network"""
p = Network([4, 10, 11, 4])

inputs = []
correct = []

"""Generate input, train it and show results"""
generator(20, 4, inputs, correct)
p.SGD(inputs, correct)
p.show(inputs, correct)


# p.SGD([[0.2, 0.1, 0.7, 0.1], [0.8, 0.9, 0.1, 0.1]],
#       [[0, 0, 1, 0], [0, 1, 0, 0]])
# print(p.cost([-19, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
