from f3 import *
import numpy as np
import random
import copy


class Network(object):
    def __init__(self, neutrons, deltax=None, batchSize=None, epochs=None):
        self.lr = 3
        self.deltax = deltax if deltax else 0.001  # 0.001 - reference
        self.batchSize = batchSize if batchSize else 10	 # 10 for 200 examples
        self.epochs = epochs if epochs else 2000  # 1000 - reference
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
            sum += np.linalg.norm(c-np.transpose(self.start(a))[0])**2
        return sum/(2*len(correct))

    def cost(self, array, correct):
        """Calculate cost just for one input"""
        c = np.linalg.norm(correct-np.transpose(self.start(array))[0])
        return c**2

    def diffCost(self, array, correct, w, b):
        # DEPRECATED
        """Calculate differences of costs for the single input between
        "cost2" with custom weights and biases and
        original cost for those input"""
        cost2 = np.linalg.norm(correct-np.transpose(
            self.diffForward(np.transpose([array]), w, b))[0])
        return cost2**2

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
            index = indexes[r]  # changes here
            inputA = array[index]
            correctA = correct[index]
            indexes.pop(r)
            nablaw, nablab = self.train(inputA, correctA)
            """ "deltaw" and "deltab" are arrays of matrixes and we
            have to add results to them respectively"""
            for x in range(len(self.weights)):
                for y in range(len(self.weights[x])):
                    for z in range(len(self.weights[x][y])):
                        deltaw[x][y][z] += nablaw[x][y][z]
            for x in range(len(self.biases)):
                for y in range(len(self.biases[x])):
                    for z in range(len(self.biases[x][y])):
                        deltab[x][y][z] += nablab[x][y][z]
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

        cost = self.cost(inp, out)
        weightsD = [np.zeros(w.shape) for w in self.weights]
        biasesD = [np.zeros(b.shape) for b in self.biases]
        """For each bias and weight:
        add "deltax", calculate derivative for that change,
        substract "deltax" """
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                for k in range(len(self.biases[i][j])):
                    self.biases[i][j][k] += self.deltax  # changes here
                    biasesD[i][j][k] = (self.cost(
                        inp, out) - cost) / self.deltax
                    self.biases[i][j][k] -= self.deltax
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += self.deltax
                    weightsD[i][j][k] = (self.cost(
                        inp, out) - cost) / self.deltax
                    self.weights[i][j][k] -= self.deltax
        """Return derivatives"""
        return weightsD, biasesD  # change here

    def SGD(self, array, correct):
        """Train in epochs and show partial results"""
        print("Start cost:", self.totalCost(array, correct))
        for i in range(self.epochs):
            self.trainBatch(array, correct, self.batchSize)
            if(i % 10 == 0):
                print(f"cost ({i}): {self.totalCost(array, correct)}")
            if((i+1) % 100 == 0):
                self.test(array, correct)

    def show(self, array, correct):
        """Show input, output and correct (desirable) output for each input"""
        for a, c in zip(array, correct):
            print(
                f"in: {a} out: {np.transpose(self.start(a))[0]} correct: {c}")

    def test(self, array, correct):
        """Show input, output and correct (desirable) output for
        each incorrect input and show number of positive cases"""
        print("Błędne dopasowania:")
        num = 0
        for a, c in zip(array, correct):
            r = np.transpose(self.start(a))[0]
            if(np.where(r == max(r))[0][0] == c.index(max(c))):
                num += 1
            else:
                print(f"in: {a} out: {r} correct: {c}")
        print(f"Poprawne: {num} ogółem: {len(correct)}")

    def evilTest(self, array):
        """Show input, output and correct (desirable) output for each input"""
        print("Eviltest")
        for a in array:
            r = np.transpose(self.start(a))[0]
            print(f"in: {a} out: {r}")
