from f4 import *
from N4 import Network

"""Initialize our Network"""
p = Network(
    [4, 10, 4], batchSize=100, epochs=1000, lr=1.0)
np.set_printoptions(precision=2, suppress=True)
# print(p.weights)
# print(p.biases)
inputs = []
correct = []

"""Generate input, train it and show results"""
generator(1000, 4, inputs, correct)
p.SGD(inputs, correct)

generator(200, 4, inputs, correct)
p.test(inputs, correct)

# evilGenerate(10, 4, inputs)
# p.evilTest(inputs)

# p.SGD([[0.2, 0.1, 0.7, 0.1], [0.8, 0.9, 0.1, 0.1]],
#       [[0, 0, 1, 0], [0, 1, 0, 0]])
# print(p.cost([-19, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

"""
TODO
losowanie wag -1 do 1 OK
losowanie tylko -1 i 1 OK
losowanie wag -4 do 4 OK
losowanie wag -4 do 4 Gauss OK
przesunięcia cykliczne i permutacje
mt19937 do losowań OK
htop


MEDIANA - v5D
"""
