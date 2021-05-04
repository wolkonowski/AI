from f4 import *
from N4 import Network

"""Initialize our Network"""
p = Network(
    [4, 10, 4], batchSize=500, epochs=200)
np.set_printoptions(precision=2, suppress=True)
inputs = []
correct = []

"""Generate input, train it and show results"""
generator(1000, 4, inputs, correct)
p.SGD(inputs, correct)

generator(200, 4, inputs, correct)
p.test(inputs, correct)
print(p.weights)
print(p.biases)
# evilGenerate(10, 4, inputs)
# p.evilTest(inputs)

# p.SGD([[0.2, 0.1, 0.7, 0.1], [0.8, 0.9, 0.1, 0.1]],
#       [[0, 0, 1, 0], [0, 1, 0, 0]])
# print(p.cost([-19, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
