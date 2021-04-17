from f3 import *
from N3 import Network

"""Initialize our Network"""
p = Network(
    [4, 10, 11, 4], batchSize=20, epochs=2000)
np.set_printoptions(precision=2, suppress=True)
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

# testy deltax i lr


"""
???
kiedy prerwać uczenie
175/200 a zaraz 199/200
z 200/200 na 190/200

losowy test średnio (182--186)/200
"""
