from f5 import *
from N5 import Network
import loader
import pickle

training_data, validation_data, test_data = loader.load_data_wrapper()
training_data = list(training_data)
"""Initialize our Network"""
p = Network(
    [784, 30, 10], batchSize=100, epochs=1000, lr=3)
np.set_printoptions(precision=2, suppress=True)
# print(p.weights)
# print(p.biases)
inputs = [np.transpose(training_data[i][0])[0]
          for i in range(len(training_data))]
correct = [np.transpose(training_data[i][1])[0]
           for i in range(len(training_data))]
test_data = list(test_data)
testI = [np.transpose(test_data[i][0])[0] for i in range(len(test_data))]
testC = [test_data[i][1] for i in range(len(test_data))]
testC = [[1 if x == c else 0 for x in range(0, 10)] for c in testC]
"""Generate input, train it and show results"""
p.SGD(inputs, correct)
p.test(testI, testC)
stream = open('data.pickle', 'w+b')
pickle.dump(p, stream)
stream.close()

# stream = open('data.pickle', 'rb')
# p = pickle.load(stream, encoding='bytes')
# stream.close()
# p.test(inputs, correct)
# evilGenerate(10, 4, inputs)
# p.evilTest(inputs)

# p.SGD([[0.2, 0.1, 0.7, 0.1], [0.8, 0.9, 0.1, 0.1]],
#       [[0, 0, 1, 0], [0, 1, 0, 0]])
# print(p.cost([-19, 2, 3, 4], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

"""
TODO
cyfry!!!
wejść na kartę graficzną
"""
