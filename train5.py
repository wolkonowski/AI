from f5 import *
from N5 import Network
import pickle
import loader

stream = open('data2.pickle', 'rb')
p: Network = pickle.load(stream, encoding='bytes')
stream.close()

p.epochs = 100

training_data, validation_data, test_data = loader.load_data_wrapper()
training_data = list(training_data)
inputs = [np.transpose(training_data[i][0])[0]
          for i in range(len(training_data))]
correct = [np.transpose(training_data[i][1])[0]
           for i in range(len(training_data))]
test_data = list(test_data)
testI = [np.transpose(test_data[i][0])[0] for i in range(len(test_data))]
testC = [test_data[i][1] for i in range(len(test_data))]
testC = [[1 if x == c else 0 for x in range(0, 10)] for c in testC]
testC = np.array(testC)
p.SGD(inputs, correct)
p.test(testI, testC)
p.totalEpochs += p.epochs
stream = open('data2.pickle', 'w+b')
pickle.dump(p, stream)
stream.close()
