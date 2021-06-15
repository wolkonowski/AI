from f5 import *
from N5 import Network
import pickle
import cv2 as cv
stream = open('data.pickle', 'rb')
p: Network = pickle.load(stream, encoding='bytes')
stream.close()
file = 'own.png'
image = cv.imread(file, cv.IMREAD_GRAYSCALE)
image = 255-image
image = image / 255.0
image = image.reshape(784)
s = p.start(image)
print(np.argmax(s))
