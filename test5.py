from f5 import *
from N5 import Network
import pickle
import cv2 as cv
import glob
import numpy as np
stream = open('data.pickle', 'rb')
p: Network = pickle.load(stream, encoding='bytes')
stream.close()
files = glob.glob('images/*.png')
total = len(files)
num = 0
for file in files:
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = 255-image
    image = image / 255.0
    image = image.reshape(784)
    s = p.start(image)
    print(file, "result:", np.argmax(s))
    if(int(file[-6]) == np.argmax(s)):
        num += 1
print(f"{num}/{total}")
