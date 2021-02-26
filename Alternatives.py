#! /bin/env python3
import sys
import pickle
import numpy as np
from scipy import ndimage as ndi
import cv2 as cv
from skimage import feature
from PIL import Image
from matplotlib import pyplot as plt
from math import ceil, floor
import sklearn

def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


#Loading the Data
train_data = load_pkl('train_data.pkl')
Y=np.load('finalLabelsTrain.npy')



#print(np.size(train_data))
#print(train_data.dtype)
image=np.zeros((1,6400))
data=[]
datacanny=[]
Y=Y.astype("int32")

def standardize_image(image):
    expected_shape = (50, 50)
    padding = [(0, 0), (0, 0)]
    cropping = [(0, 50), (0, 50)]
    for d in [0, 1]:
        exp = expected_shape[d]
        act = image.shape[d]
        diff = exp - act
        if diff > 0:
            padding[d] = (ceil(diff / 2), floor(diff / 2))
        elif diff < 0:
            diff *= -1
            cropping[d] = (ceil(diff / 2), 50 + floor(diff / 2))
    image = np.pad(image, padding, 'constant', constant_values=0)
    y = cropping[0]
    x = cropping[1]
    image = image[y[0]:y[1], x[0]:x[1]]
    return image
for i in range(len(train_data)):
        image=np.array(train_data[i])
        image=image.astype(int)*255
        image = standardize_image(image)
        img = image.astype(np.uint8)
        edges = cv.Canny(img, threshold1=0, threshold2=255,apertureSize=7,L2gradient=False)
        data.append(image)
        datacanny.append(edges)


a = np.array(data)
acanny=np.array(datacanny)
#print(a.shape)
#plt.imshow(a[0])
#plt.show()
areshape=a.reshape(6400,2500)
acannyreshape=acanny.reshape(6400,2500)
#test=edges.reshape(6400,2500)

# edges=edges/255
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
train_X,test_X,train_Y,test_Y=train_test_split(areshape,Y)

#MNB

from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(train_X,train_Y)
print('Multinomial Naive Bayes',MNB.score(test_X,test_Y))
G1=MNB.score(test_X,test_Y)
MNBP=MNB.predict(test_X)
print(classification_report(test_Y,MNBP))

#MNB Canny

train_X,test_X,train_Y,test_Y=train_test_split(acannyreshape,Y)
MNB.fit(train_X,train_Y)
#print('Multinomial Naive Bayes with Canny',MNB.score(test_X,test_Y))
G2=MNB.score(test_X,test_Y)
MNBPcanny=MNB.predict(test_X)
print(classification_report(test_Y,MNBPcanny))

# LR
from sklearn.linear_model import LogisticRegression
train_X,test_X,train_Y,test_Y=train_test_split(areshape,Y)
LR=LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000,multi_class='multinomial')
LR.fit(train_X,train_Y)
print('Logistical Regression',LR.score(test_X,test_Y))
G3=LR.score(test_X,test_Y)
LRP=LR.predict(test_X)
print(classification_report(test_Y,LRP))

#LR Canny

train_X,test_X,train_Y,test_Y=train_test_split(acannyreshape,Y)
LR.fit(train_X,train_Y)
print('Logistical Regression with Canny',LR.score(test_X,test_Y))
G4=LR.score(test_X,test_Y)
LRPcanny=LR.predict(test_X)
print(classification_report(test_Y,LRPcanny))

# Perceptron

from sklearn.linear_model import Perceptron
Percep = Perceptron(tol=1e-3, random_state=0)
train_X,test_X,train_Y,test_Y=train_test_split(areshape,Y)
Percep.fit(train_X,train_Y)
print('Perceptron',Percep.score(test_X,test_Y))
G5=Percep.score(test_X,test_Y)
PP=Percep.predict(test_X)
print(classification_report(test_Y,PP))

# Perceptron Canny
train_X,test_X,train_Y,test_Y=train_test_split(acannyreshape,Y)
Percep.fit(train_X,train_Y)
print('Perceptron with Canny',Percep.score(test_X,test_Y))
G6=Percep.score(test_X,test_Y)
PPcanny=Percep.predict(test_X)
print(classification_report(test_Y,PPcanny))

# Basic plot
import numpy as np
import matplotlib.pyplot as plt
left = [1, 2, 3, 4, 5,6]
height = [G1*100, G2*100, G3*100, G4*100, G5*100,G6*100]
bars = ('MNB','MNB with Canny', 'LR', 'LR with Canny', 'Perceptron', 'Perceptron with Canny')
plt.bar(left, height, tick_label = bars, width = 0.8, color = ['red', 'green'])

plt.xlabel('Algorithms')

plt.ylabel('Accuracy')

plt.title('Comparison of different Algorithms')

plt.show()

'''
# MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=150, solver='sgd', random_state=0,learning_rate_init=.8)
train_X,test_X,train_Y,test_Y=train_test_split(areshape,Y)
mlp.fit(train_X,train_Y)
print('Multi Layer Perceptron ',mlp.score(test_X,test_Y))
'''