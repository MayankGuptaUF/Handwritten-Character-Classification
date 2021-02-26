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

# path_prefix = "C:/Users/parth/OneDrive/Documents/Acad/U_Florida/Fall_19/EEL58404930_Fundamentals_of_Machine_Learning/Project"
# train_data = load_pkl(path_prefix+'./data/class_data/train_data.pkl')
# Y=np.load(path_prefix+'./data/class_data/finalLabelsTrain.npy')

train_data = load_pkl('./data/class_data/train_data.pkl')
Y=np.load('./data/class_data/finalLabelsTrain.npy')



print(np.size(train_data))
print(train_data.dtype)
image=np.zeros((1,6400))
data=[]
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
        data.append(image)



a = np.array(data)
print(a.shape)
plt.imshow(a[0])
plt.show()
img = a.astype(np.uint8)
edges = cv.Canny(img, 0, 255)
# edges=edges/255
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(edges,Y)
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(train_X,train_Y)
print(MNB.score(train_X,train_Y))
print(MNB.score(test_X,test_Y))

'''
print(a[1])
img = image.astype(np.uint8)
edges = cv.Canny(img, 0, 255)
edges=edges/255
'''