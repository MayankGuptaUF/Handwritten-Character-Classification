from data.class_data import loaddata
import numpy as np
from math import ceil, floor
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader

class ClassroomDataset(Dataset):
    
    @staticmethod
    def standardize_image(image):
        expected_shape = (50,50)
        padding = [(0,0), (0,0)]
        cropping = [(0,50), (0, 50)]
        for d in [0,1]:
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
    
    def __init__(self, data_file, label_file=None):
        # path = "data/class_data/train_data.pkl"
        path = data_file
        cls_data = loaddata.load_pkl(path)
        # label_path = "data/class_data/finalLabelsTrain.npy"
        if label_file:
            labels = np.load(label_file) 
        else: 
            labels = np.zeros(len(cls_data))
        self.data = []
        gpu = False
        for i in range(len(cls_data)):
            image = np.array(cls_data[i])
            image = image.astype(int)
            image = self.standardize_image(image)
            image = torch.from_numpy(np.array([image])).float()
            label = torch.tensor(int(labels[i]))
            pair = (image, label)
            self.data.append(pair)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=10*10*12, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=9)
        
    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv1a(X))
        X = self.pool(self.relu(self.conv2(X)))
        X = self.relu(self.conv3(X))
        X = self.relu(self.conv3a(X))
        X = self.pool(self.relu(self.conv4(X)))
        X = X.view(-1, 10*10*12)
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X


weights = "2019-12-02 13_47_01.pth"
PATH = "./weights/" + weights
net = Net2().float()
net.load_state_dict(torch.load(PATH))


def process_pred(data):
    if np.all(data < 5):
        return -1
    else:
        s = np.sort(data)
        if (s[-1] - s[-2]) < 5:
            return -1
        else:
            return 0
    

def predict(data_pkl_file):
    """
    data_pkl_file: file path to the pkl file to be loaded.
    """
    dataset = ClassroomDataset(data_pkl_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            none_pred = process_pred(outputs.data.numpy()[0])
            if none_pred == 0:
                pred = predicted.numpy()[0]
            else:
                pred = -1
            predictions.append(pred)
    return predictions




p = predict("data/class_data/train_data.pkl")
print(p)
print(len(np.where(p==-1)[0]))