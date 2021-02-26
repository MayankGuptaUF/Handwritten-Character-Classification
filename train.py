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


# input
data_pkl_file = "data/class_data/train_data.pkl"
label_file = "data/class_data/finalLabelsTrain.npy"

# ---------------- Hyper Parameters ----------------
use_MNIST = False # external data from EMNIST; *NOT* included in git repository due to data size
learning_rate = 0.01
batch_size = 5
weight_decay = 0
hyper_epochs = 10
sub_epochs = 10
use_larger_network = use_MNIST # If using more data, use larger network
# --------------------------------------------------


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
    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=10*10*12, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=9)
        
    def forward(self, X):
        X = self.pool(self.relu(self.conv1(X)))
        X = self.pool(self.relu(self.conv2(X)))
        X = X.view(-1, 10*10*12)
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
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

def calculate_accuracy(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc


def _train(dataset, epochs, batch_size=5, to_cuda=False, calculate_acc=False):
    train_acc = []
    val_acc = []
    if to_cuda:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if to_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('epoch:', epoch, ', loss:', running_loss * batch_size / dataloader.__len__())
        if calculate_acc:
            t = calculate_accuracy(dataloader)
            v = calculate_accuracy(val_loader)
            print("train acc:", t, "val acc:", v)
            train_acc.append(t)
            val_acc.append(v)
        if epoch % 1 == 0: # every x epochs reshuffle data by creating a new loader
            if to_cuda:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            else:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_acc, val_acc


def class_wise_accuracy(dataloader):
    correct = {}
    total = {}
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for l in range(len(labels)):
                label = int(labels[l].cpu().numpy())
                pred = predicted[l]
                if label in correct:
                    correct[label] += int(pred == label)
                    total[label] += 1
                else:
                    correct[label] = int(pred == label)
                    total[label] = 1
    acc = {}
    for label in correct:
        acc[label] = 100 * correct[label] / total[label]
    return acc

# print(class_wise_accuracy(val_loader))



    

cr_train = ClassroomDataset(data_pkl_file, label_file)
if use_MNIST:
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    Parth_preprocessed_dataset = ImageFolder(
        root="data/Parth_processed/",
        transform=transform
    )
    extra_train = Parth_preprocessed_dataset

# Dividing data into 50% train, 40% val, 30% test
scores = []
for i in range(1):
    print("Fold number", i)
    if use_larger_network:
        net = Net2().float()
    else:
        net = Net().float()
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    acc = []
    for j in range(hyper_epochs):
        val_loader = DataLoader(cr_train, batch_size=1, shuffle=True)
        if use_MNIST:
            to_cuda = torch.cuda.is_available()
            _train(extra_train, epochs=ceil(sub_epochs / 3), batch_size=batch_size, to_cuda=to_cuda)
        train_accuracy, val_accuracy = _train(cr_train, batch_size=batch_size, to_cuda=False, epochs=sub_epochs)
        print(train_accuracy, val_accuracy)
        acc.append((train_accuracy, val_accuracy))
    scores.append(acc)

p = "weights/train.pth"
torch.save(net.state_dict(), p)