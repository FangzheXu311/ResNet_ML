# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 08:59:44 2021

@author: 50198
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import functional as F
import matplotlib.pyplot as plt
import info
from evaluator import getAUC, getACC, save_results

def load_data():
    breast_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\breastmnist.npz')  # binary, not one hot label
    chest_data = np.load('C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\chestmnist.npz')  # one hot label
    derma_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\dermamnist.npz')  # multi ,not one hot label
    oct_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\octmnist.npz')  # multi ,not one hot label
    organ_axial_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\organmnist_axial.npz')  # multi ,not one hot label
    organ_coronal_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\organmnist_coronal.npz')  # multi ,not one hot label
    organ_sagittal_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\organmnist_sagittal.npz')  # multi, not one hot
    path_data = np.load('C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\pathmnist.npz')  # multi, not one hot
    pneumonia_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\pneumoniamnist.npz')  # binary, not one hot
    retina_data = np.load(
        'C:\\Users\\50198\\Desktop\\Courses_Related\\机器学习\\MNist\\retinamnist.npz')  # multi, not one hot
    return breast_data, chest_data, derma_data, oct_data, organ_axial_data, organ_coronal_data, organ_sagittal_data, path_data, pneumonia_data, retina_data


def Standardlize(input_data):
    if len(input_data.shape) != 4:
        input_data = input_data.reshape(input_data.shape[0], 1, input_data.shape[1],
                                        input_data.shape[2])
    else:
        input_data = input_data.reshape(input_data.shape[0], input_data.shape[3], input_data.shape[1],
                                        input_data.shape[2])
        '''
    input_data_=input_data.astype(float)
    for i in range(input_data.shape[3]):
        for j in range(input_data.shape[0]):
            input_data_[j,:,:,i]=input_data[j,:,:,i].astype(float)-np.mean(input_data[j,:,:,i])
            input_data_[j,:,:,i]=input_data[j,:,:,i].astype(float)/np.std(input_data[j,:,:,i])
    return input_data_
    '''
    return input_data


def nparray2tensor(input_data):
    return torch.from_numpy(input_data)


def reverse_one_hot(labels):
    return np.array([np.argmax(one_hot) for one_hot in labels]).reshape(labels.shape[0], 1)


def one_hot_label(labels):
    labels = labels.reshape(1, -1)[0]
    return np.eye(labels.max() + 1)[labels]


# Baseblock for ResNet16
class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Bottleneck for ResNet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class inner_Conv(nn.Module):
    expansion =4

    def __init__(self, in_planes, planes, stride=1):
        super(inner_Conv, self).__init__()
        planes = planes / 16
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out

class Bottleneck_2branch(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_2branch, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.conv4 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )


    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        out1 = self.bn3(self.conv3(out1))
        out2 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out2)))
        out2 = self.bn3(self.conv3(out2))
        out = out1 + out2
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)#每次在卷积的时候通过改变stride来降采样，可以考虑改成MaxPool
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_drop(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet_drop, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(p=0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.drop(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.drop(out)
        out = self.layer4(out)#每次在卷积的时候通过改变stride来降采样，可以考虑改成MaxPool
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def data_pre_process(data, batch_size):

    if data['train_labels'].shape[1] != 1:
        train_labels = reverse_one_hot(data['train_labels'])
        val_labels = reverse_one_hot(data['val_labels'])
        test_labels = reverse_one_hot(data['test_labels'])
    else:
        train_labels = data['train_labels']
        val_labels = data['val_labels']
        test_labels = data['test_labels']

    train_images = Standardlize(data['train_images'])
    val_images = Standardlize(data['val_images'])
    test_images = Standardlize(data['test_images'])

    classes = max(train_labels) + 1
    if ((max(train_labels) + 1) == 2):
        flag = 'binary'
    else:
        flag = 'multi'

    train_images = torch.from_numpy(train_images)
    val_images = torch.from_numpy(val_images)
    test_images = torch.from_numpy(test_images)
    if flag == 'binary':
        train_labels = torch.from_numpy(one_hot_label(train_labels.reshape(train_labels.shape[0], )))
        val_labels = torch.from_numpy(one_hot_label(val_labels.reshape(val_labels.shape[0], )))
        test_labels = torch.from_numpy(one_hot_label(test_labels.reshape(test_labels.shape[0], )))
    else:
        train_labels = torch.from_numpy(train_labels.reshape(train_labels.shape[0], ))
        val_labels = torch.from_numpy(val_labels.reshape(val_labels.shape[0], ))
        test_labels = torch.from_numpy(test_labels.reshape(test_labels.shape[0], ))

    train_images = train_images.type(torch.FloatTensor)
    val_images = val_images.type(torch.FloatTensor)
    test_images = test_images.type(torch.FloatTensor)

    train_data_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size, shuffle=True, drop_last=False)

    return train_data_loader, val_data_loader, test_data_loader, flag, classes


def train_loop(dataloader, model, loss_func, optimizer, flag, device):
    model.train()
    model.to(device)
    train_loss,correct=0.0,0.0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X.requires_grad_(True)
        pred = model(X)
        if flag == 'binary':
            loss = loss_func(pred, y)
            train_loss += loss.item()
            #pred = torch.sigmoid(pred)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        elif flag == 'multi':
            loss = loss_func(pred, y.long())
            train_loss += loss.item()
            pred = F.softmax(pred,dim=1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
                if batch % 10 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                '''
    train_loss /= size
    correct /= size
    return train_loss, correct


def val_loop(dataloader, model, loss_func, val_auc_list, flag, device):
    size = len(dataloader.dataset)
    val_loss, correct = 0, 0
    model.to(device)
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if flag == 'binary':
                val_loss += loss_func(pred, y).item()
                pred = torch.sigmoid(pred)
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            elif flag == 'multi':
                val_loss += loss_func(pred, y.long()).item()
                pred = F.softmax(pred,dim=1)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_true = torch.cat((y_true, y), 0)
            y_score = torch.cat((y_score, pred), 0)
    val_loss /= size
    correct /= size
    y_true = y_true.cpu().numpy()
    if flag == 'binary':
        y_true = y_true.argmax(1).reshape(y_true.shape[0],1)
    y_score = y_score.detach().cpu().numpy()
    auc = getAUC(y_true, y_score, flag+'label')
    val_auc_list.append(auc)
    return val_loss, correct


def test_loop(dataloader, model, loss_fn, flag, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model.to(device)
    with torch.no_grad():
        model.eval()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if flag == 'binary':
                test_loss += loss_fn(pred, y).item()
                #pred = torch.sigmoid(pred)
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            elif flag == 'multi':
                test_loss += loss_fn(pred, y.long()).item()
                pred = F.softmax(pred,dim=1)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def model_run(epochs, data_loader, model, loss_func, optimizer, flag, device):
    train_loss_ls = []
    train_acc_ls = []
    val_loss_ls = []
    val_acc_ls = []
    val_auc_list = []
    metrics = dict()
    train_loader = data_loader[0]
    val_loader = data_loader[1]
    test_loader = data_loader[2]
    for epoch in range(epochs):
        train_loss, train_acc = train_loop(train_loader, model, loss_func, optimizer, flag,device)
        val_loss, val_acc = val_loop(val_loader, model, loss_func, val_auc_list, flag, device)
        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        val_loss_ls.append(val_loss)
        val_acc_ls.append(val_acc)
        print("At epoch %d, train loss: %f ,train acc: %f  val loss: %f ,val acc: %f"
              % (epoch + 1, train_loss, train_acc, val_loss, val_acc))
    test_loop(test_loader, model, loss_func, flag, device)
    plt.title("train and validation loss")
    metrics['train loss'] = train_loss_ls
    metrics['train acc'] = train_acc_ls
    metrics['val loss'] = val_loss_ls
    metrics['val acc'] = val_acc_ls
    metrics['val auc'] = val_auc_list
    return metrics


def predict(input, model, MedClass, device):#一次只能predict一张图像
    input = torch.from_numpy(Standardlize(input)).type(torch.FloatTensor).to(device)
    model.to(device)
    pred = model(input)
    result = str(pred.argmax(1).item())
    output = info.INFO[MedClass][result]
    return output
