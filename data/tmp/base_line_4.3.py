import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
import torch.optim as optim
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
from tqdm import tqdm
from torch.utils import data
import random

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
# random.seed(53113)
# np.random.seed(53113)
# torch.manual_seed(53113)
# if USE_CUDA:
#     torch.cuda.manual_seed(53113)

pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


def fun(data):
    # data['acc_xc'] = data['acc_xg'] - data['acc_x']
    # data['acc_yc'] = data['acc_yg'] - data['acc_y']
    # data['acc_zc'] = data['acc_zg'] - data['acc_z']
    # data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5
    # data['mod'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
    # data['modg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5
    return data


def get_data():
    root_path = ''
    train = pd.read_csv(root_path + 'sensor_train.csv')
    test = pd.read_csv(root_path + 'sensor_test.csv')
    sub = pd.read_csv(root_path + '提交结果示例.csv')
    train_y = train.groupby('fragment_id')['behavior_id'].min()
    train_y = torch.from_numpy(train_y.values).long()
    train = fun(train)
    test = fun(test)
    return train, test, train_y, sub


def data_tensor(test, train):
    train_x = torch.zeros((7292, 1, 60, len(use_feat)))
    test_x = torch.zeros((7500, 1, 60, len(use_feat)))

    for i in tqdm(range(7292)):
        tmp = train[train.fragment_id == i][:60]
        data, label = resample(tmp[use_feat], 60, np.array(tmp.time_point))
        train_x[i, 0, :, :] = torch.from_numpy(data)

    for i in tqdm(range(7500)):
        tmp = test[test.fragment_id == i][:60]
        data, label = resample(tmp[use_feat], 60, np.array(tmp.time_point))
        test_x[i, 0, :, :] = torch.from_numpy(data)
    return train_x, test_x


train, test, train_y, sub = get_data()
use_feat = [f for f in train.columns if f not in ['fragment_id', 'time_point', 'behavior_id']]
train_x, test_x = data_tensor(test, train)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        nn.Sequential()
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.max_pool(x)
        x = nn.Dropout(0.2)(x)
        x = self.relu(self.conv3(x))

        x = nn.Dropout(0.3)(x)
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)

        x = nn.Dropout(0.5)(x)
        x = x.reshape(x.shape[0], -1)

        x = nn.Linear(in_features=x.shape[1], out_features=256)(x)
        # y = y.transpos(dim0=0, dim1=1)
        # print(y.shape)
        x = nn.Linear(in_features=x.shape[1], out_features=19)(x)
        x = nn.Softmax()(x)
        return x


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = nn.Dropout(0.2)(self.pool(F.relu(self.conv1(x))))
#         x = nn.Dropout(0.3)(self.pool(F.relu(self.conv2(x))))
#
#         x = x.view(x.shape[0], -1)  # 将数据平整为一维的
#         x = F.relu(nn.Linear(x.shape[1], 1024)(x))
#         x = F.relu(nn.Linear(x.shape[1], 512)(x))
#         x = nn.Linear(x.shape[1], 19)(x)
#         return x


proba_t = np.zeros((7500, 19))

epochs = 100
learning_rate = 0.001
batch_size = 500
kfold = StratifiedKFold(n_splits=5, random_state=2020)

for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
    model = LeNet().cuda()
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    x_trn, y_trn, x_val, y_val = train_x[trn_idx], train_y[trn_idx], train_x[val_idx], train_y[val_idx]
    torch_dataset = data.TensorDataset(x_trn.cuda(), y_trn.cuda())
    train_loader = data.DataLoader(torch_dataset, batch_size=batch_size)
    for epoch in range(1, epochs + 1):
        for i, (x_trn_b, y_trn_b) in enumerate(train_loader):
            output = model(x_trn_b)
            optimizer.zero_grad()
            loss = loss_func(output, y_trn_b)
            loss.backward()
            optimizer.step()
            print('Epoch: ', epoch, 'batch_size', i, '| train loss: %.4f' % loss.cpu().data.numpy())

        output_ = model(x_val.cuda())
        output_ = torch.argmax(output_, dim=1)
        print(output_)
        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
    # max_acc = 0
    # early_stop = 0
    # Adam = optim.Adam(model.parameters(), lr=learning_rate)

    #     # model.train()
    #
    #
    #     Adam.zero_grad()

# certion = nn.CrossEntropyLoss()
# epochs = 5000
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
# for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
#     x_trn, y_trn, x_val, y_val = train_x[trn_idx], train_y[trn_idx], train_x[val_idx], train_y[val_idx]
#     # x_trn = torch.from_numpy(np.array(x_trn))
#     # x_val = torch.from_numpy(np.array(x_val))
#     # y_trn = torch.from_numpy(np.array(y_trn, dtype='int'))
#     # y_val = torch.from_numpy(np.array(y_val, dtype='int'))
#     # print(x_trn.shape)
#     max_acc = 0
#     early_stop = 0
#     SGD = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     for epoch in range(1, epochs + 1):
#         model.train()
#         SGD.zero_grad()
#         output = model(x_trn)
#         loss = certion(output, y_trn)
#         loss.backward()
#         SGD.step()
#
#         model.eval()
#         output_ = model(x_val)
#         output_ = torch.argmax(output_, dim=1)
#         cnt = sum(output_ == y_val).item()
#         len_dataset = len(y_val)
#         acc = cnt / len_dataset * 100
#         print("Epoch{}: Accu in val_set is {}.".format(epoch, acc))
#
#         if max_acc < acc:
#             max_acc = acc
#             early_stop = 0
#         else:
#             early_stop += 1
#             if early_stop >= 50:
#                 break

# a = torch.randn(4, 3, 28, 28)
# aa = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)(a)
# b = nn.MaxPool2d(kernel_size=2)(aa)
# c = nn.Dropout(0.5)(b)
# print(a.shape)
# print(aa.shape)
# print(b.shape)
# print(c.shape)
# cc = c.view([4, 64*14 *14])
# # cc = c.view([4, 64*14 *14])
# print('cc',cc.shape)
# d = nn.Linear(in_features=64*14 *14, out_features=19)(cc)
# print(d.shape)
