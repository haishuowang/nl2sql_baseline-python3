import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils import data
import random

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


def fun(data):
    data['acc_xc'] = data['acc_xg'] - data['acc_x']
    data['acc_yc'] = data['acc_yg'] - data['acc_y']
    data['acc_zc'] = data['acc_zg'] - data['acc_z']
    data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5
    data['mod'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
    data['modg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5
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


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(in_features=512, out_features=19)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.adaptive_max_pool(x)
        x = x.view(-1, 512)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_func(model, device, train_loader, optimizer, epoch):
    model.train()

    for idx, (part_train, target) in enumerate(train_loader):
        data, target = part_train.to(device), target.to(device)

        output = model(part_train)  # batch_size * 10
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1)
        part_correct = pred.eq(target.view_as(pred)).sum().item()
        acc = part_correct / len(target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if idx % 100 == 0:
        print("Train Epoch: {}, iteration: {}, Train loss: {}, Acc={}".format(
            epoch, idx, round(loss.item(), 4), round(acc * 100, 2)))


def test_func(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (part_test, target) in enumerate(test_loader):
            part_test, target = part_test.to(device), target.to(device)

            output = model(part_test)  # batch_size * 10
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)  # batch_size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    val_acc = correct / len(test_loader.dataset) * 100.
    print('*******************************')
    print("test loss: {}, val_acc: {}".format(total_loss, val_acc))
    print('*******************************')
    return val_acc


def predict_fun():
    pass


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proba_t = np.zeros((7500, 19))
    learning_rate = 0.001
    batch_size = 512
    kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        x_trn, y_trn, x_val, y_val = train_x[trn_idx], train_y[trn_idx], train_x[val_idx], train_y[val_idx]
        train_dataset = data.TensorDataset(x_trn.to(device), y_trn.to(device))
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = data.TensorDataset(x_val.to(device), y_val.to(device))
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        lr = 0.001
        model = MyNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 500
        early_stop_init = 0
        early_stop_step = 20
        max_acc = 0

        change_lr_con = True
        change_lr_rate = 0.1
        change_lr_init = 0
        change_lr_step = 16

        for epoch in range(num_epochs):
            train_func(model, device, train_loader, optimizer, epoch)
            val_acc = test_func(model, device, test_loader)
            print(f'max_val={max_acc}， early_stop_init={early_stop_init}, '
                  f'change_lr_init={change_lr_init}')
            if max_acc < val_acc:
                max_acc = val_acc
                early_stop_init = 0
                change_lr_init = 0

            elif change_lr_con:
                change_lr_init += 1
                if change_lr_init >= change_lr_step:
                    lr = lr * change_lr_rate
                    change_lr_con = False
            else:
                early_stop_init += 1
                if early_stop_init >= early_stop_step:
                    break

        torch.save(model.state_dict(), f"mnist_cnn{fold}.pt")

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
# if max_acc < acc:
#     max_acc = acc
#     early_stop = 0
# else:
#     early_stop += 1
#     if early_stop >= 50:
#         break

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
