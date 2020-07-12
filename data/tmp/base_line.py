import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

# from torchvision import datasets, transforms
print("PyTorch Version: ", torch.__version__)


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)

        x = F.relu(nn.Linear(x.shape[1], 500)(x))

        x = nn.Linear(500, 19)(x)
        # return x
        return F.log_softmax(x, dim=1)  # log probability


def train_func(model, device, train_loader, loss_func, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)  # batch_size * 10
        # loss = F.nll_loss(pred, target)
        loss = loss_func(pred, target)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, loss.item()))


def test_func(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)  # batch_size * 10
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)  # batch_size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))


if __name__ == '__main__':
    train, test, train_y, sub = get_data()
    use_feat = [f for f in train.columns if f not in ['fragment_id', 'time_point', 'behavior_id']]
    train_x, test_x = data_tensor(test, train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proba_t = np.zeros((7500, 19))

    epochs = 100
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 100
    kfold = StratifiedKFold(n_splits=5, random_state=2020)

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        model = Net().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        x_trn, y_trn, x_val, y_val = train_x[trn_idx], train_y[trn_idx], train_x[val_idx], train_y[val_idx]
        train_dataset = data.TensorDataset(x_trn.to(device), y_trn.to(device))
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size)
        test_dataset = data.TensorDataset(x_val.to(device), y_val.to(device))
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size)
        for epoch in range(1, epochs + 1):
            train_func(model, device, train_loader, loss_func, optimizer, epoch)
            test_func(model, device, test_loader)
