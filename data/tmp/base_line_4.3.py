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
from multiprocessing import Pool

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

    # data['acc_xg_fft_abs'] = np.nan
    # data['acc_yg_fft_abs'] = np.nan
    # data['acc_zg_fft_abs'] = np.nan
    # data['acc_x_fft_abs'] = np.nan
    # data['acc_y_fft_abs'] = np.nan
    # data['acc_z_fft_abs'] = np.nan
    # data['acc_xg_fft_angle'] = np.nan
    # data['acc_yg_fft_angle'] = np.nan
    # data['acc_zg_fft_angle'] = np.nan
    # data['acc_x_fft_angle'] = np.nan
    # data['acc_y_fft_angle'] = np.nan
    # data['acc_z_fft_angle'] = np.nan

    # for id, part_data in tqdm(data.groupby('fragment_id')):
    #     part_data_len = len(part_data)
    #     acc_xg_fft = np.fft.fft(part_data['acc_xg'])
    #     acc_yg_fft = np.fft.fft(part_data['acc_yg'])
    #     acc_zg_fft = np.fft.fft(part_data['acc_zg'])
    #     acc_x_fft = np.fft.fft(part_data['acc_x'])
    #     acc_y_fft = np.fft.fft(part_data['acc_y'])
    #     acc_z_fft = np.fft.fft(part_data['acc_z'])
    #
    #     data['acc_xg_fft_abs'].loc[part_data.index] = np.abs(acc_xg_fft) / part_data_len
    #     data['acc_yg_fft_abs'].loc[part_data.index] = np.abs(acc_yg_fft) / part_data_len
    #     data['acc_zg_fft_abs'].loc[part_data.index] = np.abs(acc_zg_fft) / part_data_len
    #     data['acc_x_fft_abs'].loc[part_data.index] = np.abs(acc_x_fft) / part_data_len
    #     data['acc_y_fft_abs'].loc[part_data.index] = np.abs(acc_y_fft) / part_data_len
    #     data['acc_z_fft_abs'].loc[part_data.index] = np.abs(acc_z_fft) / part_data_len
    #
    #     data['acc_xg_fft_angle'].loc[part_data.index] = np.angle(acc_xg_fft)
    #     data['acc_yg_fft_angle'].loc[part_data.index] = np.angle(acc_yg_fft)
    #     data['acc_zg_fft_angle'].loc[part_data.index] = np.angle(acc_zg_fft)
    #     data['acc_x_fft_angle'].loc[part_data.index] = np.angle(acc_x_fft)
    #     data['acc_y_fft_angle'].loc[part_data.index] = np.angle(acc_y_fft)
    #     data['acc_z_fft_angle'].loc[part_data.index] = np.angle(acc_z_fft)

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

    def part_data_tensor(train, i):
        tmp = train[train.fragment_id == i][:60]
        # part_data_len = len(tmp)
        # acc_xg_fft = np.fft.fft(tmp['acc_xg'])
        # acc_yg_fft = np.fft.fft(tmp['acc_yg'])
        # acc_zg_fft = np.fft.fft(tmp['acc_zg'])
        # acc_x_fft = np.fft.fft(tmp['acc_x'])
        # acc_y_fft = np.fft.fft(tmp['acc_y'])
        # acc_z_fft = np.fft.fft(tmp['acc_z'])
        #
        # tmp['acc_xg_fft_abs'] = np.abs(acc_xg_fft) / part_data_len
        # tmp['acc_yg_fft_abs'] = np.abs(acc_yg_fft) / part_data_len
        # tmp['acc_zg_fft_abs'] = np.abs(acc_zg_fft) / part_data_len
        # tmp['acc_x_fft_abs'] = np.abs(acc_x_fft) / part_data_len
        # tmp['acc_y_fft_abs'] = np.abs(acc_y_fft) / part_data_len
        # tmp['acc_z_fft_abs'] = np.abs(acc_z_fft) / part_data_len
        # tmp['acc_xg_fft_angle'] = np.angle(acc_xg_fft)
        # tmp['acc_yg_fft_angle'] = np.angle(acc_yg_fft)
        # tmp['acc_zg_fft_angle'] = np.angle(acc_zg_fft)
        # tmp['acc_x_fft_angle'] = np.angle(acc_x_fft)
        # tmp['acc_y_fft_angle'] = np.angle(acc_y_fft)
        # tmp['acc_z_fft_angle'] = np.angle(acc_z_fft)

        data, label = resample(tmp[use_feat], 60, np.array(tmp.time_point))
        return data

    for i in tqdm(range(7292)):
        data = part_data_tensor(train, i)
        train_x[i, 0, :, :] = torch.from_numpy(data)

    for i in tqdm(range(7500)):
        data = part_data_tensor(test, i)
        test_x[i, 0, :, :] = torch.from_numpy(data)
    return train_x, test_x


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.max_pool1 = nn.MaxPool2d(2)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(1)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.3)
        self.dropout6 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(in_features=512, out_features=19)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.max_pool(x)
        x = self.dropout3(x)

        x = self.relu(self.conv3(x))
        x = self.dropout4(x)
        x = self.relu(self.conv4(x))
        x = self.dropout5(x)
        x = self.max_pool1(x)
        x = self.dropout6(x)
        x = self.adaptive_max_pool(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


class PaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=[5, 1], padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[5, 1], padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=[3, 1], padding=0)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.avg_pool1 = nn.AvgPool2d([2, 1])
        self.avg_pool2 = nn.AvgPool2d([2, 1])
        self.avg_pool3 = nn.AvgPool2d(2)
        # self.max_pool1 = nn.MaxPool2d([2, 1])
        # self.max_pool2 = nn.MaxPool2d([2, 1])
        # self.max_pool3 = nn.MaxPool2d(2)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d([1, 11])
        self.bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.fc1 = nn.Linear(in_features=256 * 11, out_features=19)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.dropout(x, p=0.2)
        x = self.avg_pool1(x)
        x = F.tanh(self.conv2(x))
        x = F.dropout(x, p=0.2)
        x = self.avg_pool2(x)
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = F.dropout(x, p=0.4)
        x = self.avg_pool3(x)
        x = F.tanh(self.bn(x))
        x = self.adaptive_max_pool(x)
        x = x.view(-1, 256 * 11)
        x = F.dropout(x, p=0.5)
        # print(x.shape)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train_func(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.
    train_correct = 0.
    for idx, (part_train, target) in enumerate(train_loader):
        data, target = part_train.to(device), target.to(device)

        output = model(part_train)  # batch_size * 10
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1)
        part_correct = pred.eq(target.view_as(pred)).sum().item()
        acc = part_correct / len(target)
        train_correct += part_correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # if idx % 100 == 0:
        print("Train Epoch: {}, iteration: {}, Train loss: {}, Acc={}".format(
            epoch, idx, round(loss.item(), 4), round(acc * 100, 2)))
    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_loader.dataset) * 100
    return train_acc, train_loss


def test_func(model, device, test_loader):
    model.eval()
    test_loss = 0.
    test_correct = 0.
    with torch.no_grad():
        for idx, (part_test, target) in enumerate(test_loader):
            part_test, target = part_test.to(device), target.to(device)

            output = model(part_test)  # batch_size * 10
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)  # batch_size * 1
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset) * 100.
    print('*******************************')
    print("test loss: {}, val_acc: {}".format(test_loss, test_acc))
    print('*******************************')
    return test_acc, test_loss


def predict_fun(model, device, pred_loader):
    res_list = []
    with torch.no_grad():
        for idx, (part_pred,) in enumerate(pred_loader):
            part_pred = part_pred.to(device)

            output = model(part_pred)
            pred = output.argmax(dim=1)
            if device.type == 'cuda':
                pred = pred.cpu()
            res_list.append(list(pred))
    return res_list


def get_train_model(date_begin, device):
    lr = 0.001
    batch_size = 128
    kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        x_trn, y_trn, x_val, y_val = train_x[trn_idx], train_y[trn_idx], train_x[val_idx], train_y[val_idx]
        train_dataset = data.TensorDataset(x_trn.to(device), y_trn.to(device))
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = data.TensorDataset(x_val.to(device), y_val.to(device))
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model = MyNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = 150
        early_stop_init = 0
        early_stop_step = 20
        max_acc = 0

        change_lr_con = True
        change_lr_rate = 0.1
        change_lr_init = 0
        change_lr_step = 16

        train_loss_list = []
        test_loss_list = []

        train_acc_list = []
        test_acc_list = []

        for epoch in range(num_epochs):
            train_acc, train_loss = train_func(model, device, train_loader, optimizer, epoch)
            test_acc, test_loss = test_func(model, device, test_loader)
            print(train_acc, train_loss)
            print(test_acc, test_loss)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print(f'max_val={max_acc}， early_stop_init={early_stop_init}, '
                  f'change_lr_init={change_lr_init}')
            if max_acc < test_acc:
                max_acc = test_acc
                # early_stop_init = 0
                # change_lr_init = 0
                torch.save(model.state_dict(), f"{date_begin}_mnist_cnn{fold}.pt")
            elif max_acc > 70:
                if change_lr_con:
                    change_lr_init += 1
                    if change_lr_init >= change_lr_step:
                        lr = lr * change_lr_rate
                        change_lr_con = False
            #     else:
            #         early_stop_init += 1
            #         if early_stop_init >= early_stop_step:
            #             break
        plt.figure(figsize=[12, 10])
        plt.plot(train_loss_list)
        plt.plot(test_loss_list)
        plt.savefig(f'{date_begin}_loss{fold}.png')
        plt.figure(figsize=[12, 10])
        plt.plot(train_acc_list)
        plt.plot(test_acc_list)
        plt.savefig(f'{date_begin}_acc{fold}.png')


def get_predict_file(pred_loader, date_begin, device):
    proba_t = np.zeros((7500, 19))
    for i in range(5):
        model = MyNet().to(device)
        model.load_state_dict(torch.load(f"{date_begin}_mnist_cnn{i}.pt"))

        def predict_fun(model, device, pred_loader):
            res_list = []
            with torch.no_grad():
                for idx, (part_pred,) in enumerate(pred_loader):
                    part_pred = part_pred.to(device)

                    output = model(part_pred)
                    # pred = output.argmax(dim=1)
                    if device.type == 'cuda':
                        output = output.cpu()
                    res_list.append(output)
            result_tensor = torch.cat(res_list)
            return result_tensor

        result_tensor = predict_fun(model, device, pred_loader)
        result_array = np.array(result_tensor)
        proba_t += result_array / 5

    pred_y = np.argmax(proba_t, axis=1)

    sub = pd.read_csv('提交结果示例.csv')
    sub.behavior_id = pred_y
    sub.to_csv(f'{date_begin}_submit_cnn4.3.csv', index=False)


if __name__ == '__main__':
    from datetime import datetime

    # train, test, train_y, sub = get_data()
    use_feat = ['acc_x', 'acc_y', 'acc_z', 'mod',
                'acc_xg', 'acc_yg', 'acc_zg', 'modg',
                'acc_xc', 'acc_yc', 'acc_zc', 'G',
                ]
    # add_feat = [
    #     'acc_xg_fft_abs', 'acc_yg_fft_abs', 'acc_zg_fft_abs',
    #     'acc_x_fft_abs', 'acc_y_fft_abs', 'acc_z_fft_abs',
    #     'acc_xg_fft_angle', 'acc_yg_fft_angle', 'acc_zg_fft_angle',
    #     'acc_x_fft_angle', 'acc_y_fft_angle', 'acc_z_fft_angle'
    # ]
    # use_feat = use_feat + add_feat
    # train_x, test_x = data_tensor(test, train)
    # pd.to_pickle(train_x, 'train_x.pkl')
    # pd.to_pickle(test_x, 'test_x.pkl')
    # pd.to_pickle(train_y, 'train_y.pkl')

    train_x = pd.read_pickle('train_x.pkl')
    test_x = pd.read_pickle('test_x.pkl')
    train_y = pd.read_pickle('train_y.pkl')

    date_begin = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    get_train_model(date_begin, device)

    print(date_begin, datetime.now())
    pred_dataset = data.TensorDataset(test_x.to(device))
    pred_loader = data.DataLoader(pred_dataset, batch_size=512)
    get_predict_file(pred_loader, date_begin, device)
