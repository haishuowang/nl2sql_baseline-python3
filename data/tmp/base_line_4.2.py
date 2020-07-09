import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
import torch.optim as optim
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import gc
import os
from tqdm import tqdm


pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

data_path = ''
data_train = pd.read_csv(data_path + 'sensor_train.csv')
data_test = pd.read_csv(data_path + 'sensor_test.csv')
data_test['fragment_id'] += 10000
label = 'behavior_id'

data = pd.concat([data_train, data_test], sort=False)

df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]

a = data.drop_duplicates('fragment_id')
b = data.drop_duplicates(subset=['fragment_id'])

data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5

for f in tqdm([f for f in data.columns if 'acc' in f]):
    for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
        df[f + '_' + stat] = data.groupby('fragment_id')[f].agg(stat).values

train_df = df[df[label].isna() == False].reset_index(drop=True)
test_df = df[df[label].isna() == True].reset_index(drop=True)
train_df

drop_feat = []
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]

train_x = train_df[used_feat]
train_y = train_df[label]
test_x = test_df[used_feat]


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=0)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=336, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=19)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 336)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(input=x, dim=1)
        return x

    def num_f(self, x):
        size = x.size()[1:]
        ret = 1
        for i in size:
            ret *= i
        return ret


model = LeNet()

certion = nn.CrossEntropyLoss()
epochs = 5000
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
    x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[
        val_idx]
    # x_trn = torch.Tensor(np.array(x_trn)).unsqueeze(1)
    # x_val = torch.Tensor(np.array(x_val)).unsqueeze(1)
    # y_trn = torch.Tensor(np.array(y_trn, dtype='int')).long()
    # y_val = torch.Tensor(np.array(y_val, dtype='int')).long()

    x_trn = torch.from_numpy(np.array(x_trn)).unsqueeze(1)
    x_val = torch.from_numpy(np.array(x_val)).unsqueeze(1)
    y_trn = torch.from_numpy(np.array(y_trn, dtype='int')).long()
    y_val = torch.from_numpy(np.array(y_val, dtype='int')).long()

    # print(x_trn.shape)
    max_acc = 0
    early_stop = 0
    SGD = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, epochs + 1):
        model.train()
        SGD.zero_grad()
        output = model(x_trn)
        loss = certion(output, y_trn)
        loss.backward()
        SGD.step()

        model.eval()
        output_ = model(x_val)
        output_ = torch.argmax(output_, dim=1)
        cnt = sum(output_ == y_val).item()
        len_dataset = len(y_val)
        acc = cnt / len_dataset * 100
        print("Epoch{}: Accu in val_set is {}.".format(epoch, acc))

        if max_acc < acc:
            max_acc = acc
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 50:
                break

