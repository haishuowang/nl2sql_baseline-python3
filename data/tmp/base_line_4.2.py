import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from scipy.signal import resample

pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)

root_path = './'
train = pd.read_csv(root_path + 'sensor_train.csv')
test = pd.read_csv(root_path + 'sensor_test.csv')
sub = pd.read_csv(root_path + '提交结果示例.csv')
train_y = train.groupby('fragment_id')['behavior_id'].min()
train_y = torch.from_numpy(train_y.values).long()

def fun(data):
    data['acc_xc'] = data['acc_xg'] - data['acc_x']
    data['acc_yc'] = data['acc_yg'] - data['acc_y']
    data['acc_zc'] = data['acc_zg'] - data['acc_z']
    data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5
    data['mod'] = (data.acc_x ** 2 + data.acc_y ** 2 + train.acc_z ** 2) ** .5
    data['modg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + train.acc_zg ** 2) ** .5
    return data


train = fun(train)
test = fun(test)
use_feat = [f for f in train.columns if f not in ['fragment_id', 'time_point', 'behavior_id']]

# train = train[use_feat]
# test = test[use_feat]

train_x = torch.zeros((7292, 1, 60, len(use_feat)))
test_x = torch.zeros((7500, 1, 60, len(use_feat)))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    train_x[i, 0, :, :] = torch.from_numpy(resample(tmp[use_feat], 60, np.array(tmp.time_point))[0])

for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    test_x[i, 0, :, :] = torch.from_numpy(resample(tmp[use_feat], 60, np.array(tmp.time_point))[0])


# class LeNet(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
#         self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(in_features=336, out_features=256)
#         self.fc2 = nn.Linear(in_features=256, out_features=19)
#
#     def forward(self, x):
#         x = nn.Conv1d(in_channels=60, out_channels=64, kernel_size=3, padding=1)(x)
#         x = F.relu(x)
#
#         x = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)(x)
#         x = F.relu(x)
#
#         x = nn.MaxPool2d()(x)
#         x = nn.Dropout(0.2)(x)
#         x = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)(x)
#         x = F.relu(x)
#         x = nn.Dropout(0.3)(x)
#
#         x = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)(x)
#         x = F.relu(x)
#         x = nn.MaxPool2d()(x)
#         x = nn.Dropout(0.5)(x)
#         x = F.softmax(input=x, dim=1)
#
#         # x = self.pool(x)
#         # x = self.conv2(x)
#         # x = F.relu(x)
#         # x = x.view(-1, 336)
#         # x = self.fc1(x)
#         # x = F.relu(x)
#         # x = self.fc2(x)
#         # x = F.softmax(input=x, dim=1)
#         return x
#
#     def num_f(self, x):
#         size = x.size()[1:]
#         ret = 1
#         for i in size:
#             ret *= i
#         return ret


# kfold = StratifiedKFold(5, random_state=2020, shuffle=True)

# for fold, (xx, yy) in enumerate(kfold.split(train_x, train_y)):
#     print(xx.shape, yy.shape)
#     x_trn = train_x[xx]

#     # y_ = to_categorical(y, num_classes=19)
#     model = LeNet()
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(),
#                   metrics=['acc'])
#     plateau = ReduceLROnPlateau(monitor="val_acc",
#                                 verbose=1,
#                                 mode='max',
#                                 factor=0.5,
#                                 patience=8)
#     early_stopping = EarlyStopping(monitor='val_acc',
#                                    verbose=0,
#                                    mode='max',
#                                    patience=18)
#     checkpoint = ModelCheckpoint(f'fold{fold}.h5',
#                                  monitor='val_acc',
#                                  verbose=0,
#                                  mode='max',
#                                  save_best_only=True)
#     model.fit(x[xx], y_[xx],
#               epochs=500,
#               batch_size=256,
#               verbose=1,
#               shuffle=True,
#               validation_data=(x[yy], y_[yy]),
#               callbacks=[plateau, early_stopping, checkpoint])
#     model.load_weights(f'fold{fold}.h5')
#     proba_t += model.predict(t, verbose=0, batch_size=1024) / 5.
# sub.behavior_id = np.argmax(proba_t, axis=1)
# from datetime import datetime
#
# sub.to_csv(f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_submit_cnn4.1_C.csv', index=False)
model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                      nn.ReLU(),

                      nn.MaxPool2d(2),
                      nn.Dropout(0.2),

                      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                      nn.ReLU(),

                      nn.Dropout(0.3),

                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d((2, 2)),
                      nn.Dropout(0.5),
                      nn.Linear(in_features=512 *7 * 7, out_features=256),
                      nn.Linear(in_features=258, out_features=19),
                      nn.Softmax(div=1)
                      )

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
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
        x = nn.Linear(in_features=512 * 7 * 7, out_features=256)(x)
        x = nn.Linear(in_features=258, out_features=19)(x)
        x = nn.Softmax(dim=1)(x)
        return x



proba_t = np.zeros((7500, 19))

certion = nn.CrossEntropyLoss()
epochs = 100
learning_rate = 0.001
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
    x_trn, y_trn, x_val, y_val = train_x[trn_idx], train_y[trn_idx], train_x[val_idx], train_y[val_idx]
    max_acc = 0
    early_stop = 0
    Adam = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        # model.train()
        Adam.zero_grad()
        exit()
        output = model(x_trn)
        loss = certion(output, y_trn)
        loss.backward()
        Adam.step()
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
