import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

root_path = './'
train = pd.read_csv(root_path + 'sensor_train.csv')
test = pd.read_csv(root_path + 'sensor_test.csv')
sub = pd.read_csv(root_path + '提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()


def fun(data):
    data['acc_xc'] = data['acc_xg'] - data['acc_x']
    data['acc_yc'] = data['acc_yg'] - data['acc_y']
    data['acc_zc'] = data['acc_zg'] - data['acc_z']
    data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5
    data['mod'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** .5
    data['modg'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** .5
    return data


train = fun(train)
test = fun(test)
use_feat = [f for f in train.columns if f not in ['fragment_id', 'time_point', 'behavior_id']]

# train = train[use_feat]
# test = test[use_feat]

x = np.zeros((7292, 60, len(use_feat), 1))
t = np.zeros((7500, 60, len(use_feat), 1))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    x[i, :, :, 0] = resample(tmp[use_feat], 60, np.array(tmp.time_point))[0]

for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i, :, :, 0] = resample(tmp[use_feat], 60, np.array(tmp.time_point))[0]


def Net():
    input = Input(shape=(60, len(use_feat), 1))
    X = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(input)
    print(X.shape)
    X = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    print(X.shape)
    # X = Conv2D(filters=256,
    #            kernel_size=(3, 3),
    #            activation='relu',
    #            padding='same')(X)

    X = MaxPooling2D()(X)
    print(X.shape)
    X = Dropout(0.2)(X)
    print(X.shape)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    print(X.shape)
    X = Dropout(0.3)(X)
    print(X.shape)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    print(X.shape)
    X = GlobalMaxPooling2D()(X)
    print(X.shape)
    X = Dropout(0.5)(X)
    print(X.shape)
    X = Dense(19, activation='softmax')(X)
    return Model([input], X)

(None, 60, 12, 64)
(None, 60, 12, 128)
(None, 30, 6, 128)
(None, 30, 6, 128)
(None, 30, 6, 256)
(None, 30, 6, 256)
(None, 30, 6, 512)
(None, 512)
(None, 512)
kfold = StratifiedKFold(5, random_state=2020, shuffle=True)

proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    y_ = to_categorical(y, num_classes=19)
    model = Net()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=8)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=0,
                                   mode='max',
                                   patience=18)
    checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    model.fit(x[xx], y_[xx],
              epochs=500,
              batch_size=256,
              verbose=1,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'fold{fold}.h5')
    proba_t += model.predict(t, verbose=0, batch_size=1024) / 5.
sub.behavior_id = np.argmax(proba_t, axis=1)
from datetime import datetime

sub.to_csv(f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_submit_cnn4.1_C.csv', index=False)
