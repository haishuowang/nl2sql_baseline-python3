import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# 字段1	fragment_id	int	行为片段id
# 字段2	time_point	int	采集时间点（ms）
# 字段3	acc_x	float	不含重力加速度的x轴分量（m/s^2）
# 字段4	acc_y	float	不含重力加速度的y轴分量（m/s^2）
# 字段5	acc_z	float	不含重力加速度的z轴分量（m/s^2）
# 字段6	acc_xg	float	包含重力加速度的x轴分量（m/s^2）
# 字段7	acc_yg	float	包含重力加速度的y轴分量（m/s^2）
# 字段8	acc_zg	float	包含重力加速度的z轴分量（m/s^2）
# 字段9	behavior_id	int	编号的行为id

def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
               4: 'D_4', 5: 'A_5', 6: 'B_1', 7: 'B_5',
               8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
               12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
               16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred:  # 编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]:  # 编码仅字母部分相同得分1.0/7
        return 1.0 / 7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]:  # 编码仅数字部分相同得分1.0/3
        return 1.0 / 3
    else:
        return 0.0


def add_feature(data):
    data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
    data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5

    data['acc_x_diff'] = data['acc_x'].diff()
    data['acc_y_diff'] = data['acc_y'].diff()
    data['acc_z_diff'] = data['acc_z'].diff()
    data['acc_diff'] = data['acc'].diff()

    data['acc_xc'] = data['acc_xg'] - data['acc_x']
    data['acc_yc'] = data['acc_yg'] - data['acc_y']
    data['acc_zc'] = data['acc_zg'] - data['acc_z']
    data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5

    data['acc_xc_diff'] = data['acc_xc'].diff()
    data['acc_yc_diff'] = data['acc_yc'].diff()
    data['acc_zc_diff'] = data['acc_zc'].diff()
    data['g_diff'] = data['G'].diff()
    return data


data_train = pd.read_csv('sensor_train.csv')
data_test = pd.read_csv('sensor_test.csv')

add_feature(data_train)
add_feature(data_test)
# data_train.drop_duplicates(subset=)
data_all = pd.concat([data_train, data_test], sort=False)


def deal_out_data(column, data_train, data_test, sigma=3):
    train_std = data_train[column].std()
    train_mean = data_train[column].mean()
    a = sigma * train_std
    data_train


for label, part_df in data_train.groupby(by='behavior_id'):
    part_df = part_df.sort_values(by='time_point')
    plt.figure(figsize=[20, 10])
    plt.title='behavior_id'
    print('____________________________________________')
    for i, col in enumerate(['acc_x', 'acc_y', 'acc_z', 'acc',
                             'acc_xg', 'acc_yg', 'acc_zg', 'accg',
                             'acc_xc', 'acc_yc', 'acc_zc', 'G'
                               ]):
        print(i, col, sep='   ')
        print(part_df[col].describe(), sep='   ')

        ax = plt.subplot(3, 4, i+1)
        plt.plot(part_df['time_point'].values, part_df[col].values)
    # plt.show()
    plt.savefig(f'behavior_id {label}.png')
    plt.close()

# for label, part_df in data_train.groupby(by='behavior_id'):
#     part_df = part_df.sort_values(by='time_point')
#     # a = 0
#     fragment_id_list = list(set(part_df['fragment_id']))
#     import random
#
#     select_list = random.sample(fragment_id_list, 10)
#     for fragment_id in select_list:
#         fragment_id_df = part_df[part_df['fragment_id']==fragment_id]
#         plt.figure(figsize=[20, 10])
#         plt.suptitle = label
#         print('____________________________________________')
#         for i, col in enumerate(['acc_x', 'acc_y', 'acc_z', 'acc',
#                                  'acc_xg', 'acc_yg', 'acc_zg', 'accg',
#                                  'acc_xc', 'acc_yc', 'acc_zc', 'G',
#                                  'acc_x_diff', 'acc_y_diff', 'acc_z_diff', 'acc_diff',
#                                  'acc_xc_diff', 'acc_yc_diff', 'acc_zc_diff', 'g_diff',
#                                    ]):
#             # print(i, col, sep='   ')
#             # print(fragment_id_df[col].describe(), sep='   ')
#
#             ax = plt.subplot(5, 4, i+1)
#             ax.plot(fragment_id_df['time_point'].values, fragment_id_df[col].values)
#         # plt.show()
#         plt.savefig(f'behavior_id {label}_{fragment_id}.png')
#         plt.close()

# for x in data_test.columns:
#     g = sns.kdeplot(data_train[x], color="Red", shade=True)
#     g = sns.kdeplot(data_test[x], ax=g, color="Blue", shade=True)
#     g.set_xlabel(x)
#     g.set_ylabel("Frequency")
#     g = g.legend(["train", "test"])
#     plt.show()

# fragment_id_df = data_train[data_train['fragment_id']==6664]
# plt.figure(figsize=[20, 10])
# print('____________________________________________')
# for i, col in enumerate(['acc_x', 'acc_y', 'acc_z', 'acc',
#                          'acc_xg', 'acc_yg', 'acc_zg', 'accg',
#                          'acc_xc', 'acc_yc', 'acc_zc', 'G',
#                          'acc_x_diff', 'acc_y_diff', 'acc_z_diff', 'acc_diff',
#                          'acc_xc_diff', 'acc_yc_diff', 'acc_zc_diff', 'g_diff',
#                            ]):
#     # print(i, col, sep='   ')
#     # print(fragment_id_df[col].describe(), sep='   ')
#
#     ax = plt.subplot(5, 4, i+1)
#     ax.plot(fragment_id_df['time_point'].values, fragment_id_df[col].rolling(window=5).mean().values)
# plt.show()
#
#
# from scipy import signal
# fragment_id_df = data_train[data_train['fragment_id']==6664]
# print('____________________________________________')
# for i, col in enumerate(['acc_x', 'acc_y', 'acc_z', 'acc',
#                          'acc_xg', 'acc_yg', 'acc_zg', 'accg',
#                          'acc_xc', 'acc_yc', 'acc_zc', 'G',
#                          'acc_x_diff', 'acc_y_diff', 'acc_z_diff', 'acc_diff',
#                          'acc_xc_diff', 'acc_yc_diff', 'acc_zc_diff', 'g_diff',
#                            ]):
#     print(i, col)
#     a = fragment_id_df[col].rolling(window=5).mean().values
#     a = fragment_id_df[col].values
#     print(signal.find_peaks(a, distance=3))
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.linspace(0, 1, 1400)

# 设置需要采样的信号，频率分量有200，400和600
y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)

fft_y = fft(y)  # 快速傅里叶变换

N = 1400
x = np.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度
normalization_y = abs_y / N  # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

plt.subplot(231)
plt.plot(x, y)
plt.title('原始波形')

plt.subplot(232)
plt.plot(x, fft_y, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

plt.subplot(233)
plt.plot(x, abs_y, 'r')
plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')

plt.subplot(234)
plt.plot(x, angle_y, 'violet')
plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')

plt.subplot(235)
plt.plot(x, normalization_y, 'g')
plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

plt.subplot(236)
plt.plot(half_x, normalization_half_y, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
