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
