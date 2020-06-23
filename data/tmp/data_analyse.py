import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
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
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0

data_test = pd.read_csv('sensor_test.csv')
data_train = pd.read_csv('sensor_train.csv')

def deal_out_data(column, data_train, data_test, sigma=3):
    train_std = data_train[column].std()
    train_mean = data_train[column].mean()
    a = sigma * train_std
    data_train


for x in data_test.columns:
    g = sns.kdeplot(data_train[x], color="Red", shade=True)
    g = sns.kdeplot(data_test[x], ax=g, color="Blue", shade=True)
    g.set_xlabel(x)
    g.set_ylabel("Frequency")
    g = g.legend(["train", "test"])
    plt.show()

KNeighborsRegressor().fit()