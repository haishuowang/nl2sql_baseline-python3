import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from datetime import datetime
from loc_lib.tools_train import train, classifier_dict

data_train = pd.read_csv('sensor_train.csv')
data_test = pd.read_csv('sensor_test.csv')


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


def get_data(run_new=False):
    if not os.path.exists('tmp.csv') or run_new:
        data_test['fragment_id'] += 10000

        data = pd.concat([data_train, data_test], sort=False)
        df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]

        # data['acc_x_diff'] = data['acc_x'].diff()
        # data['acc_y_diff'] = data['acc_y'].diff()
        # data['acc_z_diff'] = data['acc_z'].diff()

        data['acc_xc'] = data['acc_xg'] - data['acc_x']
        data['acc_yc'] = data['acc_yg'] - data['acc_y']
        data['acc_zc'] = data['acc_zg'] - data['acc_z']
        data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5

        data['acc_xc_diff'] = data['acc_xc'].diff()
        data['acc_yc_diff'] = data['acc_yc'].diff()
        data['acc_zc_diff'] = data['acc_zc'].diff()

        data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
        data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
        for f in tqdm([f for f in data.columns if f in ['acc_x', 'acc_y', 'acc_z', 'acc',
                                                        'acc_xg', 'acc_yg', 'acc_zg', 'accg',
                                                        'acc_xc', 'acc_yc', 'acc_zc', 'G',
                                                        # 'acc_x_diff', 'acc_y_diff', 'acc_z_diff',
                                                        # 'acc_xc_diff', 'acc_yc_diff', 'acc_zc_diff',
                                                        ]]):
            # for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
            for stat in ['min', 'max', 'mean', 'median', 'std']:
                df[f + '_' + stat] = data.groupby('fragment_id')[f].agg(stat).values

            df[f + '_' + 'qtl_05'] = data.groupby('fragment_id')[f].quantile(0.05).values
            df[f + '_' + 'qtl_95'] = data.groupby('fragment_id')[f].quantile(0.95).values

            df[f + '_' + 'qtl_20'] = data.groupby('fragment_id')[f].quantile(0.20).values
            df[f + '_' + 'qtl_80'] = data.groupby('fragment_id')[f].quantile(0.80).values

            # df[f + '_' + 'max_min'] = df[f + '_' + 'max'] - df[f + '_' + 'min']
            # df[f + '_' + 'max_mean'] = df[f + '_' + 'max'] - df[f + '_' + 'mean']
            # df[f + '_' + 'mean_min'] = df[f + '_' + 'mean'] - df[f + '_' + 'min']

            def fun(x):
                x = x.rolling(window=5).mean()
                return len(signal.find_peaks(x, distance=3)[0])

            df[f + '_' + 'peaks_num'] = data.groupby('fragment_id')[f].apply(fun).values
        df.to_csv('tmp.csv')
    else:
        df = pd.read_csv('tmp.csv', index_col=0)
    return df


label = 'behavior_id'
df = get_data()

train_df = df[df[label].notna()].reset_index(drop=True)
test_df = df[df[label].isna()].reset_index(drop=True)

drop_feat = []
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
# used_feat = ['acc_xg_peaks_num', 'acc_xg_qtl_80', 'acc_zc_min', 'acc_y_qtl_05', 'G_
# std', 'acc_zg_qtl_05',
#              'acc_yg_qtl_95', 'acc_xg_qtl_95', 'acc_xg_mean', 'acc_x_peaks_num', 'acc_y_qtl_80', 'acc_xg_std',
#              'acc_xg_min', 'acc_x_std', 'accg_mean', 'accg_qtl_05', 'acc_y_qtl_20', 'G_min', 'acc_yc_std',
#              'acc_z_mean', 'acc_xg_qtl_05', 'G_qtl_95', 'G_max', 'acc_zg_min', 'acc_yc_min', 'acc_median', 'G_qtl_20',
#              'acc_xg_qtl_20', 'G_median', 'acc_xc_std', 'G_mean', 'acc_yg_qtl_20', 'accg_qtl_80', 'G_qtl_05',
#              'acc_mean', 'acc_zc_qtl_05', 'acc_zg_qtl_20', 'acc_xc_qtl_80', 'accg_qtl_20', 'G_qtl_80', 'acc_yg_max',
#              'acc_yg_std', 'accg_median', 'acc_y_std', 'acc_y_peaks_num', 'acc_zg_mean', 'acc_yg_qtl_80', 'acc_xc_min',
#              'acc_qtl_20', 'acc_x_mean', 'acc_qtl_05', 'acc_yg_qtl_05', 'acc_y_mean', 'acc_yg_min', 'acc_xg_max',
#              'acc_zg_std', 'acc_yc_max', 'acc_zc_std', 'acc_yc_qtl_95', 'acc_yg_mean']
print(len(used_feat))
print(used_feat)

train_x = train_df[used_feat]
train_y = train_df[label]
test_x = test_df[used_feat]

folds = 5
seed = 2020

model = VotingClassifier(estimators=[('model1', classifier_dict['RandomForestClassifier']()),
                                     ('model2', classifier_dict['XGBClassifier']()),
                                     ('model3', classifier_dict['LGBMClassifier']()),
                                     ],
                         voting='soft')
res_list, pred_y = train(model, folds, train_x, train_y, test_x)

res_df = pd.DataFrame(res_list)
res_sr = res_df.apply(lambda x: x.value_counts().idxmax())
sub = pd.Series(res_sr.values, index=test_df['fragment_id'] - 10000)

score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(train_y, pred_y)) / len(pred_y)
# sub.to_csv(f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_sub{round(score, 5)}.csv', index=False)
