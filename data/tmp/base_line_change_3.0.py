import numpy as np
import pandas as pd
import os
import warnings
import sys

sys.path.append('/home/haishuowang/PycharmProjects/nl2sql_baseline-python3')
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from datetime import datetime
from loc_lib.tools_train import train, classifier_dict
from loc_lib.tools_analyse import kdeplot

data_train = pd.read_csv('sensor_train.csv')
data_test = pd.read_csv('sensor_test.csv')
data_test['fragment_id'] += 10000

data = pd.concat([data_train, data_test], sort=False)


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


def get_data(run_new=False, add=False):
    if add:
        df = pd.read_csv('tmp.csv', index_col=0)

        # data['acc_x_diff'] = data['acc_x'].diff()
        # data['acc_y_diff'] = data['acc_y'].diff()
        # data['acc_z_diff'] = data['acc_z'].diff()

        data['acc_xc'] = data['acc_xg'] - data['acc_x']
        data['acc_yc'] = data['acc_yg'] - data['acc_y']
        data['acc_zc'] = data['acc_zg'] - data['acc_z']
        data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5

        data['delta_xy'] = np.arcsin(data['acc_zc'] / data['G'])
        data['delta_xz'] = np.arcsin(data['acc_yc'] / data['G'])
        data['delta_yz'] = np.arcsin(data['acc_xc'] / data['G'])

        data['acc_xc_diff'] = data['acc_xc'].diff()
        data['acc_yc_diff'] = data['acc_yc'].diff()
        data['acc_zc_diff'] = data['acc_zc'].diff()

        data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
        data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
        for f in tqdm([f for f in data.columns if f in [
            'acc_x', 'acc_y', 'acc_z', 'acc',
            'acc_xg', 'acc_yg', 'acc_zg', 'accg',
            'acc_xc', 'acc_yc', 'acc_zc', 'G',
            'delta_xy', 'delta_xz', 'delta_yz',
            'acc_x_diff', 'acc_y_diff', 'acc_z_diff',
            'acc_xc_diff', 'acc_yc_diff', 'acc_zc_diff',
        ]]):
            # for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
            #     # for stat in ['min', 'max', 'mean', 'median', 'std']:
            #     df[f + '|' + stat] = data.groupby('fragment_id')[f].agg(stat).values
            # def fun(x):
            #     x = x.rolling(window=5).mean()
            #     return len(signal.find_peaks(x, distance=5)[0])
            # df[f + '|' + 'peaks_num'] = data.groupby('fragment_id')[f].apply(fun).values
            df[f + '|' + 'qtl_05'] = data.groupby('fragment_id')[f].quantile(0.05).values
            df[f + '|' + 'qtl_95'] = data.groupby('fragment_id')[f].quantile(0.95).values

            df[f + '|' + 'qtl_20'] = data.groupby('fragment_id')[f].quantile(0.20).values
            df[f + '|' + 'qtl_80'] = data.groupby('fragment_id')[f].quantile(0.80).values

            df[f + '|' + 'qtl_02'] = data.groupby('fragment_id')[f].quantile(0.02).values
            df[f + '|' + 'qtl_98'] = data.groupby('fragment_id')[f].quantile(0.98).values

            df[f + '|' + 'max_min'] = df[f + '|' + 'max'] - df[f + '|' + 'min'].values
            df[f + '|' + 'max_mean'] = df[f + '|' + 'max'] - df[f + '|' + 'mean'].values
            df[f + '|' + 'mean_min'] = df[f + '|' + 'mean'] - df[f + '|' + 'min'].values

            df[f + '|' + 'max_median'] = df[f + '|' + 'max'] - df[f + '|' + 'median'].values
            df[f + '|' + 'median_min'] = df[f + '|' + 'median'] - df[f + '|' + 'min'].values


            df[f + '|' + 'power_rate'] = data.groupby('fragment_id')[f].apply(lambda x: (x.abs() > 0.1).sum() / len(x)).values

            df[f + '|' + '95_05'] = df[f + '|' + 'qtl_95'] - df[f + '|' + 'qtl_05'].values
            df[f + '|' + '80_20'] = df[f + '|' + 'qtl_80'] - df[f + '|' + 'qtl_20'].values
            df[f + '|' + '98_02'] = df[f + '|' + 'qtl_95'] - df[f + '|' + 'qtl_05'].values
        df.to_csv('tmp.csv')
        return df

    elif not os.path.exists('tmp.csv') or run_new:
        df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]

        # data['acc_x_diff'] = data['acc_x'].diff()
        # data['acc_y_diff'] = data['acc_y'].diff()
        # data['acc_z_diff'] = data['acc_z'].diff()

        data['acc_xc'] = data['acc_xg'] - data['acc_x']
        data['acc_yc'] = data['acc_yg'] - data['acc_y']
        data['acc_zc'] = data['acc_zg'] - data['acc_z']
        data['G'] = (data['acc_xc'] ** 2 + data['acc_yc'] ** 2 + data['acc_zc'] ** 2) ** 0.5

        data['delta_xy'] = np.arcsin(data['acc_zc'] / data['G'])
        data['delta_xz'] = np.arcsin(data['acc_yc'] / data['G'])
        data['delta_yz'] = np.arcsin(data['acc_xc'] / data['G'])

        data['acc_xc_diff'] = data['acc_xc'].diff()
        data['acc_yc_diff'] = data['acc_yc'].diff()
        data['acc_zc_diff'] = data['acc_zc'].diff()

        data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
        data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
        for f in tqdm([f for f in data.columns if f in [
            'acc_x', 'acc_y', 'acc_z', 'acc',
            'acc_xg', 'acc_yg', 'acc_zg', 'accg',
            'acc_xc', 'acc_yc', 'acc_zc', 'G',
            'delta_xy', 'delta_xz', 'delta_yz',
            'acc_x_diff', 'acc_y_diff', 'acc_z_diff',
            'acc_xc_diff', 'acc_yc_diff', 'acc_zc_diff',
        ]]):
            for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
                # for stat in ['min', 'max', 'mean', 'median', 'std']:
                df[f + '|' + stat] = data.groupby('fragment_id')[f].agg(stat).values

            df[f + '|' + 'qtl_05'] = data.groupby('fragment_id')[f].quantile(0.05).values
            df[f + '|' + 'qtl_95'] = data.groupby('fragment_id')[f].quantile(0.95).values

            df[f + '|' + 'qtl_20'] = data.groupby('fragment_id')[f].quantile(0.20).values
            df[f + '|' + 'qtl_80'] = data.groupby('fragment_id')[f].quantile(0.80).values

            df[f + '|' + '95_05'] = df[f + '|' + 'qtl_95'] - df[f + '|' + 'qtl_05'].values
            df[f + '|' + '80_20'] = df[f + '|' + 'qtl_80'] - df[f + '|' + 'qtl_20'].values

            df[f + '|' + 'max_min'] = df[f + '|' + 'max'] - df[f + '|' + 'min'].values
            df[f + '|' + 'max_mean'] = df[f + '|' + 'max'] - df[f + '|' + 'mean'].values
            df[f + '|' + 'mean_min'] = df[f + '|' + 'mean'] - df[f + '|' + 'min'].values

            df[f + '|' + 'max_median'] = df[f + '|' + 'max'] - df[f + '|' + 'median'].values
            df[f + '|' + 'median_min'] = df[f + '|' + 'median'] - df[f + '|' + 'min'].values

            def fun(x):
                x = x.rolling(window=5).mean()
                return len(signal.find_peaks(x, distance=3)[0])

            def fun_2(x):
                sigma = 3
                std = x.std()
                mean = x.mean()
                upper = mean + sigma * std
                lower = mean - sigma * std
                return len(x[(x > upper) & (x < lower)])

            df[f + '|' + 'out_num'] = data.groupby('fragment_id')[f].apply(fun_2).values
            df[f + '|' + 'peaks_num'] = data.groupby('fragment_id')[f].apply(fun).values
        df.to_csv('tmp.csv')
    else:
        df = pd.read_csv('tmp.csv', index_col=0)
    return df


label = 'behavior_id'
df = get_data(add=True)
# df = get_data()
df['move_range'] = (df['acc_xc|max_min'] + df['acc_yc|max_min'] + df['acc_zc|max_min'])
train_df = df[df[label].notna()].reset_index(drop=True)
test_df = df[df[label].isna()].reset_index(drop=True)

drop_feat = []
# used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]

# used_feat = ['accg|qtl_20', 'acc_yc|std', 'acc_yg|qtl_95', 'G|qtl_80', 'acc_y|mean', 'acc_xg|std', 'acc_zc|min',
#              'acc_xg|qtl_05', 'acc_y|qtl_20', 'acc_zc|std', 'accg|qtl_05', 'acc_xg|mean', 'acc_yg|min', 'acc_x|std',
#              'acc|median', 'acc_yg|std', 'acc|qtl_20', 'acc_yg|mean', 'acc_z|mean', 'acc_y|std', 'acc_x|mean',
#              'acc_yc|max', 'acc_zg|mean', 'accg|mean', 'acc_xg|qtl_95', 'acc_yg|qtl_05', 'G|std', 'G|qtl_20',
#              'G|median']

# used_feat = ['G|mean',
#              'acc_yg|qtl_95',
#              'acc|qtl_20',
#              'acc_y|qtl_20',
#              'acc_x|mean',
#              'acc_yc|std',
#              'acc|median',
#              'G|median',
#              'acc_yc|qtl_95',
#              'acc_xg|max',
#              'acc_y|std',
#              'acc_yg|qtl_05',
#              'G|qtl95',
#              'acc_xc_diff|skew',
#              'acc_zc|min',
#              'acc_yc|max',
#              'acc_yg|min',
#              'acc_xg|qtl_05',
#              'acc_xg|qtl_95',
#              'accg|mean',
#              'acc_z|mean',
#              'accg|qtl_20',
#              'acc|qtl_05',
#              'acc_xg|qtl_80',
#              'G|qtl_20',
#              'accg|median',
#              'G|qtl_80',
#              'G|qtl_05',
#              'acc_y|mean',
#              'acc_zc|qtl_05']
# used_feat = [f for f in train_df.columns
#              if f.split('|')[0] in [
#                  'acc_x', 'acc_y', 'acc_z', 'acc',
#                  'acc_xg', 'acc_yg', 'acc_zg', 'accg',
#                  'acc_xc', 'acc_yc', 'acc_zc', 'G',
#                  'delta_xy', 'delta_xz', 'delta_yz',
#              ] and f.split('|')[1] in [
#                  'min', 'max',
#                  'mean',
#                  'median',
#                  'std',
#                  # 'max_min',
#                  # 'peaks_num',
#              ]
#              ]
used_feat = [f for f in train_df.columns
             if f.split('|')[0] in [
                 'acc_x', 'acc_y', 'acc_z', 'acc',
                 'acc_xg', 'acc_yg', 'acc_zg', 'accg',
                 'acc_xc', 'acc_yc', 'acc_zc', 'G',
                 # 'delta_xy', 'delta_xz', 'delta_yz',
             ] and f.split('|')[1] in [
                 # 'min', 'max',
                 'mean',
                 'median',
                 'std',
                 # '95_05',
                 # '80_20',
                 # 'qtl_02',
                 # 'qtl_98',
                 # 'qtl_20',
                 # 'qtl_80',
                 # 'out_num',
                 # 'max_min',
                 # 'peaks_num',
             ]] + ['acc_yg|qtl_05', 'acc_yg|qtl_95', 'acc_xg|qtl_05', 'acc_xg|qtl_95'] + [
                f for f in train_df.columns
                if f.split('|')[0] in ['acc_x', 'acc_y', 'acc_z', 'acc'] and f.split('|')[1] in [
        'power_rate'
    ]
            ]

# used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
# used_feat = ['G|mean', 'acc_xg|qtl_05', 'acc_xg|qtl_95', 'acc_yg|qtl_95', 'acc_yg|mean', 'G|std',
#              'acc_z|mean', 'acc_y|std', 'acc_x|mean', 'acc|median', 'acc_x|std', 'acc|std', 'acc_zc|std', 'accg|median',
#              'G|median', 'acc_yg|qtl_05', 'acc_yc|std', 'acc_zg|mean', 'acc_y|mean', 'acc_yg|std', 'acc_xg|std',
#              'accg|mean', 'acc_xg|mean']
# used_feat = ['G|std', 'acc_yg|mean', 'acc_yg|qtl_95', 'acc_xg|mean', 'acc_yg|qtl_05', 'acc_y|std', 'G|median',
#              'acc_y|mean', 'acc_xg|qtl_05', 'acc_zg|mean', 'acc|median', 'acc_yc|std', 'acc_xg|qtl_95']
print(len(used_feat))
print(used_feat)

train_x = train_df[used_feat]
train_y = train_df[label]
test_x = test_df[used_feat]

# kdeplot(test_x, train_x, save_path='.')


folds = 5
seed = 2020

# model = VotingClassifier(estimators=[('model1', classifier_dict['RandomForestClassifier']()),
#                                      ('model2', classifier_dict['XGBClassifier']()),
#                                      ('model3', classifier_dict['LGBMClassifier']()),
#                                      ],
#                          # voting='soft'
#                          )

params = {
    'learning_rate': 0.1,
    'num_iterations': 20,
    'metric': 'multi_error',
    'objective': 'multiclass',
    'num_class': 19,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'n_jobs': 4,
    'seed': 2020,
    'max_depth': 10,
    'num_leaves': 64,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1
}

model_name = 'LGBMClassifier'
# learning_rate_list = [0.01, 0.02, 0.05, 0.1, 0.2]
# num_iterations_list = [50, 100, 150, 200]
# feature_fraction_list = [0.5, 0.7, 0.9]
# bagging_fraction_list = [0.5, 0.7, 0.9]
# lambda_l1_list = [0, 0.5, 1]
# lambda_l2_list = [0, 0.5, 1]
# max_depth_num_leaves_list = [[4, 8], [6, 16], [8, 32], [10, 64], [12, 128]]
# res_all_list = []
# for learning_rate in learning_rate_list:
#     for feature_fraction in feature_fraction_list:
#         for bagging_fraction in bagging_fraction_list:
#             for lambda_l1 in lambda_l1_list:
#                 for lambda_l2 in lambda_l2_list:
#                     for num_iterations in num_iterations_list:
#                         for max_depth, num_leaves in max_depth_num_leaves_list:

# learning_rate = 0.01
# feature_fraction = 0.5
# bagging_fraction = 0.5
# lambda_l1 = 0.5
# lambda_l2 = 0.5
# num_iterations = 150
# max_depth = 12
# num_leaves = 128
# model = classifier_dict[model_name](metric='multi_error', objective='multiclass', seed=2020,
#                                     n_jobs=4,
#                                     learning_rate=learning_rate,
#                                     feature_fraction=feature_fraction,
#                                     bagging_fraction=bagging_fraction,
#                                     lambda_l1=lambda_l1,
#                                     lambda_l2=lambda_l2,
#                                     num_iterations=num_iterations,
#                                     max_depth=max_depth,
#                                     num_leaves=num_leaves,
#                                     )
# model = classifier_dict[model_name](metric='multi_error', objective='multiclass', learning_rate=0.05,
#                                     lambda_l1=0.5, lambda_l2=0.5, max_depth=10, num_leaves=128)
model = classifier_dict[model_name](metric='multi_error', objective='multiclass')
res_list, pred_y, info_df = train(model, folds, train_x, train_y, test_x, info_return=True)

res_df = pd.DataFrame(res_list)
res_sr = res_df.apply(lambda x: x.value_counts().idxmax())
sub = pd.DataFrame([(test_df['fragment_id'] - 10000).values, res_sr.values],
                   index=['fragment_id', 'behavior_id']).T

pred_y = pred_y.sort_index()
score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(train_y, pred_y)) / len(
    pred_y)
print('*****')
print(model_name)
# print(feature_fraction, bagging_fraction, lambda_l1, lambda_l2, num_iterations, max_depth,
#       num_leaves)
print(score)
print('*****')
score_df = pd.concat([train_y, pred_y], axis=1)
# d = score_df[train_y != pred_y]

res_list = []
for id, part_df in score_df.groupby('behavior_id'):
    tmp_sr = part_df[0].value_counts().sort_index()
    tmp_sr.name = id
    # print(id, tmp_sr)
    res_list.append(tmp_sr)
a = pd.concat(res_list, axis=1)

#
# b = a[a>=17]
# tmp_list = []
# for i in range(19):
#     for j in range(i):
#         print(a.loc[i, j] + a.loc[i, j])
#         tmp_list.append([f'{i}_{j}', a.loc[i, j] + a.loc[i, j]])
# cor_df = pd.DataFrame(tmp_list)
# cor_df = cor_df.sort_values(1, ascending=False)


# res_all_list.append(
#     [feature_fraction, bagging_fraction, lambda_l1, lambda_l2, num_iterations,
#      max_depth, num_leaves, score])
# res_all_df = pd.DataFrame(res_all_list)
# res_all_df.to_csv('res_all_df.csv')
# sub.to_csv(f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_sub{round(score, 5)}.csv', index=False)

# n = 20
# gain_list = info_df['gain'].sort_values(ascending=False).iloc[:n].index
# split_list = info_df['split'].sort_values(ascending=False).iloc[:n].index
# use_list = list(set(gain_list) | set(split_list))
# print(use_list)
# print(len(use_list))
# print(len(set(used_feat) - set(use_list)))
# print(set(used_feat) - set(use_list))
# a = set(use_list) & set(used_feat)
# print(len(set(use_list) - set(used_feat)))
# print(len(set(used_feat) - set(use_list)))
