import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import gc
import os
from scipy import signal
from tqdm import tqdm

pd.set_option('display.max_columns', 600)
pd.set_option('display.max_rows', 600)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

data_train = pd.read_csv('sensor_train.csv')
data_test = pd.read_csv('sensor_test.csv')


def get_data():
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

        df[f + '_' + 'qtl_05'] = data.groupby('fragment_id')[f].quantile(0.05)
        df[f + '_' + 'qtl_95'] = data.groupby('fragment_id')[f].quantile(0.95)

        df[f + '_' + 'qtl_20'] = data.groupby('fragment_id')[f].quantile(0.20)
        df[f + '_' + 'qtl_80'] = data.groupby('fragment_id')[f].quantile(0.80)

        # df[f + '_' + 'max_min'] = df[f + '_' + 'max'] - df[f + '_' + 'min']
        # df[f + '_' + 'max_mean'] = df[f + '_' + 'max'] - df[f + '_' + 'mean']
        # df[f + '_' + 'mean_min'] = df[f + '_' + 'mean'] - df[f + '_' + 'min']

        def fun(x):
            x = x.rolling(window=5).mean()
            return len(signal.find_peaks(x, distance=3)[0])

        df[f + '_' + 'peaks_num'] = data.groupby('fragment_id')[f].apply(fun)

    return df


label = 'behavior_id'
df = get_data()

train_df = df[df[label].notna()].reset_index(drop=True)
test_df = df[df[label].isna()].reset_index(drop=True)

drop_feat = []
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
# used_feat = ['acc_xg_peaks_num', 'acc_xg_qtl_80', 'acc_zc_min', 'acc_y_qtl_05', 'G_std', 'acc_zg_qtl_05',
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
scores = []
imp = pd.DataFrame()
imp['feat'] = used_feat

# params = {
#     'learning_rate': 0.1,
#     'num_iterations': 20,
#     'metric': 'multi_error',
#     'objective': 'multiclass',
#     'num_class': 19,
#     'feature_fraction': 0.80,
#     'bagging_fraction': 0.75,
#     'bagging_freq': 2,
#     'n_jobs': 4,
#     'seed': 2020,
#     'max_depth': 10,
#     'num_leaves': 64,
#     'lambda_l1': 0.5,
#     'lambda_l2': 0.5,
#     'verbose': -1
# }
#
# oof_train = np.zeros((len(train_x), 19))
# preds = np.zeros((len(test_x), 19))
# folds = 4
# # seeds = [44]  # , 2020, 527, 1527]
# seeds = [2020]
# for seed in seeds:
#     kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
#     for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
#         x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], train_y.iloc[
#             val_idx]
#         train_set = lgb.Dataset(x_trn, y_trn)
#         val_set = lgb.Dataset(x_val, y_val)
#
#         model = lgb.train(params, train_set, num_boost_round=500000,
#                           valid_sets=(train_set, val_set), early_stopping_rounds=50,
#                           verbose_eval=50)
#         oof_train[val_idx] += model.predict(x_val) / len(seeds)
#         preds += model.predict(test_x) / folds / len(seeds)
#         scores.append(model.best_score['valid_1']['multi_error'])
#         imp['gain' + str(fold + 1)] = model.feature_importance(importance_type='gain')
#         imp['split' + str(fold + 1)] = model.feature_importance(importance_type='split')
#         del x_trn, y_trn, x_val, y_val, model, train_set, val_set
#         gc.collect()
# imp['gain'] = imp[[f for f in imp.columns if 'gain' in f]].sum(axis=1) / folds
# imp['split'] = imp[[f for f in imp.columns if 'split' in f]].sum(axis=1)
# imp = imp.sort_values(by=['gain'], ascending=False)
# print(imp[['feat', 'gain', 'split']])
# imp = imp.sort_values(by=['split'], ascending=False)
# print(imp[['feat', 'gain', 'split']])


# a = set(imp[['feat', 'gain', 'split']].sort_values(by=['gain'], ascending=False).iloc[:50]['feat'])
# b = set(imp[['feat', 'gain', 'split']].sort_values(by=['split'], ascending=False).iloc[:50]['feat'])
# len(list(a | b))

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


# labels = np.argmax(preds, axis=1)
# oof_y = np.argmax(oof_train, axis=1)
# print(round(accuracy_score(train_y, oof_y), 5))
# score = sum(acc_combo(y_true, y_pred) for y_true, y_pred in zip(train_y, oof_y)) / oof_y.shape[0]
# print(round(score, 5))
# sub = pd.read_csv('提交结果示例.csv')
#
# sub['behavior_id'] = labels

# plt.figure(figsize=[10, 8])
# vc = data_train['behavior_id'].value_counts().sort_index()
# sns.barplot(vc.index, vc.values)
# plt.show()
# plt.figure(figsize=[10, 8])
# vc = sub['behavior_id'].value_counts().sort_index()
# sns.barplot(vc.index, vc.values)
# plt.show()
# from datetime import datetime
#
# sub.to_csv(f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_sub{round(score, 5)}.csv', index=False)

# train_set = xgb.Dataset()
# from xgboost.sklearn import XGBClassifier

# model = XGBClassifier(n_estimators=20, learning_rate=0.1, num_class=19, max_depth=6,
#                       reg_lambda=0.5, reg_alpha=0.5, gamma=0.1, seed=2020,
#                       objective='multi:softmax', colsample_bytree=0.8, subsample=0.8, eval_metric='merror')
# model.fit(X_train, y_train, eval_set=[(x_test, y_test)], verbose=False, early_stopping_rounds=50)


parameters = {
    'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'n_estimators': [500, 1000, 2000, 3000, 5000],
    'min_child_weight': [0, 2, 5, 10, 20],
    'max_delta_step': [0, 0.2, 0.6, 1, 2],
    'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
}

model = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=20,
                        silent=True,
                        objective='multi:softprob',
                        nthread=4,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0.5,
                        reg_lambda=0.5,
                        scale_pos_weight=1,
                        seed=2020,
                        missing=None,
                        eval_metric='merror')
oof_train = np.zeros((len(train_x), 19))
preds = np.zeros((len(test_x), 19))
folds = 4
# seeds = [44]  # , 2020, 527, 1527]
seeds = [2020]
for seed in seeds:
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], \
                                     train_y.iloc[val_idx]
        model.fit(x_trn, y_trn,eval_set=[(x_val, y_val)],verbose=-1)
        oof_train[val_idx] += model.predict(x_val) / len(seeds)
