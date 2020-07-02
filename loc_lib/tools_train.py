import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RepeatedKFold, cross_val_score, \
    cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from collections import OrderedDict

classifier_dict = OrderedDict({
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
})

regressor_dict = OrderedDict({
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'AdaBoostRegressor': AdaBoostRegressor,
    'LinearRegression': LinearRegression,
    'Lasso': Lasso,
    'Ridge': Ridge,
    'ElasticNet': ElasticNet,
    'LinearSVR': LinearSVR,
    'SVR': SVR,
    'XGBRegressor': XGBRegressor,
    'LGBMRegressor': LGBMRegressor,
    'KNeighborsRegressor': KNeighborsRegressor,
})


def train_reg_model(model, param_grid, X, y, splits=5, repeats=5):
    # get unmodified training data, unless data to use already specified
    # if len(y) == 0:
    #     X, y = get_trainning_data_omitoutliers()
    #     # poly_trans=PolynomialFeatures(degree=2)
    #     # X=poly_trans.fit_transform(X)
    #     # X=MinMaxScaler().fit_transform(X)

    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

    # perform a grid search if param_grid given
    # if len(param_grid) > 0:
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring="mean_squared_error",
                           verbose=1, return_train_score=True, n_jobs=3)

    # search the grid
    gsearch.fit(X, y)

    # extract best model from the grid
    model = gsearch.best_estimator_
    best_idx = gsearch.best_index_

    # get cv-scores for best model
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
    cv_std = grid_results.loc[best_idx, 'std_test_score']

    # no grid search, just cross-val score for given model
    # else:
    #     grid_results = []
    #     cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)
    #     cv_mean = abs(np.mean(cv_results))
    #     cv_std = np.std(cv_results)

    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)

    # print stats on model performance
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=', model.score(X, y))
    print('rmse=', np.sqrt(mean_squared_error(y, y_pred)))
    print('mse=', mean_squared_error(y, y_pred))
    print('cross_val: mean=', cv_mean, ', std=', cv_std)

    # residual plots
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    # n_outliers = sum(abs(z) > 3)
    #
    # plt.figure(figsize=(15, 5))
    # ax_131 = plt.subplot(1, 3, 1)
    # plt.plot(y, y_pred, '.')
    # plt.xlabel('y')
    # plt.ylabel('y_pred')
    # plt.title('corr = {:.3f}'.format(np.corrcoef(y, y_pred)[0][1]))
    # ax_132 = plt.subplot(1, 3, 2)
    # plt.plot(y, y - y_pred, '.')
    # plt.xlabel('y')
    # plt.ylabel('y - y_pred')
    # plt.title('std resid = {:.3f}'.format(std_resid))
    #
    # ax_133 = plt.subplot(1, 3, 3)
    # z.plot.hist(bins=50, ax=ax_133)
    # plt.xlabel('z')
    # plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results, mean_squared_error(y, y_pred)


def train_cls_model(model, param_grid, X, y, splits=5, repeats=5):
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           verbose=1, return_train_score=True, n_jobs=3)

    # search the grid
    gsearch.fit(X, y)

    # extract best model from the grid
    model = gsearch.best_estimator_
    best_idx = gsearch.best_index_

    # get cv-scores for best model
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
    cv_std = grid_results.loc[best_idx, 'std_test_score']

    # no grid search, just cross-val score for given model
    # else:
    #     grid_results = []
    #     cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)
    #     cv_mean = abs(np.mean(cv_results))
    #     cv_std = np.std(cv_results)

    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)

    # print stats on model performance
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=', model.score(X, y))
    print('rmse=', np.sqrt(mean_squared_error(y, y_pred)))
    print('mse=', mean_squared_error(y, y_pred))
    print('cross_val: mean=', cv_mean, ', std=', cv_std)

    # residual plots
    y_pred = pd.Series(y_pred, index=y.index)
    return model, cv_score, grid_results, mean_squared_error(y, y_pred)


def train(model, folds, train_x, train_y, test_x, info_return=False):
    def part_train(trn_idx, val_idx, fold):
        x_trn, y_trn, x_val, y_val = train_x.iloc[trn_idx], train_y.iloc[trn_idx], train_x.iloc[val_idx], \
                                     train_y.iloc[val_idx]
        # model.fit(x_trn, y_trn, eval_set=[(x_val, y_val)], verbose=-1)
        model.fit(x_trn, y_trn)
        y_val_pred = model.predict(x_val)
        y_trn_pred = model.predict(x_trn)
        print('train score:', accuracy_score(y_trn, y_trn_pred))
        print('test score:', accuracy_score(y_val, y_val_pred))
        res = model.predict(test_x)
        if info_return:
            model.importance_type = 'split'
            info_df[f'split_{fold}'] = model.feature_importances_
            model.importance_type = 'gain'
            info_df[f'gain_{fold}'] = model.feature_importances_
        return res, pd.Series(y_val_pred, index=val_idx)

    seeds = [2020]
    res_list = []
    pred_y_list = []
    info_df = pd.DataFrame(index=train_x.columns)
    for seed in seeds:
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
            print('_______________________________')
            res, y_val_pred = part_train(trn_idx, val_idx, fold)
            res_list.append(res)
            pred_y_list.append(y_val_pred)
    pred_y = pd.concat(pred_y_list, axis=0)
    if info_return:
        info_df['split'] = info_df[[x for x in info_df.columns if 'split' in x]].sum(1)
        info_df['gain'] = info_df[[x for x in info_df.columns if 'gain' in x]].sum(1)
    return res_list, pred_y, info_df
