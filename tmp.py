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
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# mean_squared_error(y_true, y_pred)

data_train = pd.read_csv('./data/zhengqi_train.txt', sep='\t')
data_test = pd.read_csv('./data/zhengqi_test.txt', sep='\t')


def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff ** 2)
    n = len(y_pred)

    return np.sqrt(sum_sq / n)


def mse(y_ture, y_pred):
    return mean_squared_error(y_ture, y_pred)


def find_outliers(model, X, y, sigma=3):
    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # print and plot the results
    print('R2=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print("mse=", mean_squared_error(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:', mean_resid)
    print('std of residuals:', std_resid)
    print('---------------------------------------')

    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    # plt.figure(figsize=(15, 5))
    # ax_131 = plt.subplot(1, 3, 1)
    # plt.plot(y, y_pred, '.')
    # plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    # plt.legend(['Accepted', 'Outlier'])
    # plt.xlabel('y')
    # plt.ylabel('y_pred');
    #
    # ax_132 = plt.subplot(1, 3, 2)
    # plt.plot(y, y - y_pred, '.')
    # plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    # plt.legend(['Accepted', 'Outlier'])
    # plt.xlabel('y')
    # plt.ylabel('y - y_pred');
    #
    # ax_133 = plt.subplot(1, 3, 3)
    # z.plot.hist(bins=50, ax=ax_133)
    # z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    # plt.legend(['Accepted', 'Outlier'])
    # plt.xlabel('z')
    #
    # plt.savefig('outliers.png')

    return outliers


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def train_model(model, param_grid, X, y, splits=5, repeats=5):
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
                           scoring="neg_mean_squared_error",
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
    print('rmse=', rmse(y, y_pred))
    print('mse=', mse(y, y_pred))
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

    return model, cv_score, grid_results, mse(y, y_pred)


data_train = reduce_mem_usage(data_train)
data_test = reduce_mem_usage(data_test)

drop_list = ["V5", "V9", "V11", "V17", "V22", "V28",
             ]
data_train.drop(columns=drop_list, inplace=True)
data_test.drop(columns=drop_list, inplace=True)


# low_corr = ['V14', 'V21', 'V25', 'V26', 'V32', 'V33', 'V34',]
#
# data_train.drop(columns=low_corr, inplace=True)
# data_test.drop(columns=low_corr, inplace=True)

# scatter_matrix(data_train, figsize=(20, 16))

num_pipeline = Pipeline([
    # ('Imputer', Imputer("median")),
    ('StandardScaler', StandardScaler()),
])
def std_fun(data):
    data_std = num_pipeline.fit_transform(data)
    data_std = pd.DataFrame(data_std, index=data.index,
                                  columns=data.columns)
    return data_std


# data_train = std_fun(data_train)
# data_test = std_fun(data_test)


# for x in data_test.columns:
#     g = sns.kdeplot(data_train[x], color="Red", shade=True)
#     g = sns.kdeplot(data_test[x], ax=g, color="Blue", shade=True)
#     g.set_xlabel(x)
#     g.set_ylabel("Frequency")
#     g = g.legend(["train", "test"])
#     plt.show()

# savfig_send()

# data_train.iloc[:, :10].hist(bins=50, figsize=[20, 15])
# linear_regression = LinearRegression().score()
# X = data_train.iloc[:, :-1]
# y = data_train['target']
# linear_model = linear_regression.fit(X, y)

# outliers = find_outliers(Ridge(), X, y)

# model, cv_score, grid_results = train_model(LinearRegression(), {}, X=X, y=y, splits=5, repeats=5)

# places to store optimal models and scores
# opt_models = dict()
# score_models = pd.DataFrame(columns=['mean','std'])

# # no. k-fold splits
# splits=5
# # no. k-fold iterations
# repeats=5
#
# model = 'XGB'
# opt_models[model] = XGBRegressor()
#
# param_grid = {'n_estimators':[100,200,300,400,500],
#               'max_depth':[1,2,3],
#              }
#
# opt_models[model], cv_score,grid_results = train_model(opt_models[model], param_grid=param_grid, X=X, y=y,
#                                               splits=splits, repeats=1)
#
# cv_score.name = model
# score_models = score_models.append(cv_score)
#
# y_test = opt_models['XGB'].predict(data_test)
# pd.Series(y_test).to_csv('./y_test.txt', index=False, header=False)
# X.iloc[:, :10].hist(bins=50, figsize=[20, 10])
# plt.ion()
# plt.show()
# plt.plotting()
# model_dict = {}
# for fun in [
#     LinearRegression,
#     # Lasso, ElasticNet, LinearSVR, SVR,
#     # RandomForestRegressor,
#     # GradientBoostingRegressor, AdaBoostRegressor,
#     # XGBRegressor,
#     Ridge,
# ]:
#     print('______________________')
#     fun_name = fun.__name__
#     print(fun_name)
#     model = fun()
#     # model = LinearRegression()
#     model, cv_score, grid_results, model_mse = train_model(model, {}, X, y, splits=5, repeats=5)
#     model_dict[fun_name] = [model, model_mse]

# for model_name in model_dict.keys():
#     # model_name = 'Ridge'
#     model, model_mse = model_dict[model_name]
#     print(model_mse)
#     pd.Series(model.predict(data_test)).to_csv(f'./y_test_{model_name}.txt', index=False, header=False)

a = pd.read_csv('./y_test_haha.txt', index_col=None, header=None)
X = data_train.iloc[:, :-1]
y = data_train['target']
model = Ridge()
model.fit(X, y)
print(mse(y, model.predict(X)))
print(mse(a, model.predict(data_test)))
