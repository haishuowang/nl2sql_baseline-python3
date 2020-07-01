import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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
    print('rmse=', np.array(mean_squared_error(y, y_pred)))
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
