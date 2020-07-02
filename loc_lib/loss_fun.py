import numpy as np


def logistic_obj(y_hat, y):
    # y = dtrain.get_label()
    print(y_hat)
    print(len(y_hat))
    print(y)
    print(len(list(y)))
    p = y_hat
    # p = 1. / (1. + np.exp(-y_hat)) # 用于避免hessian矩阵中很多0
    grad = p - y
    hess = p * (1. - p)
    grad = 4 * p * y + p - 5 * y
    hess = (4 * y + 1) * (p * (1.0 - p))
    return grad, hess


def f1_loss(y, pred):
    print(pred)
    print(len(pred), type(pred))
    print(y)
    print(len(y), type(pred))
    beta = 2
    p = 1. / (1 + np.exp(-pred))
    grad = p * ((beta - 1) * y + 1) - beta * y
    hess = ((beta - 1) * y + 1) * p * (1.0 - p)
    return grad, hess