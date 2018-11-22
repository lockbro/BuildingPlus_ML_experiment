
# coding: utf-8

import numpy as np
from sklearn.linear_model import Ridge


"""岭回归系数 w 的计算函数
"""


def ridge_regression(X, Y, alpha):
    """
    参数:
    X -- 自变量数据矩阵
    Y -- 因变量数据矩阵
    alpha -- lamda 参数

    返回:
    W -- 岭回归系数
    """
    # 单位矩阵的大小以X的大小来定（因为要想加的话必须得形状一样）
    # 也可以len(X)获取大小
    num_eye = np.shape(X)[0]
    # 矩阵要直接乘必须满足乘数和被除数都是矩阵，要是是array的话就不能直接乘
    W = (((X.T * X) + alpha * np.eye(num_eye)).I) * X.T * Y

    return W


# 设置随机数种子
np.random.seed(10)

X = np.matrix(np.random.randint(5, size=(10, 10)))
Y = np.matrix(np.random.randint(10, size=(10, 1)))
alpha = 0.5


ridge_regression(X, Y, alpha).T


"""使用 scikit-learn 计算岭回归系数 w
"""


def ridge_model(X, Y, alpha):
    """
    参数:
    X -- 自变量数据矩阵
    Y -- 因变量数据矩阵
    alpha -- lamda 参数

    返回:
    W -- 岭回归系数
    """

    ridge_model = Ridge(alpha=alpha, fit_intercept=False)
    ridge_model.fit(X, Y)
    W = ridge_model.coef_

    return W


ridge_model(X, Y, alpha)
