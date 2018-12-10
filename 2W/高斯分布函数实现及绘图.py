
# coding: utf-8

# # 高斯分布函数实现及绘图

# 在朴素贝叶斯的实验中，我们知道可以依照特征数据类型，在计算先验概率时对朴素贝叶斯模型进行划分，并分为：多项式模型，伯努利模型和高斯模型。而在前面的实验中，我们使用了多项式模型来完成。

# 很多时候，当我们的特征是连续变量时，运用多项式模型的效果不好。所以，我们通常会采用高斯模型对连续变量进行处理，而高斯模型实际上就是假设连续变量的特征数据是服从高斯分布。其中，高斯分布函数表达式为：


import numpy as np
from matplotlib import pyplot as plt


"""实现高斯分布函数
"""


def Gaussian(x, u, d):
    """
    参数:
    x -- 变量
    u -- 均值
    d -- 标准差

    返回:
    p -- 高斯分布值
    """
    p = (1 / (np.sqrt(2 * np.pi) * d)) * \
        np.exp((-np.square(x - u)) / (2 * np.square(d)))

    return p


x = np.linspace(-5, 5, 100)
u = 3.2
d = 5.5
g = Gaussian(x, u, d)

len(g), g[10]

u = [0, -1, 1, 0.5]
d = [1, 2, 0.5, 5]
colors = ['r', 'b', 'g', 'y']
x = np.linspace(-6, 6, 10000)
for i in range(4):
    plt.plot(x, Gaussian(x, u[i], d[i]), color=colors[i],
             label='u=%s, d=%s' % (u[i], d[i]))
