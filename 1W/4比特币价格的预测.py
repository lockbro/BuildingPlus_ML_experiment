
# coding: utf-8

# # 比特币价格预测挑战

# ### 挑战知识点

# - 使用 Pandas 数据处理
# - 使用 Matplotlib 绘图
# - 使用 scikit-learn 训练多项式回归预测模型

# 首先，需要导入比特币历史数据集，并预览数据集前 5 行。数据集名称为 `challenge-2-bitcoin.csv`。

# import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# 读数据
df = pd.read_csv("challenge-2-bitcoin.csv")
df.head()
data = df[['btc_market_price', 'btc_total_bitcoins', 'btc_transaction_fees']]
data.head()

"""绘制数据图像"""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
line_num = data['btc_market_price'].size

# axes[0].plot([i for i in range(0, line_num)],
#              data['btc_market_price'].values, 'k')
axes[0].plot(data['btc_market_price'], 'k')
axes[0].set_xlabel('time')
axes[0].set_ylabel('btc_market_price')

axes[1].plot([i for i in range(0, line_num)],
             data['btc_total_bitcoins'].values, 'k')
axes[1].set_xlabel('time')
axes[1].set_ylabel('btc_total_bitcoins')

axes[2].plot([i for i in range(0, line_num)],
             data['btc_transaction_fees'].values, 'k')
axes[2].set_xlabel('time')
axes[2].set_ylabel('btc_transaciton_fees')


"""划分数据集函数
"""


def split_dataset():
    """
    参数:
    无

    返回:
    train_x, train_y, test_x, test_y -- 训练集特征、训练集目标、测试集特征、测试集目标
    """

    # 我们先把训练集和测试集分开
    train_data = data[:int(len(data) * 0.7)]
    test_data = data[int(len(data) * 0.7):]

    # 然后再把训练数据的特征和目标分出来
    train_x = train_data[['btc_total_bitcoins', 'btc_transaction_fees']].values
    train_y = train_data[['btc_market_price']].values

    # 然后再把测试数据的特征和目标分出来
    test_x = test_data[['btc_total_bitcoins', 'btc_transaction_fees']].values
    test_y = test_data[['btc_market_price']].values

    return train_x, train_y, test_x, test_y


# 测试结果
len(split_dataset()[0]),
len(split_dataset()[1]),
len(split_dataset()[2]),
len(split_dataset()[3]),
split_dataset()[0].shape,
split_dataset()[
    1].shape,
split_dataset()[2].shape,
split_dataset()[3].shape

train_x = split_dataset()[0]
train_y = split_dataset()[1]
test_x = split_dataset()[2]
test_y = split_dataset()[3]


"""3 次多项式回归预测模型
"""


def poly3():
    """
    参数:
    无

    返回:
    mae -- 预测结果的 MAE 评价指标
    """

    # 先把测试集和训练集的数据转化为特征矩阵
    poly_feature_3 = PolynomialFeatures(degree=3, include_bias=False)
    poly_train_x_3 = poly_feature_3.fit_transform(
        train_x.reshape(len(train_x), 2))
    poly_test_x_3 = poly_feature_3.fit_transform(
        test_x.reshape(len(test_x), 2))

    # 然后用线性回归来拟合得出训练集的w
    model = LinearRegression()
    model.fit(poly_train_x_3, train_y.reshape(len(train_y), 1))

    # 之后我们用训练完之后的model来预测试集的特征列（feature）
    result = model.predict(poly_test_x_3)
    mae = mean_absolute_error(test_y, result.flatten())

    return mae


poly3()

"""N 次多项式回归预测模型
"""


def poly_plot(N):
    """
    参数:
    N -- 标量, 多项式次数

    返回:
    mse -- N 次多项式预测结果的 MSE 评价指标列表
    """

    m = 1
    mse = []

    while m <= N:
        # 使用make_pipeline将求特征矩阵和相性回归化为一步
        model = make_pipeline(PolynomialFeatures(
            m, include_bias=False), LinearRegression())
        # 正常拟合数据
        model.fit(train_x.reshape(len(train_x), 2),
                  train_y.reshape(len(train_y), 1))
        # 用训练好的模型预测
        pre_y = model.predict(test_x.reshape(len(test_x), 2))
        mse.append(mean_squared_error(test_y, pre_y.flatten()))
        m = m + 1
    return mse


poly_plot(10)[:10:3]


mse = poly_plot(10)
# 把MSE的评测结果可视化出来
plt.plot([i for i in range(1, 11)], mse, 'k')
plt.scatter([i for i in range(1, 11)], mse)


plt.title("MSE")
plt.xlabel("N")
plt.ylabel("MSE")
