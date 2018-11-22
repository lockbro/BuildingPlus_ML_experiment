
# coding: utf-8
# ### 挑战介绍

# 关于线性回归的实验中，我们以预测波士顿地区房价举例，详细讨论了其实现过程。本次挑战中，你需要运用从线性回归中学习到的相关知识，来预测北京市的住房价格。

# ### 挑战知识点

# - 线性回归原理及应用
# - scikit-learn 回归预测

# ---

# ## 挑战内容

# ### 数据集读取与划分

# 挑战需要下载北京市部分小区的房价数据集，该数据集的名字为 `challenge-1-beijing.csv`。
# 点击运行下载数据集
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 读数据
get_ipython().system(
    'wget -nc http://labfile.oss.aliyuncs.com/courses/1081/challenge-1-beijing.csv')
# 代码开始 ### (≈ 2 行代码)
df = pd.read_csv('challenge-1-beijing.csv')
df.head()


# 可以看到，该数据集中共包含有 `12` 列。由于线性回归需要输入数值型数据，
# 所以我们选用的特征包括「公交，写字楼，医院，商场，地铁，学校，建造时间，楼层，面积」
# 等 `9` 项，而「每平米价格」则是预测目标值。
features = df[['公交', '写字楼', '医院', '商场', '地铁', '学校', '建造时间', '楼层', '面积']]
target = df['每平米价格']


# 将原始 DataFrame 分割为特征值 `features` 和目标值 `target` 之后。
# 我们还需要将这两个 DataFrame 划分为 `70%` 和 `30%` 的训练集和测试集。
# 其中，训练集特征、训练集目标、测试集特征和测试集目标分别定义为：
# `train_x`, `train_y`, `test_x`, `test_y`。
split_num = int(len(df) * 0.7)  # 70% 分割数
train_x = features[:split_num]
train_y = target[:split_num]
test_x = features[split_num:]
test_y = target[split_num:]
len(train_x), len(train_y), len(test_x), len(test_y)

# 拟合数据
model = LinearRegression()
model.fit(train_x, train_y)

# ### 模型评价
# 平均绝对百分比误差


def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """
    n = int(len(y_true))
    mape = sum(np.abs((y_true - y_pred) / y_true)) * 100 / n
    return mape


y_true = test_y.values
y_pred = model.predict(test_x)
mape(y_true, y_pred)
