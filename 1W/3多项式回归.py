

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 加载示例数据
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]


get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(x, y)


"""实现 2 次多项式函数及误差函数
"""


def func(p, x):
    """根据公式，定义 2 次多项式函数
    """
    w0, w1, w2 = p
    f = w0 + w1 * x + w2 * x * x
    return f|


def err_func(p, x, y):
    """残差函数（观测值与拟合值之间的差距）
    """
    ret = func(p, x) - y
    return ret


p_init = np.random.randn(3)  # 生成 3 个随机数

p_init


"""使用 Scipy 提供的最小二乘法函数得到最佳拟合参数
"""

parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))

print('Fitting Parameters: ', parameters[0])


"""绘制 2 次多项式拟合图像
"""
# 绘制拟合图像时需要的临时点
x_temp = np.linspace(0, 80, 10000)

# 绘制拟合函数曲线
plt.plot(x_temp, func(parameters[0], x_temp), 'r')

# 绘制原数据点
plt.scatter(x, y)


"""实现 n 次多项式拟合
"""


def fit_func(p, x):
    """根据公式，定义 n 次多项式函数
    """
    f = np.poly1d(p)
    return f(x)


def err_func(p, x, y):
    """残差函数（观测值与拟合值之间的差距）
    """
    ret = fit_func(p, x) - y
    return ret


def n_poly(n):
    """n 次多项式拟合
    """
    p_init = np.random.randn(n)  # 生成 n 个随机数
    parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
    return parameters[0]


n_poly(3)


"""绘制出 3，4，5，6，7, 8, 9 次多项式的拟合图像
"""

# 绘制拟合图像时需要的临时点
x_temp = np.linspace(0, 80, 10000)

# 绘制子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].plot(x_temp, fit_func(n_poly(4), x_temp), 'r')
axes[0, 0].scatter(x, y)
axes[0, 0].set_title("m = 3")

axes[0, 1].plot(x_temp, fit_func(n_poly(5), x_temp), 'r')
axes[0, 1].scatter(x, y)
axes[0, 1].set_title("m = 4")

axes[0, 2].plot(x_temp, fit_func(n_poly(6), x_temp), 'r')
axes[0, 2].scatter(x, y)
axes[0, 2].set_title("m = 5")

axes[1, 0].plot(x_temp, fit_func(n_poly(7), x_temp), 'r')
axes[1, 0].scatter(x, y)
axes[1, 0].set_title("m = 6")

axes[1, 1].plot(x_temp, fit_func(n_poly(8), x_temp), 'r')
axes[1, 1].scatter(x, y)
axes[1, 1].set_title("m = 7")

axes[1, 2].plot(x_temp, fit_func(n_poly(9), x_temp), 'r')
axes[1, 2].scatter(x, y)
axes[1, 2].set_title("m = 8")


"""使用 PolynomialFeatures 自动生成特征矩阵
"""

X = [2, -1, 3]
X_reshape = np.array(X).reshape(len(X), 1)  # 转换为列向量
PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_reshape)


"""使用 sklearn 得到 2 次多项式回归特征矩阵
"""

x = np.array(x).reshape(len(x), 1)  # 转换为列向量
y = np.array(y).reshape(len(y), 1)


poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_x = poly_features.fit_transform(x)

poly_x


"""转换为线性回归预测
"""

# 定义线性回归模型
model = LinearRegression()
model.fit(poly_x, y)  # 训练

# 得到模型拟合参数
model.intercept_, model.coef_


"""绘制拟合图像
"""
x_temp = np.array(x_temp).reshape(len(x_temp), 1)
poly_x_temp = poly_features.fit_transform(x_temp)

plt.plot(x_temp, model.predict(poly_x_temp), 'r')
plt.scatter(x, y)


get_ipython().system(
    'wget -nc http://labfile.oss.aliyuncs.com/courses/1081/course-6-vaccine.csv')


# In[19]:


"""加载数据集
"""

df = pd.read_csv("course-6-vaccine.csv", header=0)
df


"""数据绘图
"""
# 定义 x, y 的取值
x = df['Year']
y = df['Values']
# 绘图
plt.plot(x, y, 'r')
plt.scatter(x, y)


"""训练集和测试集划分
"""
# 首先划分 dateframe 为训练集和测试集
train_df = df[:int(len(df) * 0.7)]
test_df = df[int(len(df) * 0.7):]

# 定义训练和测试使用的自变量和因变量
train_x = train_df['Year'].values
train_y = train_df['Values'].values

test_x = test_df['Year'].values
test_y = test_df['Values'].values


"""线性回归预测
"""
# 建立线性回归模型
model = LinearRegression()
model.fit(train_x.reshape(len(train_x), 1), train_y.reshape(len(train_y), 1))
results = model.predict(test_x.reshape(len(test_x), 1))
results  # 线性回归模型在测试集上的预测结果


"""线性回归误差计算
"""


print("线性回归平均绝对误差: ", mean_absolute_error(test_y, results.flatten()))
print("线性回归均方误差: ", mean_squared_error(test_y, results.flatten()))


# 接下来，开始训练 2 次多项式回归模型，并进行预测。

# In[25]:


"""2 次多项式预测
"""
# 2 次多项式回归特征矩阵
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_train_x_2 = poly_features_2.fit_transform(
    train_x.reshape(len(train_x), 1))
poly_test_x_2 = poly_features_2.fit_transform(test_x.reshape(len(test_x), 1))

# 2 次多项式回归模型训练与预测
model = LinearRegression()
model.fit(poly_train_x_2, train_y.reshape(len(train_x), 1))  # 训练模型

results_2 = model.predict(poly_test_x_2)  # 预测结果

results_2.flatten()  # 打印扁平化后的预测结果

"""2 次多项式预测误差计算
"""
print("2 次多项式回归平均绝对误差: ", mean_absolute_error(test_y, results_2.flatten()))
print("2 次多项式均方根误差: ", mean_squared_error(test_y, results_2.flatten()))


"""更高次多项式回归预测
"""

train_x = train_x.reshape(len(train_x), 1)
test_x = test_x.reshape(len(test_x), 1)
train_y = train_y.reshape(len(train_y), 1)

for m in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(
        m, include_bias=False), LinearRegression())
    model.fit(train_x, train_y)
    pre_y = model.predict(test_x)
    print("{} 次多项式回归平均绝对误差: ".format(m),
          mean_absolute_error(test_y, pre_y.flatten()))
    print("{} 次多项式均方根误差: ".format(m), mean_squared_error(test_y, pre_y.flatten()))
    print("---")


"""计算 m 次多项式回归预测结果的 MSE 评价指标并绘图
"""
mse = []  # 用于存储各最高次多项式 MSE 值
m = 1  # 初始 m 值
m_max = 10  # 设定最高次数
while m <= m_max:
    model = make_pipeline(PolynomialFeatures(
        m, include_bias=False), LinearRegression())
    model.fit(train_x, train_y)  # 训练模型
    pre_y = model.predict(test_x)  # 测试模型
    mse.append(mean_squared_error(test_y, pre_y.flatten()))  # 计算 MSE
    m = m + 1

print("MSE 计算结果: ", mse)
# 绘图
plt.plot([i for i in range(1, m_max + 1)], mse, 'r')
plt.scatter([i for i in range(1, m_max + 1)], mse)

# 绘制图名称等
plt.title("MSE of m degree of polynomial regression")
plt.xlabel("m")
plt.ylabel("MSE")
