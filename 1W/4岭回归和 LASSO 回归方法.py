
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # 岭回归和 LASSO 回归方法

# ---

# ### 实验介绍
# 
# 在回归预测中，除了前面所讲到的线性回归和多项式回归，还有很多的回归分析方法。例如，岭回归、LASSO 回归、以及由下一周分类方法变形而来的各类回归方法。本节实验中，我们先介绍岭回归和 LASSO 回归方法，这两种方法都应用到了正则化手段，可以解决线性回归无法覆盖到的一些问题。

# ### 实验知识点
# 
# - 岭回归原理
# - 岭回归拟合
# - LASSO 回归原理
# - LASSO 回归拟合
# - 正则化
# - 相关性系数

# ### 实验目录
# 
# - <a href="#普通最小二乘线性回归的局限性">普通最小二乘线性回归的局限性</a>
# - <a href="#岭回归">岭回归</a>
# - <a href="#LASSO-回归">LASSO 回归</a>
# - <a href="#实验总结">实验总结</a>
# - <a href="#本周思维导图">本周思维导图</a>

# ---

# ## 普通最小二乘法的局限性

# 前面的实验中，我们学习了使用最小二乘法来求解线性回归以及多项式回归问题。同时，在多项式回归的章节中，我们了解到多项回归其实可以看作是一种特殊的线性回归形式，也就是说回归方法的核心就是线性回归。

# ### 普通最小二乘法的矩阵表示

# 首先，回顾一下线性回归的内容。在线性回归中，我们需要求解：

# $$ f(x)=\sum_{i=1}^{m}w_i x_i =w^Tx \tag1$$

# 当我们使用普通最小二乘法（OLS）进行学习时（平方损失函数），就变成最小化目标函数：

# $$F=\sum_{i=1}^{n}(y_{i}-w^Tx)^2  \tag2$$

# 我们可以把 $(2)$ 式改写为向量表示：

# $$F={\left \| y-Xw \right \|_{2}}^{2}  \tag3$$

# <div style="color: #999;font-size: 12px;font-style: italic;">* $\left \| y-Xw \right \|_{2}$ 表示 2-范数，即向量元素绝对值的平方和再开方。</div>

# $(3)$ 式中，$X$ 为 $(x_{1},x_{2},……,x_{n})^{T}$ 数据构成的矩阵；$y$ 为 $(y_{1},y_{2},……,y_{n})^{T}$ 构成的列向量。同时，公式 $(3)$ 的解析解为：

# $$\hat{w} = (X^{T}X)^{-1}X^{T}y \tag4$$

# 公式 $(4)$ 成立的条件就是 $\left | X^{T}X \right |$ 不能为 `0`。而当变量之间的相关性较强（多重共线性），或者 $m$ 大于 $n$ 时，$(4)$ 式中的 $X$ 不是满秩（$rank(A)\neq dim(x)$）矩阵。从而使得 $\left | X^{T}X \right |$ 的结果趋近于 `0`，造成拟合参数的数值不稳定性增加，这也就是普通最小二乘法的局限。

# ### 希尔伯特矩阵 OLS 线性拟合

# 线性代数中，希尔伯特矩阵是一种系数都是单位分数的方块矩阵。具体来说一个希尔伯特矩阵 $H$ 的第 $i$ 横行第 $j$ 纵列的系数是：
# 
# $$H_{{ij}}={\frac  {1}{i+j-1}} \tag5$$

# 一个 `5x5` 的希尔伯特矩阵如公式$(6)$所示：
# 
# $$
# H_{5}={\begin{bmatrix}1&{\frac  {1}{2}}&{\frac  {1}{3}}&{\frac  {1}{4}}&{\frac  {1}{5}}\\[4pt]{\frac  {1}{2}}&{\frac  {1}{3}}&{\frac  {1}{4}}&{\frac  {1}{5}}&{\frac  {1}{6}}\\[4pt]{\frac  {1}{3}}&{\frac  {1}{4}}&{\frac  {1}{5}}&{\frac  {1}{6}}&{\frac  {1}{7}}\\[4pt]{\frac  {1}{4}}&{\frac  {1}{5}}&{\frac  {1}{6}}&{\frac  {1}{7}}&{\frac  {1}{8}}\\[4pt]{\frac  {1}{5}}&{\frac  {1}{6}}&{\frac  {1}{7}}&{\frac  {1}{8}}&{\frac  {1}{9}}\end{bmatrix}}\tag6
# $$

# 这里介绍希尔伯特矩阵的原因，是因为下面打算使用希尔伯特矩阵的数值完成一个最小二乘法的线性拟合实验。而希尔伯特矩阵每列数据之间存在较强的线性相关性，正好符合 `1.1` 节中提到的 $\left | X^{T}X \right |$ 的结果趋近于 `0`。

# 这里，我们使用 `Scipy` 提供的 `hilbert()` 方法，直接创建一个 `10x10` 的希尔伯特矩阵

# In[3]:


"""生成 10x10 的希尔伯特矩阵
"""
from scipy.linalg import hilbert

x = hilbert(10)
x


# **☞ 动手练习：**

# In[4]:


"""希尔伯特转置矩阵与原矩阵相乘
"""
import numpy as np

np.linalg.det(np.matrix(x).T * np.matrix(x))


# 可以看到，这里的 $\left | X^{T}X \right |$ 的结果的确趋近于 `0`。

# #### 皮尔逊相关系数（Pearson Correlation Coefficient）

# 皮尔逊相关系数（Pearson Correlation Coefficient）通常用于度量两个变量 `X` 和 `Y` 之间的线性相关程度，其值介于 `-1` 与 `1` 之间。其中，数值越趋近于 `1` 表示正相关程度越高，趋近于 `0` 表示线性相关度越低，趋近于 `-1` 则表示负相关程度越高。
# 
# Pandas 提供了直接计算相关系数的方法，从而计算出上方 `10x10` 的希尔伯特矩阵中，数据列之间的相关性系数。

# In[5]:


"""计算希尔伯特矩阵每列数据的相关性系数
"""
import pandas as pd

pd.DataFrame(x, columns=['x%d'%i for i in range(1,11)]).corr()


# 如上所示，`10x10` 的希尔伯特矩阵中，每一列数据间都存在着较高的数值相关性。

# #### 最小二乘线性拟合

# 上面由希尔伯特矩阵对应的示例数据集中，假设已知的 $x_{1}$ 到 $x_{10}$ 服从线性分布：
# 
# $$ y= w_{1} * x_{1} + w_{2} * x_{2} +……+w_{10} * x_{10}\tag7$$

# 接下来我们先使用 NumPy 随机生成 `w` 参数，并得到实际对应的 `y` 值。

# In[6]:


"""定义线性分布函数及实际参数
"""
from scipy.optimize import leastsq
import numpy as np
from scipy.linalg import hilbert

x = hilbert(10) # 生成 10x10 的希尔伯特矩阵
np.random.seed(10) # 随机数种子能保证每次生成的随机数一致
w = np.random.randint(2,10,10) # 随机生成 w 系数
y_temp = np.matrix(x) * np.matrix(w).T # 计算 y 值
y = np.array(y_temp.T)[0] #将 y 值转换成 1 维行向量

print("实际参数 w: ", w)
print("实际函数值 y: ", y)


# 接下来，我们便使用前面课程中学习到的最小二乘法对数据集进行线性拟合，并求出拟合得到的参数。

# In[7]:


"""使用最小二乘法线性拟合
"""
func=lambda p,x: np.dot(x, p) # 函数公式
err_func = lambda p, x, y: func(p, x)-y # 残差函数
p_init=np.random.randint(1,2,10) # 全部参数初始化为 1

parameters = leastsq(err_func, p_init, args=(x, y)) # 最小二乘法求解
print("拟合参数 w: ",parameters[0])


# 你会发现，实际参数 $w$ 和拟合参数 $w$ 差距非常大。这也就印证了 `1.1` 节中所说的普通最小二乘法的局限性。

# ## 岭回归

# 综上所示，普通最小二乘法带来的局限性，导致许多时候都不能直接使用其进行线性回归拟合。特别是以下两种情况：
# 
# -  **数据集的列（特征）数量 > 数据量（行数量），即 $X$ 不是列满秩。**
# - **数据集列（特征）数据之间存在较强的线性相关性，即模型容易出现过拟合。**

# ### 岭回归推导

# 为了解决上述两种情况中出现的问题，岭回归（Ridge Regression）应运而生。岭回归可以被看作为一种改良后的最小二乘估计法，它通过向损失函数中添加 $L_{2}$ 正则项（2-范数）有效防止模型出现过拟合，且以助于解决非满秩条件下求逆困难的问题，从而提升模型的解释能力。即，上面公式 $(2)$ 对应的损失函数变为：

# $$F_{Ridge}=\sum_{i=1}^{n}(y_{i}-w^Tx)^2 + \lambda \sum_{i=1}^{n}(w_{i})^2  \tag8$$

# 我们可以把 $(8)$ 式改写为向量表示：

# $$F_{Ridge}={\left \| y-Xw \right \|_{2}}^{2} + \lambda{\left \| w \right \|_{2}}^{2} \tag9$$

# 公式 $(9)$ 中回归系数 $w$ 的解析解为：
# 
# $$\hat w_{Ridge} = (X^TX + \lambda I)^{-1} X^TY \tag{10}$$

# 从公式$(4)$和公式$(10)$的区别可以看出，通过给 $X^TX$ 增加一个单位矩阵，从而使得矩阵变成满秩，完善普通最小二乘法的不足。

# ### 岭回归拟合

# 接下来，我们通过 scikit-learn 提供的岭回归方法 `Ridge()` 完成对 `1.2` 节中的数据拟合。

# ```python
# sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
# ```
# - `alpha`: 正则化强度，默认为 1.0，对应公式 8 中的 $\lambda$。
# - `fit_intercept`: 默认为 True，计算截距项。
# - `normalize`: 默认为 False，不针对数据进行标准化处理。
# - `copy_X`: 默认为 True，即使用数据的副本进行操作，防止影响原数据。
# - `max_iter`: 最大迭代次数，默认为 None。
# - `tol`: 数据解算精度。
# - `solver`: 根据数据类型自动选择求解器。
# - `random_state`: 随机数发生器。

# In[ ]:


"""使用岭回归拟合
"""
from sklearn.linear_model import Ridge

ridge_model = Ridge(fit_intercept=False) # 参数代表不增加截距项
ridge_model.fit(x, y)


# In[8]:


from sklearn.linear_model import Ridge

ridge_model = Ridge(fit_intercept=False)
ridge_model.fit(x,y)


# In[ ]:


ridge_model.coef_ # 打印模型参数


# In[9]:


ridge_model.coef_


# 可以看出，虽然和模型真实参数 `[3 7 6 9 2 3 5 6 3 7]` 存在差距，但是已经比普通最小二乘法拟合的参数好很多了。

# <div style="color: #999;font-size: 12px;font-style: italic;">
# **常见疑问**：如果你足够细心，在此使用 `sklearn.linear_model.LinearRegression()` 构建线性回归模型去求解参数时，你会发现得到的参数和岭回归拟合接近，而与上文中使用最小二乘法相差较大，为什么呢？
# <br><br>
# **导师答疑**：其实这就说明像 scikit-learn 提供封装好的线性回归类 `LinearRegression()` 并不只是实现了简单的最小二乘法。可能是为了方便使用，也会在某些条件下默认使用正则化等手段，所以才会出现和普通最小二乘法结果不一的情况。scikit-learn 是一个十分成熟的模块，每个类支持的参数和实现的功能非常多，大家可以通过官方文档详细了解。
# </div>

# `Ridge()` 模型中的 `alpha` 参数代表正则化强度。下面，我们尝试调节该参数得到不同的拟合结果。

# In[ ]:


"""不同 alpha 参数拟合
"""
alphas = np.linspace(-3,2,20)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(x, y)
    coefs.append(ridge.coef_)


# In[11]:


alphas = np.linspace(-3,2,20)
coefs = []

for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(x,y)
    coefs.append(ridge.coef_)
coefs


# In[ ]:


"""绘制不同 alpha 参数结果
"""
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(alphas, coefs) # 绘制不同 alpha 参数下的 w 拟合值
plt.scatter(np.linspace(0,0,10), parameters[0]) # 普通最小二乘法拟合的 w 值放入图中
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Ridge Regression')


# In[12]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(alphas, coefs)
plt.scatter(np.linspace(0,0,10), parameters[0])
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Ridge Regression')


# 由图可见，当 `alpha` 取值越大时，正则项主导收敛过程，各 $w$ 系数趋近于 `0`。当 `alpha` 很小时，各 $w$ 系数波动幅度变大。

# ## LASSO 回归

# 当我们使用普通最小二乘法进行回归拟合时，如果特征变量间的相关性较强，则可能会导致某些 $w$ 系数很大，而另一些系数变成很小的负数。所以，我们通过上文中的岭回归添加 $L_{2}$ 正则项来解决这个问题。

# 与岭回归相似的是，LASSO 回归同样是通过添加正则项来改进普通最小二乘法，不过这里添加的是 $L_{1}$ 正则项。即：

# $$F_{LASSO}={\left \| y-Xw \right \|_{2}}^{2} + \lambda{\left \| w \right \|_{1}} \tag{12}$$

# ### LASSO 回归拟合

# 同样，我们通过 scikit-learn 提供的 LASSO 回归方法 `Lasso()` 完成对 `1.2` 节中的数据拟合。这里就不再像岭回归一节中分步骤书写，而是写在同一个单元格中。

# ```python
# sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
# ```
# - `alpha`: 正则化强度，默认为 1.0。
# - `fit_intercept`: 默认为 True，计算截距项。
# - `normalize`: 默认为 False，不针对数据进行标准化处理。
# - `precompute`: 是否使用预先计算的 Gram 矩阵来加速计算。
# - `copy_X`: 默认为 True，即使用数据的副本进行操作，防止影响原数据。
# - `max_iter`: 最大迭代次数，默认为 1000。
# - `tol`: 数据解算精度。
# - `warm_start`: 重用先前调用的解决方案以适合初始化。
# - `positive`: 强制系数为正值。
# - `random_state`: 随机数发生器。
# - `selection`: 每次迭代都会更新一个随机系数。

# In[14]:


"""使用 LASSO 回归拟合并绘图
"""
from sklearn.linear_model import Lasso

alphas = np.linspace(-2,2,10)
lasso_coefs = []

for a in alphas:
    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(x, y)
    lasso_coefs.append(lasso.coef_)
    
plt.plot(alphas, lasso_coefs) # 绘制不同 alpha 参数下的 w 拟合值
plt.scatter(np.linspace(0,0,10), parameters[0]) # 普通最小二乘法拟合的 w 值放入图中
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Lasso Regression')


# In[15]:


from sklearn.linear_model import Lasso

alphas = np.linspace(-2,2,10)
lasso_coefs = []

for a in alphas:
    lasso = Lasso(alpha=a, fit_intercept=False)
    lasso.fit(x, y)
    lasso_coefs.append(lasso.coef_)
    
plt.plot(alphas, lasso_coefs) # 绘制不同 alpha 参数下的 w 拟合值
plt.scatter(np.linspace(0,0,10), parameters[0]) # 普通最小二乘法拟合的 w 值放入图中
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Lasso Regression')


# 由图可见，当 `alpha` 取值越大时，正则项主导收敛过程，各 $w$ 系数趋近于 `0`。当 `alpha` 很小时，各 $w$ 系数波动幅度变大。

# ## 实验总结

# 本次实验，我们接触到了两种引入正则化方法的回归拟合方法，分别是岭回归和 LASSO 回归。二者有很多相似之处，但也有一些不同。本次试验涉及到的数学公式和原理较多，这需要你具备一定的线性代数知识。本次实验的知识点主要有：
# - 岭回归原理
# - 岭回归拟合
# - LASSO 回归原理
# - LASSO 回归拟合
# - 正则化
# - 相关性系数

# **拓展阅读**
# 
# - [机器学习中使用正则化来防止过拟合是什么原理？](https://www.zhihu.com/question/20700829)
# - [最小二乘法的本质是什么？](https://www.zhihu.com/question/37031188/answer/111336809)
# - [机器学习中常常提到的正则化到底是什么意思？](https://www.zhihu.com/question/20924039)

# ---

# ## 本周思维导图

# 学习完监督学习：回归的内容，我们总结本周知识点并绘制思维导图。思维导图是一种非常高效的学习手段，我们非常推荐你在学习的过程中自行梳理知识点。

# ![image](https://doc.shiyanlou.com/document-uid214893labid7506timestamp1542013311291.png)

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
