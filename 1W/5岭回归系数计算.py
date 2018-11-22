
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # 岭回归系数 $w$ 计算

# ---

# ### 挑战介绍

# 前面的实验中，我们学习了岭回归和 LASSO 回归方法，并使用 scikit-learn 对两种方法进行了实战训练。本次挑战中，我们将尝试直接使用 Python 完成岭回归系数 $w$ 计算，并与 scikit-learn 计算结果进行比较。

# ### 挑战知识点

# - Python 计算岭回归系数
# - scikit-learn 计算岭回归系数

# ---

# ## 挑战内容

# ### 使用 Python 计算岭回归系数 $w$ 

# 前面的课程中，我们已经知道了岭回归的向量表达式：
# 
# $$F_{Ridge}={\left \| y-Xw \right \|_{2}}^{2} + \lambda{\left \| w \right \|_{2}}^{2} \tag1$$

# 以及该向量表达式的解析解：
# 
# $$\hat w_{Ridge} = (X^TX + \lambda I)^{-1} X^TY \tag2$$

# ---

# **<font color='red'>挑战</font>：参考公式 $(2)$，完成 Python 实现岭回归系数 $w$ 的计算函数。**

# **<font color='green'>提示</font>**：使用 `np.eye()` 生成单位矩阵，并注意公式是矩阵乘法。

# In[16]:


"""岭回归系数 w 的计算函数
"""

import numpy as np

def ridge_regression(X, Y, alpha):
    """
    参数:
    X -- 自变量数据矩阵
    Y -- 因变量数据矩阵
    alpha -- lamda 参数

    返回:
    W -- 岭回归系数
    """
    
    ### 代码开始 ### (≈ 3 行代码)
    num_eye = np.shape(X)[0]
    W = (((X.T * X) + alpha * np.eye(num_eye)).I) * X.T * Y
    ### 代码结束 ###
    
    return W


# 下面，我们生成测试数据：

# In[17]:


np.random.seed(10) # 设置随机数种子

X = np.matrix(np.random.randint(5, size=(10, 10)))
Y = np.matrix(np.random.randint(10, size=(10,1 )))
alpha = 0.5


# **运行测试：**

# 计算岭回归系数$w$的值：

# In[18]:


ridge_regression(X, Y, alpha).T


# **期望输出：**

# ```
# matrix([[ 1.42278923,  2.20583559, -0.6391644 ,  0.64022529, -0.44014758,
#           1.66307858, -0.83879894, -0.25611354, -0.06951638, -2.56882017]])
# ```

# ### 使用 scikit-learn 计算岭回归系数 $w$ 

# 上面的挑战中，你已经学会了使用 Python 计算岭回归系数 $w$。下面，我们看一看结果是否与 scikit-learn 的计算结果一致。

# ---

# **<font color='red'>挑战</font>：使用 scikit-learn 计算岭回归系数 $w$。**

# **<font color='green'>提示</font>**：请向岭回归模型中增加 `fit_intercept=False` 参数取消截距。

# In[19]:


"""使用 scikit-learn 计算岭回归系数 w
"""

from sklearn.linear_model import Ridge

def ridge_model(X, Y, alpha):
    """
    参数:
    X -- 自变量数据矩阵
    Y -- 因变量数据矩阵
    alpha -- lamda 参数

    返回:
    W -- 岭回归系数
    """
    
    ### 代码开始 ### (≈ 3 行代码)
    ridge_model = Ridge(alpha=alpha, fit_intercept=False)
    ridge_model.fit(X, Y)
    W = ridge_model.coef_
    ### 代码结束 ###
    
    return W


# **运行测试：**

# 计算岭回归系数$w$的值：

# In[20]:


ridge_model(X, Y, alpha)


# **期望输出：**

# ```
# matrix([[ 1.42278923,  2.20583559, -0.6391644 ,  0.64022529, -0.44014758,
#           1.66307858, -0.83879894, -0.25611354, -0.06951638, -2.56882017]])
# ```

# 我们可以看到，和预想的一致，两种方法计算出的 $w$ 系数值是一模一样的。

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
