
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # 小麦种子层次聚类及绘制二叉树

# ---

# ### 挑战介绍

# 本次挑战将针对小麦种子数据集进行层次聚类，并绘制层次聚类二叉树图像。

# ### 挑战知识点

# - 层次聚类
# - 层次聚类二叉树图像

# ---

# ## 挑战内容

# ### 数据集介绍

# 本次挑战将用的小麦种子数据集，该数据集由若干小麦种子的几何参数组成，共包含有 7 个维度。这些维度有：种子面积、种子周长、种子致密度、核仁长度、核仁宽度、种子不对称系数、核沟长度。

# 你可以下载并加载预览该数据集：

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

get_ipython().system(
    'wget -nc http://labfile.oss.aliyuncs.com/courses/1081/challenge-8-seeds.csv')


# In[2]:


df = pd.read_csv("challenge-8-seeds.csv")
df.head()


# 可以看到，数据集从 f1-f7 代表 7 种特征。下面，我就要通过层次聚类方法完成对该种子数据集的聚类，从而估计出数据集到底采集了几种类别的小麦种子。

# ### 层次聚类

# 前面的实验中，我们学习了如何实现一个自底向上的层次聚类算法，并了解通过 scikit-learn 完成层次聚类。这次的挑战中，我们将尝试通过 Scipy 完成，Scipy 作为知名的科学计算模块也同样提供了层次聚类的方法。

# ---

# **<font color='red'>挑战</font>：使用 Scipy 中的 Agglomerative 聚类方法完成小麦种子层次聚类。**

# **<font color='blue'>规定</font>**：使用 `ward` 离差平方和法度量相似度，距离计算使用欧式距离。

# **<font color='green'>提示</font>**：Scipy 中的 Agglomerative 聚类方法类为 `scipy.cluster.hierarchy.linkage()`，阅读[官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)。

# In[15]:


# 代码开始 ### (≈ 1 行代码)
Z = hierarchy.linkage(df, 'ward')
### 代码结束 ###


# **运行测试：**

# In[16]:


Z[:5]


# **期望输出：**

# <div class="output_subarea output_text output_result">
# <pre>array([[1.72000000e+02, 2.06000000e+02, 1.17378192e-01, 2.00000000e+00],
#        [1.48000000e+02, 1.98000000e+02, 1.33858134e-01, 2.00000000e+00],
#        [1.22000000e+02, 1.33000000e+02, 1.35824740e-01, 2.00000000e+00],
#        [7.00000000e+00, 2.80000000e+01, 1.79010642e-01, 2.00000000e+00],
#        [1.37000000e+02, 1.38000000e+02, 1.91444744e-01, 2.00000000e+00]])</pre></div>

# 你会发现，Scipy 中的 linkage 方法会返回一个 Nx4 的矩阵（上面的期望输出为前 5 行）。该矩阵其实包含了每一步合并类别的信息，以第一行举例：

# `[1.72000000e+02, 2.06000000e+02, 1.17378192e-01, 2.00000000e+00]` 表示 `172` 类别和 `206` 类别被合并，当前距离为 `1.17378192e-01` 属于全集合最短距离，合并后类别中包含有 `2` 个数据样本。
#

# 也就是说 Scipy 把整个层次聚类的过程都呈现出来了，这一点对于理解层次聚类是非常有帮助的。除此之外，Scipy 还集成了一个绘制层次聚类二叉树的方法`dendrogram`。接下来，就尝试使用它来绘制出上面聚类的层次树。

# ---

# **<font color='red'>挑战</font>：使用 Scipy 中的 dendrogram 方法绘制小麦种子层次聚类二叉树。**

# **<font color='green'>提示</font>**：Scipy 中绘制层次聚类二叉树的方法为 `scipy.cluster.hierarchy.dendrogram()`，阅读[官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)。本次挑战使用默认参数即可。

# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15, 8))
# 代码开始 ### (≈ 1 行代码)
hierarchy.dendrogram(Z)
### 代码结束 ###
plt.show()


# **期望输出：**

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1531806113660.png)

# 层次聚类二叉树中，$x$ 轴代表数据点原类别，也就是样本序号，而 $y$ 轴表示类别之间的距离。
#
# 特别地，图中的横线所在高度表明类别间合并时的距离。如果相邻两横线的间距越大，则说明前序类别在合并时的距离越远，也就表明可能并不属于一类不需要被合并。
#
# 上图中蓝色线所对应的 $y$ 差值最大，即说明红色和绿色两个分支很有可能不属于一类。

# ### 修剪层次聚类二叉树

# 上面，我们使用 `dendrogram()` 来绘制二叉树。你会发现当样本数量越多时，叶节点就越密集，最终导致通过二叉树辨识不同类别的可视性降低。

# 其实，你可以指定多个参数来修剪完整的二叉树结果，让其具备更好地观赏性。

# ---

# **<font color='red'>挑战</font>：对小麦种子层次聚类二叉树进行修剪。**

# **<font color='green'>提示</font>**：修改参数 `truncate_mode`, `p`, `show_leaf_counts`, `show_contracted`。

# In[31]:


plt.figure(figsize=(15, 8))
# 代码开始 ### (≈ 1 行代码)
hierarchy.dendrogram(Z, truncate_mode='lastp', p=15,
                     show_leaf_counts=True, show_contracted=True)
### 代码结束 ###
plt.show()


# **期望输出：**

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1531806114000.png)

# 此时的二叉树看起来就更美观了。那么，本次挑战中到底判定小麦种子大致为几类呢？下面通过层次聚类二叉树给出建议：

# <img width='700px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1531806114224.png"></img>

# 所以，最终建议将小麦种子数据集划为 3 类，也就是其中包含 3 种不同品种的小麦籽粒。

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
