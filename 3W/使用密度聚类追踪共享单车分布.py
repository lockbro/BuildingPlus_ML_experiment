
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # 使用密度聚类追踪共享单车分布

# ---

# ### 挑战介绍

# 本次挑战将考察密度聚类的应用。

# ### 挑战知识点

# - DBSCAN 参数确定
# - HDBSCAN 聚类

# ---

# ## 挑战内容

# 如今，共享单车已经遍布大街小巷，的确方便了市民的短距离出行。不过，如果你是一家共享单车公司的运营，是否会考虑这样一个问题，那就是**公司投放到城市中的共享单车都去哪里了呢？**

# 当然，这个问题并不是为了满足你的好奇心，而是通过追踪共享单车的分布状况及时调整运营策略。比如，有一些位置的单车密度过高，那么就应该考虑将其移动到一些密度低但有需求的区域。

# 所以，今天的挑战中，将会使用到密度聚类方法来追踪共享单车的分布情况。

# ### 数据集介绍

# 我们获取到北京市某一区域的共享单车 GPS 散点数据集，该数据集名称为 `challenge-9-bike.csv`。首先，下载并预览该数据集。

# In[1]:


get_ipython().system('wget -nc http://labfile.oss.aliyuncs.com/courses/1081/challenge-9-bike.csv')


# In[2]:


import pandas as pd
import numpy as np

df = pd.read_csv("challenge-9-bike.csv")
df.describe()


# 其中，`lat` 是 latitude 的缩写，表示纬度，`lon` 是 longitude 的缩写，表示经度。于是，我们就可以通过 Matplotlib 绘制出该区域共享单车的分布情况。

# In[3]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.figure(figsize=(15,8))
plt.scatter(df['lat'], df['lon'], alpha=.6)


# 我们可以使用第三方工具将对应的经纬度坐标放置到地图中呈现。

# In[ ]:


from IPython.display import IFrame

IFrame('https://geohey.com/apps/dataviz/58cb59a4012a473e8b32207920e6993e/share?ak=ZDY2MTZkMzY4YzM0NGY3YmFhZmNhYzM3YTU1ZmY5Zjg', width=900, height=600)


# 接下来，我们尝试使用 DBSCAN 密度聚类算法对共享单车进行聚类，看一看共享单车高密度区域的分布情况。(可能会失效，对挑战无影响)

# 根据前一节实验课程可知，DBSCAN 算法的两个关键参数是 `eps` 和密度阈值 `MinPts`。那么这两个值设定为多少比较合适呢？

# ---

# **<font color='red'>挑战</font>：使用 DBSCAN 算法完成共享单车 GPS 散点数据密度聚类，需要确定 `eps` 和 `min_samples` 参数。**

# **<font color='blue'>规定</font>**：假设半径 `100` 米范围内有 `10` 辆车为高密度区域。

# **<font color='green'>提示</font>**：挑战以纬度变化为参考，初略估算纬度变化 `1` 度，对应该区域 `100km` 的地面距离。

# In[7]:


from sklearn.cluster import DBSCAN

### 代码开始 ### (≈ 2 行代码)
dbscan_m = DBSCAN(eps=0.001, min_samples=10)
dbscan_c = dbscan_m.fit_predict(df)
### 代码结束 ###
dbscan_c # 输出聚类标签


# **运行测试：**

# In[6]:


np.mean(dbscan_c)


# **期望输出：**

# <center>**`6.977333333333333`**</center>

# ---

# **<font color='red'>挑战</font>：针对上面聚类后数据，按要求重新绘制散点图。**

# **<font color='blue'>规定</font>**：未被聚类的异常点以 `alpha=0.1` 蓝色数据点呈现，聚类数据按类别呈现且设置 `cmap='viridis'`。

# In[17]:


### 代码开始 ### (≈ 4~8 行代码)
plt.figure(figsize=(15,8))

df_c = pd.concat([df, pd.DataFrame(dbscan_c, columns=['cluster'])], axis=1)

df_good = df_c[df_c['cluster'] != -1]
df_bad = df_c[df_c['cluster'] == -1]

plt.scatter(df_good['lat'], df_good['lon'], c=df_good['cluster'], cmap='viridis')
plt.scatter(df_bad['lat'], df_bad['lon'], alpha=.1, c ='b')
### 代码结束 ###


# **期望输出：**

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1531806489365.png)

# 从上图可以看出，不同区域的单车密度分布情况。

# HDSCAN 算法很多时候不仅仅是完成聚类，由于其本身的特性，很多时候还用其识别异常点。在本次实验中，我们同样可以通过调节参数来识别位置异常的共享单车。

# ---

# **<font color='red'>挑战</font>：针对聚类后数据，将异常点（不符合半径 100 米内有 2 辆共享单车）绘制到散点图。**

# **<font color='blue'>规定</font>**：未被聚类的边界点以红色数据点呈现，聚类数据按类别呈现且设置 `cmap='viridis', alpha=.1`。

# In[21]:


### 代码开始 ### (≈ 6~10 行代码)
plt.figure(figsize=(15, 8))

dbscan_m = DBSCAN(eps=0.001, min_samples=2)
dbscan_c = dbscan_m.fit_predict(df)

df_c = pd.concat([df, pd.DataFrame(dbscan_c, columns=['cluster'])], axis=1)

df_good = df_c[df_c['cluster'] != -1]
df_bad = df_c[df_c['cluster'] == -1]

plt.scatter(df_good['lat'], df_good['lon'], c=df_good['cluster'], cmap='viridis', alpha=.1)
plt.scatter(df_bad['lat'], df_bad['lon'], c='r')
### 代码结束 ###


# **期望输出：**

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1531806489629.png)

# 本次挑战主要是了解了如何快速确定 DBSCAN 初始参数以及使用该算法标记离群点的方法。如果你有兴趣，还可以自行尝试使用 HDBSCAN 聚类，并对比二者的聚类效果。当然，在这之前你需要先使用实验课程中的方法安装 hdbscan 模块。

# In[ ]:


# 自行练习


# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
