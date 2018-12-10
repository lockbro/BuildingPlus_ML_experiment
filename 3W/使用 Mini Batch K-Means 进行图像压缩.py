
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # 使用 Mini Batch K-Means 进行图像压缩

# ---

# ### 挑战介绍
#
# 本次挑战将针对一张成都著名景点：锦里的图片，通过 Mini Batch K-Means 的方法将相近的像素点聚合后用同一像素点代替，以达到图像压缩的效果。

# ### 挑战知识点

# - 图像处理
# - Mini Batch K-Means 图像聚类

# ---

# ## 挑战内容

# ### 图像导入

# 首先，我们下载并导入示例图片，图片名为 `challenge-7-chengdu.png`。

# In[1]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

get_ipython().system(
    'wget -nc http://labfile.oss.aliyuncs.com/courses/1081/challenge-7-chengdu.png')


# 使用 Matplotlib 可视化示例图片。

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


chengdu = mpimg.imread('challenge-7-chengdu.png')  # 将图片加载为 ndarray 数组
plt.imshow(chengdu)  # 将数组还原成图像


# In[3]:


chengdu.shape


# 在使用 `mpimg.imread` 函数读取图片后，实际上返回的是一个 `numpy.array` 类型的数组，该数组表示的是一个像素点的矩阵，包含长，宽，高三个要素。如成都锦里这张图片，总共包含了 $516$ 行，$819$ 列共 $516*819=422604$ 个像素点，每一个像素点的高度对应着计算机颜色中的三原色 $R, G, B$（红，绿，蓝），共 3 个要素构成。

# ### 数据预处理

# 为方便后期的数据处理，需要对数据进行降维。

# ---

# **<font color='red'>挑战</font>：将形状为 $(516, 819, 3)$ 的数据转换为 $(422604, 3)$ 形状的数据。**

# **<font color='green'>提示</font>**：使用 `np.reshape` 进行数据格式的变换。

# In[5]:


"""数据格式变换
"""
data = chengdu.reshape(422604, 3)


# **运行测试：**

# In[8]:


data.shape, data[10]


# **期望输出：**

# <center>`((422604, 3), array([0.12941177, 0.13333334, 0.14901961], dtype=float32))`</center>

# ### 像素点种类个数计算

# 尽管有 `422604` 个像素点，但其中仍然有许多相同的像素点。在此我们定义：$R, G, B$ 值相同的点为一个种类，其中任意值不同的点为不同种类。

# ---

# **<font color='red'>挑战</font>：计算 `422604` 个像素点中种类的个数。 **

# **<font color='green'>提示</font>**：提供一个思路：可以将数据转化为 list 类型，然后将每一个元素转换为 tuple 类型，最后利用 set() 和 len() 函数进行计算。也可以按照自己的想法完成。
# 如果要用集合的话，对象必须是元组
# In[18]:


data.shape, data[10]


# In[19]:


"""计算像素点种类个数
"""


def get_variety(data):
    """
    参数:
    预处理后像素点集合

    返回:
    num_variety -- 像素点种类个数
    """

    # 代码开始 ### (≈ 3 行代码)
    new_data = []
    for i in range(len(data)):
        new_data.append(tuple(data[i]))
    final_data = set(new_data)
    num_variety = len(final_data)
    ### 代码结束 ###

    return num_variety


# **运行测试：**

# In[20]:


get_variety(data), data[20]


# **期望输出：**

# <center>`(100109, array([0.24705882, 0.23529412, 0.2627451 ], dtype=float32))`</center>

# ### Mini Batch K-Means 聚类

# 像素点种类的数量是决定图片大小的主要因素之一，在此使用 Mini Batch K-Means 的方式将图片的像素点进行聚类，将相似的像素点用同一像素点值来代替，从而降低像素点种类的数量，以达到压缩图片的效果。

# ---

# **<font color='red'>挑战</font>：使用 Mini Batch K-Means 聚类方法对像素点进行聚类，并用每一个中心的像素点代替属于该类别的像素点。 **

# **<font color='brown'>要求</font>**：聚类簇数量设置为 10 类。

# **<font color='green'>提示</font>**：使用 `MiniBatchKMeans` 中 `fit()` 和 `predict()` 函数进行聚类，使用 `cluster_centers_()`函数进行替换，阅读 [官方文档](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)，本次挑战基本使用默认参数。

# In[29]:


# 代码开始 ###（≈ 4 行代码）
model = MiniBatchKMeans(n_clusters=10)
model.fit(data)
predict = model.predict(data)
### 代码结束 ###

# 有多少个聚类中心？
new_colors = model.cluster_centers_[predict]


# **运行测试：**

# In[30]:


# 调用前面实现计算像素点种类的函数，计算像素点更新后种类的个数
get_variety(new_colors)


# **期望输出：**

# <center>`10`</center>

# ### 图像压缩前后展示

# ---

# **<font color='red'>挑战</font>：将聚类后并替换为类别中心点值的像素点，变换为数据处理前的格式，并绘制出图片进行对比展示。 **

# **<font color='green'>提示</font>**：使用 `reshape()` 函数进行格式变换，使用 `imshow()`函数进行绘图。

# In[50]:


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 代码开始 ###（≈ 3 行代码）
new_chengdu = new_colors.reshape(chengdu.shape)
ax[0].imshow(data.reshape(516, 819, 3))
ax[1].imshow(new_chengdu)
### 代码结束 ###


# **运行测试：**

# **期望输出：**

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1531805600911.png)

# 通过图片对比，可以十分容易发现画质被压缩了。其实，因为使用了聚类，压缩后的图片颜色就变为了 10 种。

# 接下来，使用 `mpimg.imsave()` 函数将压缩好的文件进行存储，并对比压缩前后图像的体积变化。

# In[51]:


# 运行对比
mpimg.imsave("new_chengdu.png", new_chengdu)
get_ipython().system('du -h new_chengdu.png')
get_ipython().system('du -h challenge-7-chengdu.png')


# 可以看到，使用 Mini Batch K-Means 聚类方法对图像压缩之后，体积明显缩小。

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
