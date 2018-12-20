
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # Keras 介绍及使用

# ---

# ### 实验介绍

# Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK，或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时长把你的想法转换为实验结果。目前，TensorFlow 已经吸纳 Keras，可以通过 `tf.keras` 调用相关的类，这和使用 Keras 是一致的。当然，本次实验还是以 Keras 模块为主。

# ### 实验知识点
# 
# - Keras 基本介绍
# - 序贯模型的使用
# - 函数模型的介绍
# - Layers 神经网络层
# - 常用模型组件及配置
# - 模型结构可视化

# ### 实验目录
# 
# - <a href="#Keras-本地安装">Keras 本地安装</a>
# - <a href="#Keras-快速上手">Keras 快速上手</a>
# - <a href="#Keras-神经网络层">Keras 神经网络层</a>
# - <a href="#Keras-模型">Keras 模型</a>
# - <a href="#模型结构及可视化">模型结构及可视化</a>
# - <a href="#实验总结">实验总结</a>

# ---

# ## Keras 本地安装

# <img width='400px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532063509969.png"></img>

# 从前面实验介绍中我们知道，Keras 可以建立在 TensorFlow，CNTK, 或者 Theano 后端之上。所以在安装 Keras 之前，需要安装其中一个后端引擎。在实际应用中 TensorFlow 由于应用广泛，所以在选择 Keras 后端引擎时，更推荐安装 TensorFlow。如果安装了 TensorFlow 也可以通过 `tensorflow.keras` 使用 Keras。
# 
# 实验楼的在线环境已经安装好相关的模块，本小节的内容主要是方便想在本地安装的同学。

# ### 后端引擎安装

# 对于后端安装可以通过以下链接来完成：
# 
# - [TensorFlow 官方安装说明](https://www.tensorflow.org/install/)  
# - [Theano 官方安装说明](http://deeplearning.net/software/theano/install.html#install)   
# - [CNTK 官方安装说明](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)

# ### Keras 安装

# #### 使用 PyPI 安装 Keras：

# ```
# !sudo pip install keras
# ```

# #### 使用 Github 源码安装 Keras：

# 首先，使用 git 来克隆 Keras 仓库：

# ```
# !git clone https://github.com/keras-team/keras.git
# ```

# 然后，切换到 Keras 目录并且运行安装命令：

# ```
# !cd keras
# !sudo python setup.py install
# ```

# ### 安装可选依赖包

# - `cuDNN`：如果你计划在 GPU 上运行 Keras，建议安装。
# - `HDF5` 和 `h5py`：如果你需要将 Keras 模型保存到磁盘，则需要这些。
# - `graphviz` 和 `pydot`：用于可视化工具绘制模型图。

# ## Keras 快速上手

# 在详细了解 Keras 之前，我们首先通过构建一个简单的神经网络来对 Keras 进行简单介绍。Keras 的核心数据结构是 model，一种组织网络层的方式。其中有两种模式，最简单且最常用的模型是 Sequential 序贯模型，它是由多个网络层线性堆叠的栈。对于更复杂的结构，可以使用 Keras 函数式 API，它允许构建任意的神经网络图。

# 首先，生成一组示例数据。

# In[ ]:


import numpy as np

x_train = np.random.random((100, 100))
y_train = np.random.randint(2, size=(100, 10))
x_test = np.random.random((100, 100))
y_test = np.random.randint(2, size=(100, 10))


# **☞ 动手练习：**

# 接下来，使用 Keras 创建一个 Sequential 序贯模型。运行后会出现 `Using TensorFlow backend.`，代表正在使用 TensorFlow 作为后端。

# In[ ]:


import keras
from keras.models import Sequential

model = Sequential() # 声明序贯模型


# 然后，我们通过使用 `.add()` 来堆叠模型，也就是向模型添加不同的神经网络层。这里添加 `Dense` 全连接层，并实现如下所示的网络结构。

# <img width='700px' style="border:2px solid #888;" src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532063510383.png"></img>

# In[ ]:


from keras.layers import Dense

model.add(Dense(units=100, activation='relu', input_dim=100)) # 100 个神经元全连接层，relu 激活
model.add(Dense(units=10))


# 上面就已经完成了模型的构建工作。接下来，就可以使用 `.compile()` 来编译模型，编译时需要配置损失函数和优化器。这也是使用 Keras 的方便之处，一步到位！

# In[ ]:


# 交叉熵损失函数 + 随机梯度下降 + 准确度评估
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# 接下来就是训练过程，你可以在训练数据上进行批量迭代了：

# In[ ]:


# x_train 和 y_train 是 Numpy 数组

model.fit(x_train, y_train, batch_size=32)


# 可以看到，Keras 会生成一个漂亮的训练过程，你可以直接看到损失和准确率。如果使用 TensorFlow，这一切都需要自己搞定。接下来，我们对测试数据进行预测：

# In[ ]:


classes = model.predict(x_test, batch_size=128)
classes[:5]


# 以上是对 Keras 的一个简单入门，虽然只有短短的几行代码，但是已经实现了一个神经网络。相比于 TensorFlow 的复杂过程，Keras 就会简洁很多，这也是大家喜爱 Keras 的直接原因。

# 事实上，构建一个问答系统，一个图像分类模型，一个神经图灵机，或者其他的任何模型，使用 Keras 就是这么快。深度学习背后的思想很简单，那么它们的实现又何必要那么痛苦呢？

# ## Keras 神经网络层

# <div style="color: #999;font-size: 12px;font-style: italic;">* 本小节内容节选自 [Keras 官方中文文档](https://keras.io/zh/)，以便呈现出最权威的模块介绍。</div>

# ### 关于 Keras 层

# 所有 Keras 层都有很多共同的函数：
# 
# - `layer.get_weights()`: 以 NumPy 矩阵的形式返回层的权重。
# - `layer.set_weights(weights)`: 从 NumPy 矩阵中设置层的权重（与 `get_weights` 的输出形状相同）。
# - `layer.get_config()`: 返回包含层配置的字典。
# 
# 例如，我们获取层的配置：

# In[ ]:


layer = Dense(32)
config = layer.get_config()
config


# 如果一个层具有单个节点（不是共享层）, 你可以得到它的输入张量，输出张量，输入尺寸和输出尺寸:
# 
# - `layer.input`
# - `layer.output`
# - `layer.input_shape`
# - `layer.output_shape`

# 如果层有多个节点 , 可以使用以下函数:
# 
# - `layer.get_input_at(node_index)`
# - `layer.get_output_at(node_index)`
# - `layer.get_input_shape_at(node_index)`
# - `layer.get_output_shape_at(node_index)`

# ### 全连接层

# #### 基本概念

# ```python
# Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```

# __参数__
# 
# - __units__: 正整数，输出空间维度。
# - __activation__: 激活函数。
# 若不指定，则不使用激活函数
# (即， “线性”激活: `a(x) = x`)。
# - __use_bias__: 布尔值，该层是否使用偏置向量。
# - __kernel_initializer__: `kernel` 权值矩阵的初始化器。
# - __bias_initializer__: 偏置向量的初始化器.
# - __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数。
# - __bias_regularizer__: 运用到偏置向的正则化函数。
# - __activity_regularizer__: 运用到层的输出的正则化函数。
# - __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数。
# - __bias_constraint__: 运用到偏置向量的约束函数。

# Dense 实现以下操作： `output = activation(dot(input, kernel) + bias)` 其中 `activation` 是按逐个元素计算的激活函数，`kernel` 是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置向量 (只在 `use_bias` 为 `True` 时才有用)。

# 注意: 如果该层的输入的秩大于 `2`，那么它首先被展平然后 再计算与 `kernel` 的点乘。例如：

# In[ ]:


# 作为 Sequential 模型的第一层
model = Sequential()

# 现在模型就会以尺寸为 (*, 16) 的数组作为输入
model.add(Dense(32, input_shape=(16,)))

# 其输出数组的尺寸为 (*, 32)
# 在第一层之后，你就不再需要指定输入的尺寸了：
model.add(Dense(32))


# #### Activation

# ```python
# keras.layers.Activation(activation)
# ```

# __参数__
# 
# - __activation__: 要使用的激活函数的名称。

# #### Dropout 

# 在神经网络中随着层数的增加，特别容易出现过拟合现象。那么如何避免过拟合呢？最好的方法就是 Dropout ，在每一层进行计算的时候，随机断开一部分神经元，这样就可以比较好的解决过拟合。

# ```python
# keras.layers.Dropout(rate, noise_shape=None, seed=None)
# ```

# __参数__
# 
# - __rate__: 在 0 和 1 之间浮动。需要丢弃的输入比例。
# - __noise_shape__: 1D 整数张量，
# 表示将与输入相乘的二进制 dropout 掩层的形状。
# 例如，如果你的输入尺寸为
# `(batch_size, timesteps, features)`，然后
# 你希望 dropout 掩层在所有时间步都是一样的，
# 你可以使用 `noise_shape=(batch_size, 1, features)`。
# - __seed__: 一个作为随机种子的 Python 整数。

# #### Flatten

# 将输入展平但不影响批量大小。

# ```python
# keras.layers.Flatten()
# ```

# 例如：

# In[ ]:


from keras.layers import Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (3, 3), data_format='channels_first', input_shape=(3, 32, 32), padding='same'))
# 现在：model.output_shape == (None, 64, 32, 32)
print(model.output_shape)

# 现在：model.output_shape == (None, 65536)
model.add(Flatten())
model.output_shape


# #### Reshape 

# 将输入重新调整为特定的尺寸

# ```python
# keras.layers.Reshape(target_shape)
# ```

# __参数__
# 
# - __target_shape__: 目标尺寸。整数元组。
# 不包含表示批量的轴。

# In[ ]:


from keras.layers import Reshape

# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# 现在：model.output_shape == (None, 3, 4)
print(model.output_shape)
# 注意： `None` 是批表示的维度

# 作为 Sequential 模型的中间层
model.add(Reshape((6, 2)))
# 现在： model.output_shape == (None, 6, 2)
print(model.output_shape)

# 还支持使用 `-1` 表示维度的尺寸推断
model.add(Reshape((-1, 2, 2)))
# 现在： model.output_shape == (None, 3, 2, 2)
model.output_shape


# #### Permute 

# 根据给定的模式置换输入的维度。在某些场景下很有用，例如将 RNN 和 CNN 连接在一起。

# ```python
# keras.layers.Permute(dims)
# ```

# __参数__
# 
# - __dims__: 整数元组。置换模式，不包含样本维度。
# 索引从 1 开始。
# 例如, `(2, 1)` 置换输入的第一和第二个维度。

# In[ ]:


from keras.layers import Permute

model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# 现在： model.output_shape == (None, 64, 10)
model.output_shape


# ### 卷积层

# 在卷积神经网络中，卷积层是最为核心的概念，下面拿 Conv1D 进行举例：

# ```python
# keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```

# 当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组或 `None`）。例如， `(10, 128)` 表示 `10` 个 `128` 维的向量组成的向量序列，`(None, 128)` 表示 `128` 维的向量组成的变长序列。

# __参数__
# 
# - __filters__: 整数，输出空间的维度
# （即卷积中滤波器的输出数量）。
# - __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
# 指明 1D 卷积窗口的长度。
# - __strides__: 一个整数，或者单个整数表示的元组或列表，
# 指明卷积的步长。
# 指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
# - __padding__: `"valid"`, `"causal"` 或 `"same"` 之一 (大小写敏感)
# `"valid"` 表示「不填充」。
# `"same"` 表示填充输入以使输出具有与原始输入相同的长度。
# `"causal"` 表示因果（膨胀）卷积。
# - __dilation_rate__: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。
# 当前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
# - __activation__: 要使用的激活函数。
# 如果你不指定，则不使用激活函数
# (即线性激活： `a(x) = x`)。
# - __use_bias__: 布尔值，该层是否使用偏置向量。
# - __kernel_initializer__: `kernel` 权值矩阵的初始化器。
# - __bias_initializer__: 偏置向量的初始化器。
# - __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数。
# - __bias_regularizer__: 运用到偏置向量的正则化函数。
# - __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数。
# - __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数。
# - __bias_constraint__: 运用到偏置向量的约束函数。

# ### 池化层

# #### MaxPooling1D
# 对于时序数据的最大池化。

# ```python
# keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
# ```

# __参数__
# 
# - __pool_size__: 整数，最大池化的窗口大小。
# - __strides__: 整数，或者是 `None`。作为缩小比例的因数。
# 例如，2 会使得输入张量缩小一半。
# 如果是 `None`，那么默认值是 `pool_size`。
# - __padding__: `"valid"` 或者 `"same"` （区分大小写）。

# #### MaxPooling2D
# 对于空域数据的最大池化。

# ```python
# keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# ```

# __参数__
# 
# - __pool_size__: 整数，或者 2 个整数元组，（垂直方向，水平方向）缩小比例的因数。（2，2）会把输入张量的两个维度都缩小一半。
# 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
# - __strides__: 整数，整数元组或者是 `None`。
# 步长值。
# 如果是 `None`，那么默认值是 `pool_size`。
# - __padding__: `"valid"` 或者 `"same"` （区分大小写）。
# - __data_format__: 一个字符串，`channels_last` （默认值）或者 `channels_first`。
# 输入张量中的维度顺序。
# `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
# 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
# 如果还没有设置过，那么默认值就是 "channels_last"。

# #### 3.3.3 AveragePooling1D
# 对于时序数据的平均池化。

# ```python 
# keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
# ```

# __参数__
# 
# - __pool_size__: 整数，平均池化的窗口大小。
# - __strides__: 整数，或者是 `None	`。作为缩小比例的因数。
# 例如，2 会使得输入张量缩小一半。
# 如果是 `None`，那么默认值是 `pool_size`。
# - __padding__: `"valid"` 或者 `"same"` （区分大小写）。

# #### 3.3.4 AveragePooling2D
# 对于空域数据的平均池化。

# ```python
# keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# ```

# __参数__
# 
# - __pool_size__: 整数，或者 2 个整数元组，（垂直方向，水平方向）缩小比例的因数。（2，2）会把输入张量的两个维度都缩小一半。
# 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
# - __strides__: 整数，整数元组或者是 `None`。
# 步长值。
# 如果是 `None`，那么默认值是 `pool_size`。
# - __padding__: `"valid"` 或者 `"same"` （区分大小写）。
# - __data_format__: 一个字符串，`channels_last` （默认值）或者 `channels_first`。
# 输入张量中的维度顺序。
# `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
# 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
# 如果还没有设置过，那么默认值就是 "channels_last"。

# ### 循环层

# #### RNN

# ```python
# keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
# ```

# __参数__
# 
# - __cell__: 一个 RNN 单元实例。RNN 单元是一个具有以下项目的类：
# - 一个 `call(input_at_t, states_at_t)` 方法，
# 它返回 `(output_at_t, states_at_t_plus_1)`。
# 单元的调用方法也可以采用可选参数 `constants`，
# 详见下面的小节 "关于传递外部常量的注意事项"。
# - 一个 `state_size` 属性。这可以是单个整数（单个状态），
# 在这种情况下，它是循环层状态的大小（应该与单元输出的大小相同）。
# 这也可以是整数的列表/元组（每个状态一个大小）。
# 在这种情况下，第一项（`state_size [0]`）应该与单元输出的大小相同。
# `cell` 也可能是 RNN 单元实例的列表，在这种情况下，RNN 的单元将堆叠在另一个单元上，实现高效的堆叠 RNN。
# - __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
# - __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
# - __go_backwards__: 布尔值 (默认 False)。
# 如果为 True，则向后处理输入序列并返回相反的序列。
# - __stateful__: 布尔值 (默认 False)。
# 如果为 True，则批次中索引 i 处的每个样品的最后状态将用作下一批次中索引 i 样品的初始状态。
# - __unroll__: 布尔值 (默认 False)。
# 如果为 True，则网络将展开，否则将使用符号循环。
# 展开可以加速 RNN，但它往往会占用更多的内存。
# 展开只适用于短序列。
# - __input_dim__: 输入的维度（整数）。
# 将此层用作模型中的第一层时，此参数（或者，关键字参数 `input_shape`）是必需的。
# - __input_length__: 输入序列的长度，在恒定时指定。
# 如果你要在上游连接 `Flatten` 和 `Dense` 层，
# 则需要此参数（如果没有它，无法计算全连接输出的尺寸）。
# 请注意，如果循环神经网络层不是模型中的第一层，
# 则需要在第一层的层级指定输入长度（例如，通过 `input_shape` 参数）。

# 上面，我们对常用的一些神经网络层进行了简单介绍。实际使用时，我们都会通过查阅官网文档来了解最新、最详细的 API。非常高兴，Keras 提供了全面的中文文档，这样就给许多人扫除了英文的门槛。

# ## Keras 模型

# 在 Keras 中有两类模型：序贯（Sequential）模型和函数模型。

# ### 序贯模型

# 序贯模型是多个网络层的线性堆叠，通俗来讲就是「一条路走到黑」。下面我们通过线性回归模型来对序贯模型有一个直观的理解。

# #### 生成数据

# 首先，我们生成 100 组数据，并用 `matplotlib` 绘制出来。

# In[ ]:


import numpy as np  
   
from keras.models import Sequential  
from keras.layers import Dense  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(520) # 定义随机种子
X = np.linspace(-1, 1, 100) #再返回（-1, 1）范围内的等差序列  
np.random.shuffle(X)    # 打乱顺序  
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (100, )) # 生成 Y 并添加噪声  

plt.scatter(X, Y)  
plt.show()  


# 在这 100 组数据中根据定义可知： $weihgt=0.5$, $bias=2$。

# #### 数据集划分

# 接下来，将生成数据划分为测试集（$30\%$）和训练集（$70\%$）：

# In[ ]:


X_train, Y_train = X[:70], Y[:70]     # 前 70 组数据为训练数据集  
X_test, Y_test = X[70:], Y[70:]      # 后 30 组数据为测试数据集  
X_train[:5]


# #### 构建神经网络

# 现在，我们使用序贯模型构建神经网络。

# In[ ]:


model = Sequential()   
model.add(Dense(input_dim=1, units=1)) # 全连接层，也就是一个线性变换

# 选定 loss 函数和优化器  
model.compile(loss='mse', optimizer='sgd')


# #### 模型训练

# 构建好了神经网络，接下来对数据进行训练：

# In[ ]:


model.fit(X_train, Y_train)


# 上面的训练过程只进行了一次迭代，所以最终的效果不一定很好。如果想得到较优的参数，可以使用 `model.fit(X_train, Y_train, epochs=150)` 来进行训练。

# #### 数据测试

# 训练数据完成后，我们可以对数据进行测试：

# In[ ]:


model.evaluate(X_test, Y_test, batch_size=40)  


# 同时，可以输出拟合参数，前面是 $W$，后面是截距项：

# In[ ]:


model.layers[0].get_weights()  


# 上面就是使用 Keras 序贯模型完成线性回归任务的过程，非常简单。

# ### 函数模型

# 和序贯模型最大的不同在于，函数模型可以通过多输入多输出的方式。并且所有的模型都是可调用的，就像层一样利用函数式模型的接口，我们可以很容易的重用已经训练好的模型。
# 
# 你可以把模型当作一个层一样，通过提供一个 Tensor 来调用它。注意当你调用一个模型时，你不仅仅重用了它的结构，也重用了它的权重。使用函数式模型的一个典型场景是搭建多输入、多输出的模型。

# 关于函数模型的使用，在后面的实验中，使用到的时候再详细说明。

# ## 模型结构及可视化

# Keras 提供了让我们方便直观认识模型结构的方法，我们先定义一个简单模型：

# In[ ]:


model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 此时，你可以直接使用 `model.summary()` 查看模型结构：

# In[ ]:


model.summary()


# 除此之外，`keras.utils.vis_utils` 模块提供了绘制 Keras 模型的实用功能（需要安装 `graphviz` 和 `pydot`）。

# In[ ]:


get_ipython().system('pip install pydot')


# <div style="color: red;font-size: 12px;font-style: italic;">* 执行完上面的安装步骤后需要重启内核才能正常执行下面的绘图操作，重启方法：P → confirm restart kernel and clear output。</div>

# In[ ]:


from keras.utils.vis_utils import plot_model
from IPython.display import Image

plot_model(model, to_file="model.png", show_shapes=True)
Image('model.png')


# `plot_model` 有两个可选参数:
# 
# - `show_shapes` (默认为 `False`) 控制是否在图中输出各层的尺寸。
# - `show_layer_names` (默认为 `True`) 控制是否在图中显示每一层的名字。

# ## 实验总结

# 本次实验中，我们了解到了 Keras 的一些基础用法，并通过两个例子构建了简单的序贯模型。你会发现使用 Keras 构建模型的过程比 TensorFlow 要简单很多。除此之外，我们初步了解了 Keras 常用的神经网络层以及其他组件，名词和参数与 TensorFlow 非常接近。一般情况下，我们都会在实际使用时，通过阅读官方文档来更加深入的了解每个参数的用法和用途。回归本次实验的知识点有：
# 
# - Keras 基本介绍
# - 序贯模型的使用
# - 函数模型的介绍
# - Layers 神经网络层
# - 常用模型组件及配置
# - 模型结构可视化

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
