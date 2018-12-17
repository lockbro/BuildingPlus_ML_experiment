
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # TensorFlow 介绍及使用

# ---

# ### 实验介绍

# 本次实验开始，我们将正式进入到深度学习的内容。深度学习的关键，其实在于深度神经网络的构建，而如果你从 0 开始自己编程构建一个深度神经网络，那么过程将会是十分复杂的。所以，为了更方便地实现深度学习模型，我们需要掌握一些常见的深度学习框架的使用。目前在整个深度学习社区里，比较流行的框架有 TensorFlow，Keras，Pytorch 等，他们都有各自独特的特点。其中，TensorFlow 因为背靠谷歌 Google 这座大山，再加上庞大的开发者群体，更新和发版速度着实非常快。了解并掌握 TensorFlow 的使用，将使你在搭建深度学习模型时更加得心应手。

# ### 实验知识点

# - TensorFlow 介绍
# - TensorFlow 工作原理
# - 计算流图
# - 张量的类型
# - 启动会话
# - 常量生成
# - 梯度优化器
# - GPU 使用
# - 小批量梯度下降
# - 预训练模型

# ### 实验目录
# 
# - <a href="#TensorFlow-简介">TensorFlow 简介</a>
# - <a href="#TensorFlow-工作原理">TensorFlow 工作原理</a>
# - <a href="#张量-Tensor-和会话-Session">张量 Tensor 和会话 Session</a>
# - <a href="#常用类和方法">常用类和方法</a>
# - <a href="#用-TensorFlow-实现线性回归">用 TensorFlow 实现线性回归</a>
# - <a href="#存储或重启模型">存储或重启模型</a>
# - <a href="#使用-GPU-训练模型">使用 GPU 训练模型</a>
# - <a href="#实验总结">实验总结</a>

# ---

# ## TensorFlow 简介

# TensorFlow 是由谷歌在 2015 年 11 月发布的深度学习开源工具，我们可以用它来快速构建深度神经网络，并训练深度学习模型。运用 TensorFlow 及其他开源框架的主要目的，就是为我们提供一个更利于搭建深度学习网络的模块工具箱，使开发时能够简化代码，最终呈现出的模型更加简洁易懂。

# <img width='300px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532052758481.png"></img>

# 在学习 TensorFlow 之前，我们先来了解一下 TensorFlow 的特点：

# 1. **高度灵活性：**采用数据流图结构，只要计算可以表示为一个数据流图，就可以使用 TensorFlow。
# 2. **可移植性： **在 CPU，GPU，服务器，移动端，云端，Docker 容器上都可以运行。
# 3. **自动求微分： **在 TensorFlow 中，梯度的运算都将基于你输入的模型结构和目标函数自动完成。
# 4. **多语言支持： **提供 Python，C++，Java，Go 接口。
# 5. **优化计算资源：**TensorFlow 允许用户将数据流图上的不同计算元素分配到用户不同设备上（CPU 内核，GPU 显卡），最大化利用你的硬件资源来进行深度学习运算。

# 看完这些特点，简单来讲就是很好、很强大。与此同时，TensorFlow 还在日益发展。所以，如果我们使用 TensorFlow 作为生产平台和科研基础研发，已经非常坚实可靠。  

# 最后，我们来看一下 TensorFlow 的技术栈图：

# <img width='600px' style="border:2px solid #888;" src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532052618695.png"></img>
# 
# <div style="color: #888; font-size: 10px; text-align: right;"><a href="https://www.tensorflow.org/">©️ 图片来源</a></div>

# 最下面的是 TensorFlow 内核，底层 API 是通过 Python，C++，Java，Go 编写，中层仅提供了支持 Python 的主要的 3 类 API 接口，最上层是高度简略的 Estimator 估计器接口。当然，我们在了解时，将重点学习中层和顶层 API 使用。

# ## TensorFlow 工作原理

# 想要了解 TensorFlow 的工作原理，我们需要先知道一个概念，那就是数据流图（Data Flow Graphs）。
# 
# 什么是数据流图？简单来讲就是把一个完整的数值计算过程的重要步骤进行拆解，并形成一张有向图。图中，节点通常代表数学运算（如加减乘除），边表示节点之间的某种联系，它负责传输数据。

# 如下所示，呈现出一张数据流图：

# <img width='400px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532052620003.png"></img>

# 那么，这张数据流图反映出怎样的数值计算过程呢？其实就是下面这个公式：

# $$ result = b\times b - a\times c\times 5$$

# 根据数据流图的思想，我们可以把运算过程划分如下 4 个步骤：

# $$ t_{1} =a\times c $$

# $$ t_{2} = 5\times t_{1} $$

# $$ t_{3} = b\times b $$

# $$ result = t_{3} - t_{2} $$

# 此时，你可能会有一个很大的疑问：**那就是数据流图似乎是把简单的过程复杂化了呢？为什么可以一步到位的公式，硬是要拆开计算呢？** 

# 其实，在 TensorFlow 中使用数据流图的目的，主要是为了将计算表示为独立的指令之间的依赖关系。数据流图可以为并行处理和分布式计算带来优势。节点可以被分配到多个计算设备上，可以异步和并行地执行操作。因为是有向图，所以只有等到之前的节点计算完成后，当前节点才能执行操作。

# 回到我们给出的数据流图，当输入 $a$ , $b$ , $c$ 的值后，模型就会按数据流图计算出结果。我们可以考虑把 $t_{3}$ 的运算放在 CPU 上，把  $t_{1}$ 和 $t_{2}$ 的运算放在 GPU 上，最后再在 CPU 上得到结果，TensorFlow 就是按照这样的思想允许用户将数据流图的运算整合到不同设备上的。值得注意的是，无论如何分配 $t_{2}$ 和 $t_{1}$ 的运算设备, $t_{2}$ 总是要等待 $t_{1}$ 运算结束才能开始运算。

# 深度学习需要大量的算力提供支持，通过数据流图来实现计算过程，可以尽可能地提升计算效率。下面，我们看一下 TensorFlow 官方给出的一张动态的数据流图计算过程：

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532052621658.png)
# 
# <div style="color: #888; font-size: 10px; text-align: right;"><a href="https://www.tensorflow.org/">©️ 图片来源</a></div>

# 你可以看到，输入的数据流陆续到达不同的层，最终通过随机梯度下降 SGD 找到最优参数。

# ## 张量 Tensor 和会话 Session

# 在 TensorFlow 中，有 3 个重要的组件，它们分别是：张量、会话和图。其中，张量 Tensor 是基本要素，也就是数据；图 Graph 就是上面数据流图；会话 Session 是运行数据流图的机制。

# 所以，你应该就明白 TensorFlow 名字的来源了吧，它其实就是反映了张量在数据流图中的执行过程。正如上面的动态图一样，其就是一个典型的 Tensor → Flow 的动态演示。接下来，我们就详细了解这 3 个基本组成构建。

# ### 张量 Tensor

# 张量，如果你第一次听说，一定会感觉到它是一个很厉害的东西吧。他的确很厉害，但是不难理解。看完下面的内容，你应该就明白了。

# 张量的概念贯穿于物理学和数学中，如果你去看它的很多理论描述，可能并不那么浅显易懂。例如，下面有两种关于张量的定义([维基百科](https://zh.wikipedia.org/wiki/%E5%BC%B5%E9%87%8F))：

# - 通常定义张量的物理学或传统数学方法，是把张量看成一个多维数组，当变换坐标或变换基底时，其分量会按照一定规则进行变换，这些规则有两种：即协变或逆变转换。
# - 通常现代数学中的方法，是把张量定义成某个矢量空间或其对偶空间上的多重线性映射，这矢量空间在需要引入基底之前不固定任何坐标系统。例如协变矢量，可以描述为 1-形式，或者作为逆变矢量的对偶空间的元素。

# 不知道你看懂了没有？如果没有看懂，就来看一下我们的简洁版介绍：

# 你应该知道什么是向量和矩阵。先前的介绍中，我们把 `1` 维的数组称之为向量，`2` 维的数组称之为矩阵。那么，现在告诉你张量其实代表着更大的范围，你也可以把其看作是 `n` 维数组。
# 
# 所以，如果现在重新描述向量和矩阵，就可以是：一阶张量为向量，二阶张量为矩阵。当然，零阶张量也就是标量，而更重要的是 $n$ 阶张量，也就是 `n` 维数组。

# | 阶 | 数学实例 |
# |:--:|:--------------------:|
# | 0 | 标量（只有大小） |
# | 1 | 矢量（大小和方向） |
# | 2 | 矩阵（数据表） |
# | 3 | 3 阶张量（数据立体） |
# | n | n 阶张量（自行想象） |

# <img width='800px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532052623486.png"></img>

# 所以，张量并不是什么晦涩难懂的概念。如果不严谨的讲，**张量就是 `n` 维数组**。前面提到的向量、矩阵，也是张量。

# 在 TensorFlow 中，每一个 Tensor 都具备两个基础属性：**数据类型（默认：float32）和形状**。
# 
# 其中，数据类型大致如下表所示：

# | Tensor 类型 | 描述 |
# |:------------:|:-------------------:|
# | `tf.float32` | 32 位浮点数 |
# | `tf.float64` | 64 位浮点数 |
# | `tf.int64` | 64 位有符号整型 |
# | `tf.int32` | 32 位有符号整型 |
# | `tf.int16` | 16 位有符号整型 |
# | `tf.int8` | 8 位有符号整型 |
# | `tf.uint8` | 8 位无符号整型 |
# | `tf.string` | 可变长度的字节数组 |
# | `tf.bool` | 布尔型 |
# | `tf.complex64` | 实数和虚数 |

# 另外，TensorFlow 通过三种符号约定来描述张量维度：阶，形状和维数。三者之间的关系如下：

# | 形状 | 阶 | 维数 | 示例 |
# |:-----------------------------:|:--:|:----:|:----------------------------------:|
# | [] | 0 | 0-D | 0 维张量。标量。 |
# | [$D_0$] | 1 | 1-D | 形状为 [5] 的 1 维张量。 |
# | [$D_0$, $D_1$] | 2 | 2-D | 形状为 [3, 4] 的 2 维张量。 |
# | [$D_0$, $D_1$, $D_2$] | 3 | 3-D | 形状为 [1, 4, 3] 的 3 维张量。 |
# | [$D_0$, $D_1$, ... $D_{n-1}$] | 4 | n-D | 形状为 [$D_0$, $D_1$, ... $D_{n-1}$] 的张量。 |

# 根据不同的用途，TensorFlow 中的张量大致有 4 种类型，分别是：
# 
# - `tf.Variable`：变量 Tensor，需要指定初始值，常用于定义可变参数。
# - `tf.constant`：常量 Tensor，需要指定初始值，定义不变化的张量。
# - `tf.placeholder`：占位 Tensor，不必指定初始值，可在运行时传入数值。
# - `tf.SparseTensor`：稀疏 Tensor，不常用。

# 除 `tf.Variable` 以外，其余类型张量的值不可变，这意味着在单一执行的情况下，张量只有一个值。

# 我们可以通过传入数组来新建变量和常量类型的张量：

# In[1]:


import tensorflow as tf

tf.Variable([1., 2., 3., 4.])


# **☞ 动手练习：**

# In[2]:


tf.constant([1., 2., 3., 4.])


# 此时，你会发现得到的都是这样的结果：`<tf.Variable 'Variable_1:0' shape=(4,) dtype=float32_ref>`。这是什么意思呢？按照我们的设想，这里不应该输出一个类似于带数值的 Tensor 吗？

# 仔细观察，你会发现这里其实输出了 Tensor 的属性，也就是包含有形状 `shape` 以及数据类型 `dtype`。那么，如果我们想打印 Tensor 的数据该怎么办呢？这就要用到下面的内容。

# ### 会话 Session

# 上面，我们有一个想法，就是得到 Tensor 的值。这里就要生硬地提出 TensorFlow 的一个特点，那就是变量 Tensor 要想输出值或者参与任何运算，都需要得到初始化，并通过会话 Session 机制完成。

# 也就是说，如果我们想得到上面变量 Tensor 的值，就需要这样做：

# In[ ]:


x = tf.Variable([1., 2., 3., 4.])

sess = tf.Session() # 建立会话
sess.run(x.initializer) # 初始化变量
sess.run(x) # 得到变量的值


# In[3]:


x = tf.Variable([1., 2., 3., 4.])

sess = tf.Session()
sess.run(x.initializer)
sess.run(x)


# 对于常量 Tensor 是不需要初始化的，但是建立会话必不可少：

# In[ ]:


x = tf.constant([1., 2., 3., 4.])

sess = tf.Session() # 建立会话
sess.run(x) # 得到变量的值


# In[4]:


x = tf.constant([1., 2., 3., 4.])

sess = tf.Session()
sess.run(x)


# 是不是觉得很麻烦？没有办法，TensorFlow 就是这样设计的，我们就需要这样执行。回到一开始的数据流图，如果我们想要实现计算过程该如何做呢？

# <img width='250px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532052787955.png"></img>

# In[ ]:


a = tf.Variable(3, name="a")
b = tf.Variable(4, name="b")
c = tf.Variable(5, name='c')
f = b * b - a * c * 5
f


# In[5]:


a = tf.Variable(3, name='a')
b = tf.Variable(4, name='b')
c = tf.Variable(5, name='c')
f = b * b - a * c * 5
f


# 不出所料的话，你应该会得到这样的结果：`<tf.Tensor 'sub_1:0' shape=() dtype=int32>`。原因和之前的一样，我们需要对变量进行初始化。同时，为了对已有的数据流图进行赋值运算，需要建立一个 Tensorflow 的会话，并在这个会话里初始化变量以及运算出 f 的结果。
# 
# 本质上，一个会话才能将运算操作分配到像 CPU 和 GPU 这样的硬件设备然后开始执行这个数据流图，并且存储所有的变量数值。所以修改代码：

# In[ ]:


sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
sess.run(c.initializer)
sess.run(f)


# In[6]:


sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
sess.run(c.initializer)
sess.run(f)


# 现在，应该得到 `f` 的计算结果 `-59` 了。注意，在执行完运算后，需要关闭会话，才能把所占空间释放掉。

# In[ ]:


sess.close()


# In[7]:


sess.close()


# 我们在之前的代码里通过调用 3 次 `sess.run()` 去初始化所有变量，另一次是通过 `sess.run()` 运算出 `f` 的值，这样会显得有些累赘。
# 
# 于是为了方便，我们常用的方法是设置一个初始节点，并将全部变量进行初始化。同时，通过 `eval()` 操作来对结果进行运算，也不用单独 `sess.close()`。

# In[ ]:


init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
result


# **☞ 动手练习：**

# In[8]:


init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

result


# 此时，你可能会想，上面的计算图中我们输入的是实数常量，为什么要定义成变量张量而不是常量张量呢？常量与变量的区别是，在未来的运算过程中，变量 Tensor 的数值是可以变化的，如利用梯度下降更新权值。但是，常量 Tensor 则保持不变，如训练数据集。
# 
# 其实，这里如何定义都是没影响的，只是变量张量通常用来定义需要更新的参数，这里不需要更新也无所谓了。当然，我们完全可以将其改为常量张量进行运算。

# In[14]:


a = tf.constant(3, name="a")     
b = tf.constant(4, name="b")
c = tf.constant(5, name='c')
f = b*b - a * c * 5

with tf.Session() as sess:
    result = f.eval()
result


# In[9]:


a = tf.constant(3, name="a")     
b = tf.constant(4, name="b")
c = tf.constant(5, name='c')
f = b*b - a * c * 5

with tf.Session() as sess:
    result = f.eval()

result


# 因为常量不需要初始化，所以还少了一些代码。

# 最后还有一种常用的类型张量，称之为占位符 Tensor `tf.placeholder`。在前面我们看到了常量 Tensor 和变量 Tensor 在创建的时候都会被赋予一个值（对变量来讲是初始值）。但对于一些张量的创建，我们是不知道它的数值的，甚至连该张量的维度都不是很清晰，但可以肯定的是，我们在之后的运算中会对这个张量调用其他的有确定数值的数据來赋值，那么我们可以利用占位符 `tf.placeholder` 来创建这个张量。
# 
# 比如下面这个例子：

# In[ ]:


import numpy as np

x = tf.placeholder(tf.float32, shape=(3, 3)) # 创建占位符张量
y = tf.matmul(x, x) # 乘法计算

with tf.Session() as sess:
    x_test = np.random.rand(3, 3) # 生成 numpy 数组作为测试数据
    result = sess.run(y, feed_dict={x: x_test}) # 通过 feed_dict 把测试数据传给占位符张量
result


# In[16]:


import numpy as np

x = tf.placeholder(tf.float32, shape=(3, 3))
y = tf.matmul(x, x) # 矩阵乘法运算

with tf.Session() as sess:
    x_test = np.random.rand(3, 3)
    result = sess.run(y, feed_dict={x: x_test})
result


# 占位符 Tensor 同样是一种非常重要的张量类型，在构建模型的时候经常会用到它。比如对于一些形式参数，占位符 Tensor 用于定义过程，而真正执行的时候再赋给具体的值。

# ## 常用类和方法

# 学习完上面的内容，相信你已经对 TensorFlow 的基本结果及使用 TensorFlow 进行数值计算的过程有了初步的了解。作为深度学习第一开源框架，TensorFlow 的特点在于快速构建深度学习网络。而快速构建的要点在于，TensorFlow 提供了大量封装好的函数和方法。
# 
# 在构建深度神经网络时，TensorFlow 可以说提供了你一切想要的组件，从不同形状的张量、激活函数、神经网络层，到优化器、数据集等，一应俱全。所以，接下来，我们过一遍 TensorFlow 中常用到的类和方法。这一小节的目的在于，让你对 TensorFlow 整体更加熟悉。

# ### 常量、序列和随机值

# 上面我们已经介绍了常量张量了，这里再列举几个经常会用到的新建特殊常量张量的方法：

# - `tf.zeros`：新建指定形状且全为 0 的常量 Tensor
# - `tf.zeros_like`：参考某种形状，新建全为 0 的常量 Tensor
# - `tf.ones`：新建指定形状且全为 1 的常量 Tensor
# - `tf.ones_like`：参考某种形状，新建全为 1 的常量 Tensor
# - `tf.fill`：新建一个指定形状且全为某个标量值的常量 Tensor

# 为了便于查看输出值，我们先创建 Session：

# In[27]:


sess = tf.Session()

c = tf.zeros([3, 3]) # 3x3 全为 0 的常量 Tensor
sess.run(c)


# In[17]:


sess = tf.Session()

c = tf.zeros([3, 3])
sess.run(c)


# In[ ]:


sess.run(tf.ones_like(c)) # 与 c 形状一致全为 1 的常量 Tensor


# In[18]:


sess.run(tf.ones_like(c))


# In[ ]:


sess.run(tf.fill([2, 3], 6)) # 2x3 全为 6 的常量 Tensor


# In[20]:


sess.run(tf.fill([2, 3], 6))


# 除此之外，我们还可以创建一些序列，例如：

# - `tf.linspace`：创建一个等间隔序列。
# - `tf.range`：创建一个数字序列。

# In[ ]:


sess.run(tf.linspace(1.0, 10.0, 5, name="linspace"))


# In[21]:


sess.run(tf.linspace(1.0, 10.0, 5, name='linspace'))


# In[ ]:


start = 1
limit = 10
delta = 2
sess.run(tf.range(start, limit, delta))


# In[22]:


start = 1
limit = 10
delta = 2
sess.run(tf.range(start, limit, delta))


# 当然，创建随机张量也是日常必不可少的操作，例如：

# - `tf.random_normal`：正态分布中输出随机值。
# - `tf.truncated_normal`：从截断的正态分布中输出随机值。
# - `tf.random_uniform`：从均匀分布中输出随机值。
# - `tf.random_shuffle`：沿着指定维度打乱张量。
# - `tf.random_crop`：随机将张量裁剪为给定的大小。
# - `tf.multinomial`：从多项分布中输出随机值。
# - `tf.set_random_seed`：设定随机数种子。

# In[ ]:


tf.set_random_seed(10)

sess.run(tf.random_normal(
    shape=[5, 5],
    mean=2.0,
    stddev=1.0,
    dtype=tf.float32
))


# In[25]:


tf.set_random_seed(10)

sess.run(tf.random_normal(
    shape=[5, 5],
    mean=2.0,
    stddev=1.0,
    dtype=tf.float32
))


# ### 数学计算

# TensorFlow 中提供的数学计算，包括线性代数计算方面的方法也是应有尽有，十分丰富。下面，我们只是列举几个示例。

# In[ ]:


a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3])
b = tf.constant([7., 8., 9., 10., 11., 12.], shape=[3, 2])

c = tf.matmul(a, b) # 矩阵乘法
sess.run(c) 


# In[29]:


a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3])
b = tf.constant([7., 8., 9., 10., 11., 12.], shape=[3, 2])

c = tf.matmul(a, b)

sess.run(c)


# In[ ]:


sess.run(tf.transpose(c)) # 转置矩阵


# In[30]:


sess.run(tf.transpose(c))


# In[ ]:


sess.run(tf.matrix_inverse(c)) # 求逆矩阵


# In[31]:


sess.run(tf.matrix_inverse(c))


# 由于操作近百个，实在太多太多。一般来讲，除了自己经常使用到的，都会在需要某种运算的时候，查阅 [官方文档](https://www.tensorflow.org/api_docs/python/tf/math)。

# ### 神经网络层

# TensorFlow 作为最出色的深度学习开源框架，在我看来主要有 2 点优势。第一是基于计算图的数值计算过程，最终目的是提升计算速度。其次就是对各种常用神经网络层进行封装，提高搭建模型的速度。

# 另外像创建张量（n 维数组），执行各类数学计算。我们之前接触到的 NumPy 也可以完成，但是上面两点只有使用 TensorFlow 这类专门用于深度学习的开源框架才能做到。所以，下面说一下 TensorFlow 内置的各种神经网络层。这些类基本都在 `tf.nn` 下面。

# 首先是激活函数，我们之前在第二周的内容中提到过一些，但是这里更加全面：
# 
# - `tf.nn.relu`
# - `tf.nn.relu6`
# - `tf.nn.crelu`
# - `tf.nn.elu`
# - `tf.nn.selu`
# - `tf.nn.softplus`
# - `tf.nn.softsign`
# - `tf.nn.dropout`
# - `tf.nn.bias_add`
# - `tf.sigmoid`
# - `tf.tanh`

# 另外，当我们搭建卷积神经网络的时候，会用到大量的卷积层：

# - `tf.nn.convolution`
# - `tf.nn.conv2d`
# - `tf.nn.depthwise_conv2d`
# - `tf.nn.depthwise_conv2d_native`
# - `tf.nn.separable_conv2d`
# - `tf.nn.atrous_conv2d`
# - `tf.nn.atrous_conv2d_transpose`
# - `tf.nn.conv2d_transpose`
# - `tf.nn.conv1d`
# - `tf.nn.conv3d`
# - `tf.nn.conv3d_transpose`
# - `tf.nn.conv2d_backprop_filter`
# - `tf.nn.conv2d_backprop_input`
# - `tf.nn.conv3d_backprop_filter_v2`
# - `tf.nn.depthwise_conv2d_native_backprop_filter`
# - `tf.nn.depthwise_conv2d_native_backprop_input`

# 当然，上面提到的卷积层我们会在卷积神经网络的学习时进行深入讲解。

# TensorFlow 中涉及到的类和方法实在太丰富，通过一节实验课是很难全面覆盖到的。同时，单纯地罗列方法并没有太多的意义。所以，像数据的读取和存储，图像处理，张量转换、优化器等等，我们会在后面使用到的时候再细说，这样也更能够理解和掌握。

# ## 用 TensorFlow 实现线性回归

# 上面的内容中，你已经大致了解 TensorFlow 的会话等机制，以及一些常用组件。接下来，我们尝试用 TensorFlow 去实现一个线性回归。你可能会认为线性回归非常基础，不过这里的目的主要是熟悉 TensorFlow 搭建模型的整个流程，以及诸如 Placeholder，Constant，Optimizer 等重要概念。

# 接下来，我们尝试对加州房价进行预测。我们利用 TensorFlow 搭建了一个线性回归模型的数据流图，并且利用最小二乘法计算出相应的权值 `weight`（代码里用 `theta` 表示）。

# 首先，我们加载示例数据集。这里使用 sklearn 里自带的数据集。

# In[ ]:


import numpy as np
from sklearn.datasets import fetch_california_housing # sklearn 自带数据集

housing = fetch_california_housing()   # 数据集，其中 housing.data 表示特征数据，housing.target 是目标数据
m, n = housing.data.shape  # m, n 存储特征数据的维度，因为最小二乘法里我们需要对特征数据补上一列元素均为1的向量
housing_feature = np.append(housing.data, np.ones((m,1)), axis=1)   # 补 1 操作

housing_feature


# In[28]:


import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_feature = np.append(housing.data, np.ones((m,1)), axis=1)

housing_feature


# ### TensorFlow 实现最小二乘法

# 接下来，我们使用 TensorFlow 实现最小二乘法，这里使用到前面实验中学过的最小二乘法矩阵推导公式。

# In[6]:


# 定义 X 为一个常数张量，值为 housing.feature 的值
X = tf.constant(housing_feature, dtype=tf.float32, name='X')

# reshape（-1,1）将目标数据设定为一个列向量，同样地，我们定义 y 为一个常数张量
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='Y')

# X 的转置
XT = tf.transpose(X)

# tf.matmul 是在 tensorflow 里的矩阵乘法，tf.matrix_inverse 求出矩阵的逆矩阵
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

theta


# In[7]:


X = tf.constant(housing_feature, dtype=tf.float32, name='X')

Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='Y')

XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)

theta


# 上面说过，要执行运算必须通过新建会话来完成。

# In[ ]:


with tf.Session() as sess:      # 建立会话开始运算 theta 的值
    theta_val = theta.eval()  # 由于数据流图里没有变量（X，Y，XT 均为常数张量），这里不需要初始化变量操作。
theta_val


# In[9]:


with tf.Session() as sess:
    theta_val = theta.eval()
theta_val


# 我们已经得到了权重 `theta` 的值。你可能在想，这个过程比用 NumPy 还复杂，毕竟 TensorFlow 是为深度学习设计的，这里用来做线性回归的确有点大材小用了。

# ### TensorFlow 手动实现梯度下降

# 在梯度下降算法里，我们需要初始化 `theta` 的值，然后在每一次迭代过程中，通过梯度下降的原则对 `theta` 值进行更新，所以 `theta` 应该是一个变量。为了使梯度下降运行起来更加顺畅，我们先对数据进行归一化操作。这里直接使用 sklearn 提供的 `StandardScaler()` 方法。

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.append(scaled_housing_data,np.ones((m,1)),axis=1)


# In[31]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.append(scaled_housing_data, np.ones((m,1)), axis=1)


# 上面提到，创建一个常量张量的时候，我们有很多种方式。常用的有 `tf.ones(shape)`，`tf.zeros(shape)` 可以分别创造出一个 `shape` 大小的全 `1` 或全 `0` 矩阵常量。回顾梯度下降我们知道随机性可能是一个比较好的选择，这里我们利用 `tf.random_uniform` 来创建一个均匀分布常量。
# 
# 当然，这里需要将常量转换为变量使用。我们接下来实现梯度下降的过程：

# In[ ]:


n_epochs = 1000 # 指定迭代次数
learning_rate = 0.01 # 指定学习率

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y")

# theta 初始化为一个（n+1）*1 大小，元素均匀分布于 -1 到 1 的矩阵
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
Y_pred = tf.matmul(X, theta, name="predictions")  # 对于每一次的 theta，y 的预测值
error = Y_pred - Y # 计算残差

# 平方误差，注意 TensorFlow 中计算矩阵元素的期望是 tf.reduce_mean
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)  # 手动算出梯度
training_op = tf.assign(theta, theta - learning_rate * gradients)  # 每一次迭代的操作， 将 theta 的值重新赋值


# In[32]:


n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
Y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="Y")

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name='theta')
Y_pred = tf.matmul(X, theta, name='predictions')
error = Y_pred - Y

# mse均方差
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)


# 接下来，我们使用 `tf.global_variables_initializer()` 初始化全局变量（不再需要一个一个初始化），并启动 Session 构建计算图：

# In[ ]:


init = tf.global_variables_initializer() # 初始化全局变量

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch ", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    
best_theta


# In[33]:


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()

best_theta


# 注意，你可能会发现这里结果跟最小二乘法结果相差很大。这是因为我们对原数据集做了归一化的操作，这里的 `theta` 将是对归一化后数据的较优权值。

# ### 利用 TensorFlow 优化器实现梯度下降

# 上文在介绍 TensorFlow 时，特意提到了 TensorFlow 的一大优势，是它能基于所搭建的模型和目标函数自动地计算出梯度。这其实是通过 TensorFlow 里内置的各种优化器（Optimizer）来实现的，比如：
# 
# - `tf.train.GradientDescentOptimizer`：梯度下降 Optimizer
# - `tf.train.MomentumOptimizer`：使用 Momentum 算法的 Optimizer
# - `tf.train.RMSPropOptimizer`：使用 RMSProp 算法的 Optimizer
# - `tf.train.AdamOptimizer`：使用 Adam 算法的 Optimizer
# 
# 当然，还有更多的优化器，可以通过 [此页面](https://www.tensorflow.org/api_guides/python/train) 查看。简单来讲，这些优化器都是以梯度优化为思想，通过学习率作为步长来逐步更新变量的值。

# 下面，我们使用 TensorFlow 提供的梯度下降优化器来完成实验，这个过程会异常简单： 

# In[34]:


learning_rate = 0.01 # 指定学习率

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)  # 选择梯度优化器，设定上文同样的学习率
training_op = optimizer.minimize(mse)  # 设定优化器优化的目标函数

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)  # 给定模型和目标函数，TensorFlow 自动计算出梯度更新变量值
    best_theta = theta.eval()

best_theta


# 不出意外，这和我们手动实现的梯度下降计算结果一模一样。

# 实际上，大量实践表明，Adam 优化器在梯度运算时，尤其是在深度神经网络时效果较佳，所以建议大家以后在做梯度下降时，默认先尝试 Adam 优化器。除非收敛效果不佳时，才考虑更换其他优化器。我们再试一次：

# In[ ]:


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # 选择 Adam 优化器，设定优化器的学习率
training_op = optimizer.minimize(mse)  # 优化器优化的目标函数

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)  # 给定模型和目标函数，TensorFlow 自动计算出梯度更新变量值
    best_theta = theta.eval()

best_theta


# In[14]:


learning_rate = 0.01

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

best_theta


# 可以看到，Adam 最终迭代的 MSE 值会更小一些，也代表效果稍好一些。

# 关于更多优化器的对比，大家可以阅读 [ [An overview of gradient descent optimization algorithms]](http://ruder.io/optimizing-gradient-descent/)

# ### 利用 TensorFlow 实现 Mini Batch 训练

# 深度学习是依赖于非常大的数据量，所运用的强力机器学习方法。如果数据量非常大，再加上神经网络很深的话，通常会花费的训练时间从数小时到数个月不等，持续时间非常长。
# 
# 于是，在实践中往往会使用一种叫 Mini Batch 的方法，也就是将整个数据分成一些小批次放进模型里进行训练。其实，在聚类算法章节，我们就已经用过 Mini Batch K-Means 方法解决 K-Means 计算时间长的问题。

# 上面的梯度下降优化过程中，我们每一次更新 `theta` 的值，则需要遍历所有的样本数据集。如果使用 Mini Batch 的方法，就只需要将这部分小批量数据作为用于本次更新的数据集，遍历这个小批量就可以使 `theta` 有一次更新。

# 实践表明，小批量梯度下降训练速度更快，最终结果也不逊于以往的传统梯度下降。所以，接下来我们就用 Mini Batch 的方法重新训练。

# 训练之前，我们需要特别注意一点。前面的实验中，我们直接将数据定为变量传入模型，而由于 Mini Batch 会不断地传入每一个 Batch，所以此时就需要用到 Placeholder 占位符张量了。使用 Placeholder 时，可以通过 `feed_dict` 将占位符予以赋值参与运算, 在调用结束后，填充数据就会消失。

# In[ ]:


n_epochs = 10
learning_rate = 0.01

# 将 X 创建为占位符，类型为 32 位浮点型，大小由于维度未知可以先记为 None，以后由 feed_dict 的数据决定
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")

# 初始化权重以及定义 MSE
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
Y_pred = tf.matmul(X, theta, name="predictions")
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error), name="mse")

# 使用 Adam 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


# In[16]:


n_epochs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
Y_pred = tf.matmul(X, theta, name='predictions')
error = Y_pred - Y
mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


# 接下来，我们需要定义一个取小批量的函数，以基于某个随机种子构建 `feed_dict` 的小批量数据。

# In[17]:


# 分为多少个mini batch
batch_size = 100
# ceil返回输入值的上限（目的是取整）
# m是样本数量
n_batches = int(np.ceil(m / batch_size))

# 定义一个取小批量的函数
def fetch_batch(epoch, batch_index, batch_size):
    # 每一次基于某个随机种子构建 feed_dict 的小批量数据
    np.random.seed(epoch * n_batches + batch_index)
    # 随机产生批量大小的索引值向量来取出批量数据
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    Y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, Y_batch

# 初始化变量并构建计算图
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # 每一次 X_batch，Y_batch 通过 fetch_batch 函数从原数据集中取出
            X_batch, Y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
    best_theta = theta.eval()

best_theta


# ## 存储或重启模型

# 上面我们提到了一个复杂的深度神经网络往往需要训练数天，甚至数个月的时间。那么，你是否想过如果有一天你一不小心踢掉了计算机的电源，是不是前功尽弃啊！TensorFlow 的开发人员当然想到了这些意外情况，于是就提供相应的模块来让我们保存训练模型的参数以便之后使用。

# 当然保持参数并不仅仅是以防万一，这在深度学习领域有重大意义。以后我们会了解到，由于深度神经网络训练艰难，对已经训练完成的网络进行进一步的设计或者迁移学习能极大提高训练效率。同时，我们经常称这些保持下来的模型为预训练模型。

# 接下来，我们就学习如何保存模型，其实非常简单，只需要在适当的位置添加几行代码即可：

# In[ ]:


import os

# 获取当前 notebook 所在目录，将预训练模型保持在该目录下方，也可以使用绝对路径
path = os.path.abspath(os.curdir)
saver = tf.train.Saver() # 存储预训练模型

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        save_path = saver.save(sess, path + '/model.ckpt')  # 在训练过程中存储模型参数
        for batch_index in range(n_batches):
            X_batch, Y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
    best_theta = theta.eval()
    save_path = saver.save(sess, path+"/model.ckpt")  # 存储最终模型参数
    print("done.")

best_theta


# In[20]:


import os

path = os.path.abspath(os.curdir)
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        save_path = saver.save(sess, path + '/model.ckpt')
        for batch_index in range(n_batches):
            X_batch, Y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X:X_batch, Y:Y_batch})
    best_theta = theta.eval()
    save_path = saver.save(sess, path+'/model.ckpt')
    print("done.")
best_theta


# 储存好预训练模型后，如果我们需要重新加载怎么操作呢？重启这个模型非常简单，我们只需要使用 `saver.restore()` 即可：

# In[21]:


with tf.Session() as sess:
    saver.restore(sess, path+"/model.ckpt")
    best_theta_restored = theta.eval()
    
best_theta_restored


# ## 使用 GPU 训练模型

# 深度学习的发展得益于两个条件：深度学习理论的进步 + 硬件算力的提升，这两个条件缺一不可。你可能知道，训练复杂的深度神经网络往往必不可少的就是 GPU，相比于适合逻辑运算的 CPU，GPU 往往拥有数千个核心，特别适合于并行数值计算。

# 我们通过 TensorFlow 搭建的深度神经网络应该如何在 GPU 上运算呢？在一套标准系统中通常有多台计算设备。TensorFlow 支持 CPU 和 GPU 这两种设备。它们均用 strings 表示。例如：

# - `"/cpu:0"`：机器的 CPU。
# - `"/device:GPU:0"`：机器的 GPU（如果有一个）。
# - `"/device:GPU:1"`：机器的第二个 GPU（以此类推）。

# 如果 TensorFlow 指令中兼有 CPU 和 GPU 实现，当该指令分配到设备时，GPU 设备有优先权。例如，如果 `matmul` 同时存在 CPU 和 GPU 核函数，在同时有 `cpu:0` 和 `gpu:0` 设备的系统中，`gpu:0` 会被选来运行 `matmul`。

# In[ ]:


with tf.device('/cpu:0'): # GPU 将在第六周提供
    a = tf.Variable([[1., 2.], [3., 4.]])
    b = tf.Variable([[5., 6.], [7., 8.]])

c = tf.matmul(a, b)
c


# In[26]:


with tf.device('/cpu:0'):
    a = tf.Variable([[1., 2.],
                    [3., 4.]])
    b = tf.Variable([[5., 6.],
                     [7., 8.]])
c = tf.matmul(a, b)
c


#  上述代码中，我们把 `a，b` 节点的运算放在了 `cpu` 上，但是 `c` 的运算没有置于这个 `tf.device` 代码块中，则它的运算在默认设备上。

# 总结一下 Tensorflow 的运算设备配置原则：
# 
# 1. 如果该节点在之前数据流图运算中已经置于了一个设备，那么它留在这个设备进行运算；
# 2. 如果用户将这个节点置换到另一个设备，那么该节点在置换后的设备进行运算；
# 3. 都不是的话，它默认在第一个 `GPU` 上运算，如果没有 `GPU` 的话则在 `CPU` 上运算。

# 虽然介绍了这么多，但实际使用 TensorFlow 在 GPU 上训练非常简单。**你什么都不需要做**，也无需指定 `tf.device('/cpu:0')` 这样的代码。如果你正确配置了 GPU 环境，TensorFlow 会自动检测到并加速计算。

# ## 实验总结

# TensorFlow 是一个非常强大的框架，许多人用了很长时间都不敢说能够学的很好，很熟悉。学习这种庞大的框架，不要想一口吃个大胖子，而是一口一口慢慢吃才是正确思路。本节实验中了解了 TensorFlow 的工作方法，并运用 TensorFlow 实现了线性回归的几个算法，学习了 TensorFlow 在搭建学习模型的大致流程和一些基本概念。回顾本次实验的知识点有：
# 
# - TensorFlow 介绍
# - TensorFlow 工作原理
# - 计算流图
# - 张量的类型
# - 启动会话
# - 常量生成
# - 梯度优化器
# - 小批量梯度下降
# - GPU 使用
# - 预训练模型

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
