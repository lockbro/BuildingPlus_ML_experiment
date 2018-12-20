
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # PyTorch 介绍及使用

# ---

# ### 实验介绍

# PyTorch 是由 Facebook 主导开发的深度学习框架，因其高效的计算过程以及良好的易用性被诸多大公司和科研人员所喜爱。18 年 5 月，PyTorch 正式宣布集成 Caffe2 和 ONNX 的功能，这是一次让业界期待的更新。本次实验中，我们将从整体上熟悉 PyTorch，并深入学习常用组件的使用。

# ### 实验知识点
# 
# - PyTorch 使用
# - Tensor 张量
# - Autograd 自动微分
# - nn 神经网络组件
# - Optimizer 优化器

# ### 实验目录
# 
# - <a href="#PyTorch-安装及介绍">PyTorch 安装及介绍</a>
# - <a href="#Tensor-张量">Tensor 张量</a>
# - <a href="#Autograd-自动微分">Autograd 自动微分</a>
# - <a href="#nn-神经网络组件">nn 神经网络组件</a>
# - <a href="#Optimizer-优化器">Optimizer 优化器</a>
# - <a href="#使用-GPU-加速计算">使用 GPU 加速计算</a>
# - <a href="#框架对比与选择">框架对比与选择</a>
# - <a href="#实验总结">实验总结</a>
# - <a href="#本周思维导图">本周思维导图</a>

# ---

# ## PyTorch 安装及介绍

# ### PyTorch 安装方法

# <img width='300px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532064070576.png"></img>

# <div style="color: #999;font-size: 12px;font-style: italic;">*实验楼在线环境已经默认安装 PyTorch，以下内容仅适合于你在本地安装体验。</div>

# 目前，PyTorch 官方支持 Linux，macOS 以及 Windows 系统，不同的系统安装命令稍有出入。虽然 PyTorch 仍然支持 Python 2，但我们强烈推荐你在 Python 3 下使用 PyTorch。同时，PyTorch 推荐使用 Anaconda 作为包管理工具（[下载并安装 Anaconda](https://www.anaconda.com/download/)）。

# 在 Python 3.6 环境下，各系统安装 CPU 版本的 PyTorch 命令如下：

# ```bash
# # Linux
# conda install pytorch-cpu torchvision-cpu -c pytorch
# # macOS
# conda install pytorch torchvision -c pytorch
# # Windows
# conda install pytorch-cpu -c pytorch 
# pip3 install torchvision
# ```

# 如果想在本地 GPU 环境中使用 PyTorch，依据 [CUDA](https://zh.wikipedia.org/wiki/CUDA) 版本的不同，安装命令如下：

# ```bash
# ### CUDA 8 ###
# # Linux
# conda install pytorch torchvision -c pytorch
# # Windows
# conda install pytorch -c pytorch 
# pip3 install torchvision
# 
# ### CUDA 9 ###
# # Linux
# conda install pytorch torchvision cuda90 -c pytorch
# # Windows
# conda install pytorch cuda90 -c pytorch 
# pip3 install torchvision
# ```

# 安装完 PyTorch 就可以通过 `import torch` 导入模块了，注意这里不是 `import pytorch`（错误）。截止本实验制作完成时，PyTorch 为 `0.4.0` 版本，我们将尽可能适配实验内容为最新版本。

# ### PyTorch 框架介绍

# PyTorch 作为深度学习的重要框架，其实就提供了 2 个核心功能：
# 
# - 高效率的张量解算, 同时支持强大的 GPU 加速计算功能。
# - 搭建深度神经网络, 构建在自动求导系统之上的网络结构。

# 接下来，介绍一下 PyTorch 模块的组件构成：

# |       Package（包）      |                           Description（描述）                          |
# |:------------------------:|:----------------------------------------------------------------------:|
# |           `torch`          |         张量计算组件, 兼容 NumPy 数组，且具备强大的 GPU 加速支持       |
# |      `torch.autograd`      | 自动微分组件, 是 PyTorch 的核心特点，支持 torch 中所有可微分的张量操作 |
# |         `torch.nn`         |          深度神经网络组件, 用于灵活构建不同架构的深度神经网络          |
# |        `torch.optim`       |   优化计算组件, 囊括了 SGD, RMSProp, LBFGS, Adam 等常用的参数优化方法  |
# |   `torch.multiprocessing`  |             多进程管理组件，方便实现相同数据的不同进程中共享视图             |
# |        `torch.utils`       |               工具函数组件，包含数据加载、训练等常用函数               |
# | `torch.legacy(.nn/.optim)` |                     向后兼容组件, 包含移植的旧代码                     |

# 上面对 PyTorch 模块的核心组件进行了罗列，下面将会对其进行更加详细的介绍，并希望大家能够反复动手练习，这是掌握 PyTorch 使用的重要过程。

# ## Tensor 张量

# 我们都知道，Tensor 张量在深度学习中充当着空气的角色，神经网络结构中的每一次输入和输出其实就是针对张量的计算过程。所以，PyTorch 最核心的组件也就是张量计算组件, 它兼容 NumPy 数组，且具备强大的 GPU 加速支持。下面，我们就来学习如何在 PyTorch 中去定义张量，并对其进行运算。

# ### Tensor 类型

# 首先，我们介绍 PyTorch 中支持的 Tensor 类型。目前，PyTorch 提供了 7 种 CPU 支持的 Tensor 类型和 8 种 GPU 支持的 Tensor 类型，它们分别是：

# |       数据类型 dtype       |     CPU Tensor     |        GPU Tensor       |
# |:--------------------------:|:------------------:|:-----------------------:|
# |         32-bit 浮点        |  torch.FloatTensor |  torch.cuda.FloatTensor |
# |         64-bit 浮点        | torch.DoubleTensor | torch.cuda.DoubleTensor |
# |      16-bit 半精度浮点     |         N/A        |  torch.cuda.HalfTensor  |
# |   8-bit 无符号整形(0~255)  |  torch.ByteTensor  |  torch.cuda.ByteTensor  |
# | 8-bit 有符号整形(-128~127) |  torch.CharTensor  |  torch.cuda.CharTensor  |
# |      16-bit 有符号整形     |  torch.ShortTensor |  torch.cuda.ShortTensor |
# |      32-bit 有符号整形     |   torch.IntTensor  |   torch.cuda.IntTensor  |
# |      64-bit 有符号整形     |  torch.LongTensor  |  torch.cuda.LongTensor  |

# 其中，默认的 `torch.Tensor` 类型为 `32-bit 浮点`，也就是 `torch.FloatTensor`。

# In[1]:


import torch as t

t.Tensor().dtype


# **☞ 动手练习：**

# 如果需要指定类型的 Tensor，使用表格中对应的方法即可：

# In[ ]:


t.DoubleTensor()


# In[2]:


t.DoubleTensor()


# In[ ]:


t.ByteTensor()


# In[3]:


t.ByteTensor()


# 如果你想指定全局为某种类型的 Tensor，使得代码书写起来更加方便，就可以通过配置完成，例如：

# In[ ]:


t.set_default_tensor_type('torch.DoubleTensor')


# In[4]:


t.set_default_tensor_type("torch.DoubleTensor")


# 此时，`torch.Tensor` 类型就被更改为 64-bit 浮点。

# In[ ]:


t.Tensor().dtype


# In[5]:


t.Tensor().dtype


# 当然，最常用的就是默认的 `torch.Tensor()` ，也就是 `torch.FloatTensor()`。我们还原配置：

# In[ ]:


t.set_default_tensor_type('torch.FloatTensor')
t.Tensor().dtype


# In[6]:


t.set_default_tensor_type('torch.FloatTensor')
t.Tensor().dtype


# ### 创建 Tensor

# 说完 PyTorch 中支持 Tensor 的类型，接下来我们看一看如何创建 Tensor。当然，最基础的方式是传入一个列表，即可创建：

# In[ ]:


t.Tensor([1, 2, 3])


# In[7]:


t.Tensor([1, 2, 3])


# In[ ]:


t.Tensor([[1, 2], [3, 4], [5, 6]])


# In[8]:


t.Tensor([[1, 2], [3, 4], [5, 6]])


# 与 Numpy 一致，可以通过 `shape` 查看 Tensor 的形状：

# In[ ]:


t.Tensor([[1, 2], [3, 4], [5, 6]]).shape


# In[9]:


t.Tensor([[1, 2], [3, 4], [5, 6]]).shape


# 除此之外，与 NumPy 类似，PyTorch 也有许多其他的方法，可以快速创建特定类型的 Tensor。

# |                方法               |                     描述                    |
# |:---------------------------------:|:-------------------------------------------:|
# |            ones(*sizes)           |             创建全为 1 的 Tensor            |
# |           zeros(*sizes)           |             创建全为 0 的 Tensor            |
# |            eye(*sizes)            |      创建对角线为 1，其他为 0 的 Tensor     |
# |         arange(s, e, step)        |     创建从 s 到 e，步长为 step 的 Tensor    |
# |       linspace(s, e, steps)       | 创建从 s 到 e，均匀切分成 steps 份的 Tensor |
# |         rand/randn(*sizes)        |          创建均匀/标准分布的 Tensor         |
# | normal(mean, std) |        创建正态分布分布的 Tensor       |
# |            randperm(m)            |            创建随机排列的 Tensor            |

# 下面，我们依次练习表格中的方法：

# In[10]:


t.ones(2, 3)


# In[11]:


t.zeros(2, 3)


# In[12]:


t.eye(3, 3)


# In[13]:


t.arange(1, 10, 1)


# In[14]:


t.linspace(1, 10, 4)


# In[15]:


t.rand(2, 3)


# In[ ]:


# 必须传入 tensor
t.normal(t.Tensor([.5]), t.Tensor([.3]))


# In[16]:


t.normal(t.Tensor([0.5]), t.Tensor([0.3]))


# In[ ]:


t.randperm(5)


# In[20]:


t.randperm(3)


# 正如上面所示，你会发现 PyTorch 中创建 Tensor 的方法和 NumPy 创建数组的方法大部分都很相似。其实，这是故意设计的，方便熟悉 NumPy 的人士能迅速上手。

# ### Tensor 基本运算

# 学会 Tensor 的花式创建方法之后，我们再了解一下 Tensor 的基本运算。

# 两个 Tensor 是可以直接相互加减的，但要保证形状一致：

# In[ ]:


a = t.Tensor([[1, 2],[3, 4]])
b = t.Tensor([[5, 6],[7, 8]])

print(a)
print(b)


# In[22]:


a = t.Tensor([[1, 2], [3, 4]])
b = t.Tensor([[5, 6], [7, 8]])

print(a)
print(b)


# In[23]:


a + b


# In[24]:


a - b


# 注意，Tensor 使用 `*` 运算时，为元素相乘。

# In[25]:


a * b


# In[26]:


a / b


# 除此之外，我们可以针对 Tensor 执行求和、求均值、求标准差等操作。详细包括：

# |方法|描述|
# |:---:|:---:|
# |mean / sum / median / mode|均值 / 和 / 中位数 / 众数|
# |norm / dist|范数 / 距离|
# |std / var|标准差 / 方差|
# |cumsum / cumprod|累加 / 累乘|

# 接下来，动手练习一下：

# In[31]:


a.norm()


# 注意，很多时候我们都需要通过 `dim=` 参数去指定操作的维度，正如 NumPy 中的 `axis=` 参数。

# In[ ]:


# 对 a 求列平均
a.mean(dim=0)


# In[32]:


a.mean(dim=0)


# In[ ]:


# 对 a 求行平均
a.mean(dim=1)


# In[33]:


a.mean(dim=1)


# 你可以自行新建单元格练习剩余的方法。另外，还有一些针对元素的常用函数：

# | 方法 | 描述 |
# |:-----------------------------:|:-------------------------------------:|
# | abs / sqrt / div / exp / fmod / log / pow | 绝对值 / 平方根 / 除法 / 指数 / 求余 / 求幂.. |
# | cos / sin / asin / atan2 / cosh | 三角函数 |
# | ceil / round / floor / trunc | 上取整 / 四舍五入 / 下取整 / 只保留整数部分 |
# | clamp(input, min, max) | 超过 min 和 max 部分截断 |
# | sigmod / tanh | 常用激活函数 |

# In[ ]:


# a 中的每个元素求平方
a.pow(2)


# In[34]:


a.pow(2)


# In[ ]:


# a 中的每个元素传入 sigmoid 函数
a.sigmoid()


# In[35]:


a.sigmoid()


# ### Tensor 线性代数

# 同样，PyTorch 提供了常见线性代数运算方法。详细包括如下：

# | 方法 | 描述 |
# |:---------:|:-----------------:|
# | trace | 对角线元素之和 |
# | diag | 对角线元素 |
# | triu / tril | 上三角 / 下三角矩阵 |
# | mm | 矩阵乘法 |
# | t | 转置 |
# | inverse | 求逆矩阵 |
# | svd | 奇异值分解 |

# 接下来，动手练习一下：

# In[46]:


a.inverse()


# In[ ]:


b.diag()


# `mm` 是 `matmul` 的缩写，也就是矩阵的点积。

# In[ ]:


b.mm(a)


# In[47]:


b.matmul(a)


# 当然，这里还有一些方法没有介绍完，如果需要的时候就去查阅官方文档即可。

# ### Tensor 索引、切片、变换

# 很多时候，我们可能只需要使用 Tensor 的一部分，就需要用到索引和切片操作了。PyTorch 中的这两个操作和 NumPy 同样非常相似，所以大家在使用时是非常方便的。

# In[ ]:


c = t.rand(5, 4)
c


# In[48]:


c = t.rand(5, 4)
c


# In[ ]:


# 取第 1 行
c[0]


# In[49]:


c[0]


# In[ ]:


# 取第 1 列
c[:, 0]


# In[50]:


c[:,0]


# In[ ]:


# 取第 2, 3 行与 3, 4 列
c[1:3, 2:4]


# In[51]:


c[1:3, 2:4]


# 有时候，我们可能需要对 Tensor 的形状做出改变。在 NumPy 中，我们会使用到 Reshape 或者 Resize，而在 PyTorch 中对应到这两个操作：

# In[ ]:


c.reshape(4, 5)


# In[54]:


c.reshape(4, 5)


# 除此之外，PyTorch 中的 `view()` 也能达到相似的效果。

# In[ ]:


c.view(4, 5)


# In[55]:


c.view(4, 5)


# 对应，`reshape()`, `resize()` 和 `view()`，三者直接的区别在于：`resize()` 和 `view()` 执行变换时和原 Tensor 共享内存，即修改一个，另外一个也会跟着改变。而 `reshape()` 则会复制到新的内存区块上。

# ### Tensor 的内部结构

# 上面介绍了 Tensor 的创建、计算、变换方法。最后，我们来看一下 Tensor 的内部结构。Tensor 大致由两部分组成，分别是信息区（Tensor）和存储区（Storage）。其中，信息区保存 Tensor 的形状、数据类型等属性，而储存区则保留着真正的数据。

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532064070825.png)

# In[56]:


d = t.rand(3, 2)
d


# In[ ]:





# In[63]:


d.storage()


# In[58]:


d = d.reshape(2, 3)
d


# In[59]:


d.storage()


# 可以看到，虽然 Tensor 的 size 属性改变，但是 Storge 则不会改变。当然，我们这里只是一个简单的概括，并不是特别严谨。如果你想深入了解 PyTorch 的内部结构，可以阅读这篇文章：[PyTorch – Internal Architecture Tour](http://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/)。

# 关于 PyTorch 中张量 Tensor 相关的内容就暂时介绍到这里。其实你会发现 PyTorch 中的 Tensor 相关函数和方法和 NumPy 中的 Ndarray 是非常相似的。所以，如果你掌握了 Ndarray 的使用，那么 Tensor 的使用就没有问题了。

# ## Autograd 自动微分

# Autograd 自动微分是 PyTorch 的核心机制，其能够根据前向传播过程自动构建计算图，并自动完成反向传播而不需要手动去实现反向传播的过程，便利性可想而知。

# Autograd 的核心数据结构叫 Variable。Variable 中封装了 Tensor，并通过追踪操作记录来构建计算图。特别地，Variable 的数据结构包含三部分：
# 
# - `data`：数据，也就是对应的 Tensor。
# - `grad`：梯度，也就是 Tensor 对应的梯度，注意 `grad` 同样是 Variable。 
# - `creator`：父节点，它指向把它作为输出的函数，追踪 Tensor 的操作历史，用来构建计算图。

# ![image](https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532064071049.png)

# 我们通过传入一个 Tensor 去构造 Variable，同时可以指定 2 个参数：
# 
# - `requires_grad (bool)`：决定是否对 Variable 微分，如果 `requires_grad=True`，计算过程将被追踪，此变量的梯度将累积到 `.grad` 属性中。
# - `volatile (bool)`：直译为「挥发」，如果 `volatile=True`，则构建在该 Variable 之上的图都不会微分，此参数为推理阶段设计。

# 与 Tensorflow 的自动微分过程相似，PyTorch 同样用到了计算图。特别地，PyTorch 还是动态运算图(Dynamic Computation Graph)，它可以让我们的计算模型更灵活、复杂，并可以让反向传播算法随时进行。

# 下面，我们就通过一个例子计算一下。

# In[ ]:


x = t.ones(3, 4, requires_grad=True)
x


# In[64]:


x = t.ones(3, 4, requires_grad=True)
x


# 打印 Variable 的数据结构：

# In[65]:


print(x.data)
print(x.grad)
print(x.grad_fn)


# 因为这里没有计算过程，所以 `grad` 和 `grad_fn` 都是 `None`。我们可以对 `x` 执行一次运算：

# In[ ]:


y = x + 2
print(y)
y.grad_fn


# In[70]:


y = x + 2
print(y)
y.grad_fn


# 此时，`grad_fn` 就追踪到了计算过程。而此时的计算图就是这样：

# <img width='100px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532064071236.png"></img>

# 接下来，我们让计算图更加复杂一点，这里添加求均值的复合运算过程：

# In[ ]:


z = t.mean(y.pow(3))
print(z)
z.grad_fn


# In[72]:


z = t.mean(y.pow(3))
print(z)
z.grad_fn


# 然后，计算图变成了下面这样：

# <img width='200px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532064071420.png"></img>

# 此时，你可以使用 `backward` 进行反向传播，并计算所有叶子节点的导数（梯度）。注意，Z 和 Y 都不是叶子节点，所以都没有梯度信息。

# In[ ]:


z.backward()
print(z.is_leaf, y.is_leaf, x.is_leaf)
print(z.grad)
print(y.grad)
print(x.grad)


# In[73]:


z.backward()
print(z.is_leaf, y.is_leaf, x.is_leaf)
print(z.grad)
print(y.grad)
print(x.grad)


# 注意，由于梯度是会累计的，所以重复运行计算 Z 的代码就相当于修改了计算图，相应的梯度值也会发生变化。除此之外，如果你重复运行 `backward()` 会报错，原因是前向传递过程之后将所需数值保存为 buffer，当计算完梯度之后会自动清空。如果想要多次反向传播，那就需要通过 `backward(retain_graph=True)` 参数来保证 buffer 的持久性。

# 下面，我们再通过一个例子对比 Autograd 的自动求导和手动求导数的过程。首先，创建一个随机 Tensor：

# In[118]:


from torch.autograd import Variable
x = Variable(t.randn(3,4), requires_grad = True)
x


# 你可能还记得 Sigmoid 函数及它对应的导数公式：

# $$
# g(z)=\frac{1}{1+e^{-z}}
# $$

# $$
# g(z)'=g(z)(1-g(z))
# $$

# 接下来，手动实现 Sigmoid 函数，及对应的导数求解函数：

# In[109]:


# sigmoid 函数
def sigmoid(x):
    return 1.0/(1.0 + t.exp(-x))

# sigmoid 函数求导
def sigmoid_derivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))


# In[119]:


y = sigmoid(x)
y


# 然后，我们可以通过 Autograd 机制得到导数计算结果。这里需要使用到 `backward`，并传入一个形状与 `y` 一致的 `grad_variables`：

# In[120]:


y.backward(t.ones(y.size()))
x.grad


# 当然，可以通过 `sigmoid_derivative()` 函数手动求导，查看计算结果：

# In[121]:


sigmoid_derivative(x)


# 你会看到，使用 Autograd 机制得到的计算结果和手动计算结果是一致的。

# ## nn 神经网络组件

# 介绍完基本元素 Tensor 以及 Autograd 机制，如果要实现深度神经网络，必不可少的就是网络结构。在 PyTorch 中，搭建神经网络结构的组件在 `torch.nn` 中。方便定义不同类型的 Tensor 及利于反向传播的 Autograd 机制是深度学习框架的重要特点，但真正带来极大便利的，莫过于已经封装好的不同神经网络结构组件，包括不同类型的层以及各式各样的损失函数、激活函数、优化器等。正因为有了像 Tensorflow，PyTorch 等开源框架，我们才有可能快速实现负责的深度神经网络。

# 下面，我们梳理一下 `torch.nn` 下用于构建不同类型神经网络的不同网络层，它们大致有：

# <ul>
# <li> Convolution Layers (卷积层) <ul>
# <li> `Conv1d` : 一维卷积层</li>
# <li> `Conv2d` : 二维卷积层</li>
# <li> `Conv3d` : 三维卷积层</li>
# <li> `ConvTranspose1d` : 一维反卷积层</li>
# <li> `ConvTranspose2d` : 二维反卷积层</li>
# <li> `ConvTranspose3d` : 三维反卷积层</li>
# </ul>
# </li>
# <li> Pooling Layers (池化层) <ul>
# <li> `MaxPool1d` : 一维最大池化</li>
# <li> `MaxPool2d` : 二维最大池化</li>
# <li> `MaxPool3d` : 三维最大池化</li>
# <li> `MaxUnpool1d` : 一维最大反池化</li>
# <li> `MaxUnpool2d` : 二维最大反池化</li>
# <li> `MaxUnpool3d` : 三维最大反池化</li>
# <li> `AvgPool1d` : 一维平均池化</li>
# <li> `AvgPool2d` : 二维平均池化</li>
# <li> `AvgPool3d` : 三维平均池化</li>
# <li> `FractionalMaxPool2d` : 二维的分数最大池化 </li>
# <li> `LPPool2d` : 二维的幂平均池化</li>
# <li> `AdaptiveMaxPool1d` : 一维自适应最大池化</li>
# <li> `AdaptiveMaxPool2d` : 二维自适应最大池化</li>
# <li> `AdaptiveMaxPool3d` : 三维自适应最大池化</li>
# <li> `AdaptiveAvgPool1d` : 一维自适应平均池化</li>
# <li> `AdaptiveAvgPool2d` : 二维自适应平均池化</li>
# <li> `AdaptiveAvgPool3d` : 三维自适应平均池化</li>
# </ul>
# </li>
# <li> Padding Layers (填充层) <ul>
# <li> `ReflectionPad2d` : 反射填充</li>
# <li> `ReplicationPad2d` : 复制填充</li>
# <li> `ReplicationPad3d` : 复制填充</li>
# <li> `ZeroPad2d` : 零填充</li>
# <li> `ConstantPad2d` : 常数填充</li>
# </ul>
# </li>
# <li> Non-linear Activations (非线性层) <ul>
# <li> `ReLU` : 修正线性单元函数</li>
# <li> `ReLU6` : ReLU6 函数</li>
# <li> `ELU` : ELU 函数</li>
# <li> `SELU` : SELU 函数</li>
# <li> `PReLU` : PReLU 函数</li>
# <li> `LeakyReLU` : LeakyReLU 函数</li>
# <li> `Threshold` : Threshold 函数</li>
# <li> `Hardtanh` : Hardtanh 函数</li>
# <li> `Sigmoid` : Sigmoid 函数</li>
# <li> `Tanh` : Tanh 函数</li>
# <li> `LogSigmoid` : LogSigmoid 函数</li>
# <li> `Softplus` : Softplus 函数</li>
# <li> `Softshrink` : Softshrink 函数</li>
# <li> `Softsign` : Softsign 函数</li>
# <li> `Tanhshrink` : Tanhshrink 函数</li>
# <li> `Softmin` : Softmin 函数</li>
# <li> `Softmax` : Softmax 函数</li>
# <li> `Softmax2d` : Softmax2d 函数</li>
# <li> `LogSoftmax` : LogSoftmax 函数</li>
# </ul>
# </li>
# <li> Normalization layers (归一化层) <ul>
# <li> `BatchNorm1d` : 批标准化</li>
# <li> `BatchNorm2d` : 批标准化</li>
# <li> `BatchNorm3d` : 批标准化</li>
# <li> `InstanceNorm1d` : 实例标准化</li>
# <li> `InstanceNorm2d` : 实例标准化</li>
# <li> `InstanceNorm3d` : 实例标准化</li>
# </ul>
# </li>
# <li> Recurrent layers (循环层) <ul>
# <li> `RNN` : 对于输入序列使用一个多层 RNN</li>
# <li> `LSTM` : 对于输入序列使用一个多层 LSTM</li>
# <li> `GRU` : 对于输入序列使用一个多层 GRU</li>
# <li> `RNNCell` : 一个 RNN 细胞 </li>
# <li> `LSTMCell` : 一个 LSTM 细胞 </li>
# <li> `GRUCell` : 一个 GRU 细胞</li>
# </ul>
# </li>
# <li> Linear layers (线性层) <ul>
# <li> `Linear` : 对输入数据进行线性变换 :</li>
# <li> `Bilinear` : 对输入数据进行双线性变换 :</li>
# </ul>
# </li>
# <li> `Dropout layers (丢弃层)` <ul>
# <li> `Dropout` : 随机断开神经元连接</li>
# <li> `Dropout2d` : 随机断开神经元连接</li>
# <li> `Dropout3d` : 随机断开神经元连接</li>
# <li> `AlphaDropout` : 随机断开神经元连接</li>
# </ul>
# </li>
# <li> Sparse layers (稀疏层) <ul>
# <li> `Embedding` : 存储 word embeddings</li>
# <li> `EmbeddingBag` : 计算 bags 中 embeddings 的均值或和</li>
# </ul>
# </li>
# <li> `Distance functions (距离函数)` <ul>
# <li> `CosineSimilarity` : 余弦相似度</li>
# <li> `PairwiseDistance` : 分批成对距离</li>
# </ul>
# </li>
# <li> Loss functions (损失函数) <ul>
# <li> `L1Loss` : L1 损失函数</li>
# <li> `MSELoss` : MSE 损失函数</li>
# <li> `CrossEntropyLoss` : CrossEntropy 损失函数</li>
# <li> `NLLLoss` : NLL 损失函数</li>
# <li> `PoissonNLLLoss` : PoissonNLL 损失函数</li>
# <li> `NLLLoss2d` : NLL 损失函数</li>
# <li> `KLDivLoss` : KLDiv 损失函数</li>
# <li> `BCELoss` : BCE 损失函数</li>
# <li> `BCEWithLogitsLoss` : BCEWithLogits 损失函数</li>
# <li> `MarginRankingLoss` : MarginRanking 损失函数</li>
# <li> `HingeEmbeddingLoss` : HingeEmbedding 损失函数</li>
# <li> `MultiLabelMarginLoss` : MultiLabelMargin 损失函数</li>
# <li> `SmoothL1Loss` : SmoothL1 损失函数</li>
# <li> `SoftMarginLoss` : SoftMargin 损失函数</li>
# <li> `MultiLabelSoftMarginLoss` : MultiLabelSoftMargin 损失函数</li>
# <li> `CosineEmbeddingLoss` : CosineEmbedding 损失函数</li>
# <li> `MultiMarginLoss` : MultiMargin 损失函数</li>
# <li> `TripletMarginLoss` : TripletMargin 损失函数</li>
# </ul>
# </li>
# </ul>

# 部分层的具体含义，我们会放到下一周中详细介绍，比如卷积神经网络会用到卷积层、池化层、填充层，当然还可能用到归一化层等。

# 可以看出，`torch.nn` 下的类是非常多的，几乎你想到的都能提供。此时，就能体现出框架的优势了，我们可以通过搭积木的形式来构建神经网络结构，从而避免掉负责的实现过程。
# 
# 下面，我们选择其中的一些类进行认识性练习，比如：

# #### 一维卷积层

# In[122]:


from torch import nn

layer = nn.Conv1d(10, 5, kernel_size=2, stride=2)
i = Variable(t.randn(1, 10, 10))
print("{} → {}".format(i.shape, layer(i).shape)) # 查看输入和输出 tensor 形状变化
layer(i)


# #### 一维最大池化层

# In[125]:


layer = nn.MaxPool1d(2, stride=2)
i = Variable(t.randn(1, 1, 10))
print("{} → {}".format(i.shape, layer(i).shape)) # 查看输入和输出 tensor 形状变化
layer(i)


# 上面我们罗列了近百种神经网络不同层所属的类。那么，如果你想封装自己的层该怎么办呢？PyTorch 有什么很好的办法吗？

# 其实是有的，这里我们来看一下 `torch.nn` 中的两个基础类：
# 
# `torch.nn.Parameter` 类，它通常被用于定义模型的参数。Parameter 是 Variable 的子类，但 Variable 与 Parameter 的不同之处在于, Parameter 不能设置 `volatile=True` 而且默认 `requires_grad=True`。
# 
# 另外，`torch.nn.Module` 类则是所有神经网络的基类，它既可以表示神经网络中的某层，也可以表示若干层的神经网络。目前，你所看到的 `torch.nn` 中的各个类实际上就是由 `torch.nn.Modules` 继承而拓展。所以，在实际使用中，我们可以继承`nn.Module`，撰写自定义网络层。

# In[126]:


from torch.nn.functional import sigmoid

class DIYModel(nn.Module): # 继承 nn.Module
    def __init__(self):
        super(DIYModel, self).__init__() # 等价于 nn.Module.__init__(self)
        self.linear1 = nn.Linear(20, 10)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x): # 定义前向传递过程
        x = sigmoid(self.linear1(x)) # 使用 sigmoid 激活函数
        return sigmoid(self.linear2(x))

DIYModel()


# 上面输出了我们重新封装模型 `DIYModel()` 的内部结构。简单来讲，我们将两个线性变换层封装在一起，并使用 `Sigmoid` 作为激活函数。结构如下所示：

# <img width='700px' src="https://doc.shiyanlou.com/document-uid214893labid6102timestamp1532064071654.png"></img>

# 此时，可以传入一个 Tensor 得到输出，并可以查看 Tensor 形状的变化。

# In[127]:


x = Variable(t.randn(1, 20)) # 生成随机数据
y = DIYModel().forward(x) # 前向传递

print("input:", x) # 打印输入
print("output:", y) # 打印输出
print("{} → {}".format(x.shape, y.shape)) # 查看 Tensor 形状变化


# ## Optimizer 优化器

# 在第二周的课程中，我们已经了解并学习过感知机及人工神经网络的内容。回忆一下神经网络结构中的几个要素，差不多有：神经元（层）、激活函数、损失函数。目前，`torch.nn` 模块已经提供了这些组件。不过，要想一个神经网络收敛，还有一样必不可少的法宝，那就是优化器。也就是说，我们需要通过一种优化手段去求解损失函数的最优解。

# PyTorch 已经为我们准备好了常用的优化器，它们都封装在 `torch.optim` 下方。主要有以下几种算法：

# - `torch.optim.Adadelta()`：Adadelta 优化算法
# - `torch.optim.Adagrad()`：Adagrad 优化算法
# - `torch.optim.Adam()`：Adam 优化算法
# - `torch.optim.SparseAdam()`：适用于 Sparse Tensors 的 Adam 优化算法
# - `torch.optim.Adamax()`：Adamax 优化算法
# - `torch.optim.ASGD()`：平均随机梯度下降优化算法
# - `torch.optim.LBFGS()`：L-BFGS 优化算法.
# - `torch.optim.RMSprop()`：RMSprop 优化算法
# - `torch.optim.Rprop()`：弹性反向传播优化算法
# - `torch.optim.SGD()`：随机梯度下降优化算法

# 使用优化器的方法非常简单直观，只需要把需要优化的参数和学习率等优化方法本身的参数传入即可。这里我们拿上面的 `DIYModel()` 进行演示：

# In[128]:


from torch import optim

model = DIYModel()
sgd = optim.SGD(params=model.parameters(), lr=.5) # 传入需要优化的参数和学习率等优化方法本身的参数
sgd.zero_grad() # 梯度清零

x = Variable(t.randn(1, 20)) # 随机数据
out = model.forward(x) # 前向传播
out.backward(out) # 反向传播
sgd.step() # 随机梯度下降
out # 输出


# ---

# ## 使用 GPU 加速计算

# <div style="color: #999;font-size: 12px;font-style: italic;">*当前环境为非 GPU 版 PyTorch，你可以在本地练习本小节内容或等待第 6 周 GPU 环境开放。</div>

# 作为一款成熟的深度学习框架，支持 GPU 加速运算是必不可少的特点。PyTorch 同样也支持在 GPU 上训练模型，但与 TensorFlow 不同之处在于，PyTorch 无法自动识别 GPU 并加载模型参与运算，而是需要手动将模型迁移到 GPU 上。
# 
# 简单来讲，如果你在 CPU 环境中定义好了一个模型，那么将此模型放到 GPU 上时，需要进行额外的处理。

# ### CUDA 环境

# 我们都知道，CUDA 是 Nvidia 推出的并行计算架构。于是，当在 Nvidia GPU 环境中使用 PyTorch 时，就需要将原先仅支持 CPU 的张量和模型迁移为支持 CUDA 的类型。这个过程听起来复杂，但也比较简单。

# 首先，你需要确认环境是否支持 GPU 加速计算。

# ```
# t.cuda.is_available() # 返回布尔类型
# ```

# 当然，还可以使用下面的方法查看 GPU 配置信息。

# ```
# t.cuda.get_device_properties(device=0) # 查看默认 GPU 属性
# ```

# ### 迁移模型及数据

# 将一个 CPU 模型迁移到 GPU 环境中主要有两个步骤，分别是**模型迁移**和**数据迁移**。例如，上一小节的例子中，我们定义好 `DIYModel()` 模型之后，如果想该模型支持 CUDA 运算，只需要多添加一行代码即可。

# ```
# model_cuda = model.cuda() # 将模型迁移为 CUDA 类型
# ```

# 迁移完模型之后，需要将输入模型的数据也转换为 `cuda()` 类型，方法一致。

# ```
# x_cuda = x.cuda() # 将输入数据迁移为 CUDA 类型
# ```

# 最后，我们使用转换好的模型和数据进行前向和反向计算。

# ```
# out_cuda = model_cuda.forward(x_cuda) # 前向传播
# out_cuda.backward(out_cuda) # 反向传播
# sgd.step() # 随机梯度下降
# out_cuda # 输出
# ```

# 此时，你可以看到输出张量的最后显示支持 `device='cuda:0'`，即代表上面的过程在 GPU 上完成运算。

# 你可能会有疑问，想要在 GPU 上使用 PyTorch 为什么这么麻烦呢？其实这里的「麻烦」是相对的，比 TensorFlow 是要麻烦一些，但你要知道很多年之前想要使用 GPU 计算，还得会 CUDA 编程才行，那可不是添加 `.cuda()` 这么简单。

# `.cuda()` 操作简单的理解就是将数据转移到显存中去，而模型也需要 `.cuda()` 操作的原因其实是因为模型中存在参数，这些参数同样需要转移到显存才行。

# ## 框架对比与选择

# ### 框架对比

# <img width='700px' src="https://doc.shiyanlou.com/document-uid214893labid7506timestamp1538037958067.png"></img>
# <div style="color: #888; font-size: 10px; text-align: right;">[©️ 图片来源](https://agi.io/2018/02/09/survey-machine-learning-frameworks/)</div>

# 在 [机器之心](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650726576&idx=3&sn=4140ee7afc67928333e971062d042c59&chksm=871b24ceb06cadd8922cde50cbc5da6a04fd3f00a78964381c593b2dcf62bb78835159a00f27&scene=0#rd) 上发表的一篇文章，对各个框架介绍的非常详细，在这里就不需要过多叙述每个框架的特性。机器学习框架非常多，不可能每一个都学一遍，这是不现实的，只能挑一个更适合自身情况的、感兴趣的框架之一、二进行深度学习。
# 
# 以下是各个框架在 github 上的地址: 

# - [CNTK](https://github.com/Microsoft/CNTK)
# - [TensorFlow](https://github.com/tensorflow/tensorflow)
# - [PyTorch](https://github.com/pytorch/pytorch)
# - [Keras](https://github.com/keras-team/keras)
# - [MXNet](https://github.com/apache/incubator-mxnet)
# - [Caffe](https://github.com/BVLC/caffe)
# - [Torch7](https://github.com/torch/torch7)
# - [Theano](https://github.com/Theano/Theano)

# MXNet、CNTK 分别是由 Amazon、Mirosoft 两家世界级公司开发，两个框架都做的非常优秀和易用，但是都不温不火。比较有趣的是，在知乎：[为什么CNTK知名度和普及率不如Tensorflow、Theano、caffe、Torch？](https://www.zhihu.com/question/51920350) 以及 [为什么强大的 MXNet 一直火不起来？](https://www.zhihu.com/question/52498007/answer/131175266) 都有讨论这两个框架火不起来的原因，如果感兴趣，可以去了解这段历史。

# Torch 和 Theano 是比较早期的两个框架，对深度学习框架的发展影响深远。Torch 是基于 Lua 的，一门比较冷门的语言。Theano 是三大神之一的 Bengio 开发，但是已经停止维护了。

# Keras 是基于 TensorFlow、Theano 以及 CNTK 的更高层次封装的 API，相对于其他框架非常容易上手。但是也正是因为这样的高层次封装，导致 Keras 很难去修改训练细节，相对于原生框架要慢，无法理解底层是怎么运行的。

# Caffe 是贾扬清在 UC Berkeley 读 PhD 时开源的框架，目的很明确，主要用于图像的深度学习。Caffe 因为非常快的处理速度，广泛用于商业部署上。
# Caffe 在 Computer Vison（计算机视觉）的地位是独一无二的，虽然现在正在被 TensorFlow、PyTorch 等新兴框架所替代，但是由于积淀 Caffe 的社区支持非常好，在 CV 领域很多新的 idea，文章都是基于 Caffe 实现，例如著名的 Object Detection 算法：Fast RCNN，Faster RCNN，SSD 等。因此，如果你主要研究方向是计算机视觉，那么你也有必要去学习这个框架。

# 还有许多其他的框架，商业上一个重点是移动端，所以支持移动端部署的框架也很多。

# 支持移动端的框架:
# - [Caffe2](https://github.com/caffe2/caffe2)
# - [NCNN](https://github.com/Tencent/ncnn)
# - [TensorFlow Lite](https://www.tensorflow.org/lite/)
# - [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
# - [Paddle-Mobile](https://github.com/PaddlePaddle/paddle-mobile)

# 其他的框架，例如国产的：百度开发的 PaddlePaddle，腾讯开发的 NCNN，不多做介绍。
# 
# 综合以上总总原因，现有的选择就是两个主流的深度学习框架：TensorFlow 和 PyTorch。在这篇 [PyTorch vs TensorFlow，哪个更适合你](https://yq.aliyun.com/articles/183473) 当中，详细对比了两个框架，可能要比本次实验介绍的更加详细。两个框架分别由两个大公司开发，差别可能并不大，因此，总结一下二者各自的优势：

# ### PyTorch 的优势

# #### 对用户更友好

# 在 [如何评价 PyTorch 1.0 Roadmap？](https://www.zhihu.com/question/275682850) 几乎全都是吐槽 TensorFlow，并且赞美 PyTorch 易用性的回答。

# PyTorch 的易用具体表现在容易调试，直接使用最基本的 `print` 或者 PyCharm 的 debugger 就能一步步调试看输出，而 TensorFlow 需要使用 `tfdbg`，如果要调试 Python 原生代码还需要使用其他工具。

# PyTorch 自带了许多已经写好的模块，如数据加载、预处理等，并且用户实现自定义模块也非常方便，可能只需要继承某各类，然后写几个方法就可以。但是在 TensorFlow 中，每定义一个网络，都需要声明、初始化权重、偏置等参数，所以需要写大量的重复代码。而 PyTorch 在 `nn.Module` 模块提供了很多卷积层、池化层等深度学习的结构，相对 TensorFlow 的抽象层次要高，所以非常方便。

# #### 代码可阅读性非常强

# PyTorch 之所以叫 PyTorch，就是因为他是用 Python 实现的，除非是非常底层的如矩阵计算才会用 C/C++ 实现，然后用 Python 封装。众所周知，Python 代码的可阅读性和 C++ 完全不在一个层次（笑）。

# 举个例子，PyTorch 实现的 [torchvision](https://github.com/pytorch/vision/tree/master/torchvision) 模块包含了很多图像的预处理、加载、预训练模型等，全部是用 Python 实现，可以去阅读里面的代码实现。比如 [图片翻转](https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py)、[AlexNet 在 PyTorch 中是如何实现的](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)。而 TensorFlow 就麻烦的多了。

# #### 文档更有好

# PyTorch 拥有非常好的 API 文档、入门教程。对于每个实现的 Loss Function 都会介绍他的实现原理、文章地址、数学公式等，可能目的就是使用户能更理解框架。有很多说法是 TensorFlow 的文档更多，官方入门教程有中文。但其实吧，一个外国公司翻译的东西，总会有点理解上的问题。这些文档读上去非常怪异，还不如直接读原文。

# TensorFlow 还有个缺陷就是文档写的很乱，比如 [TensorFlow 教程](http://www.tensorfly.cn/tfdoc/get_started/introduction.html) 光 MNIST 就写了几个，让人眼花缭乱。而 [PyTorch Tutorials](https://pytorch.org/tutorials/) 展示了很多的 Notebook 从实际应用进行学习。

# ### TensorFlow 的优势

# #### 可视化

# TensorFlow 的 Tensorboard 非常好用，可以可视化模型、曲线等等。Tensorboard 会将数据保存到本地用来自己自定义可视化。

# 当然 PyTorch 也有可视化工具，例如 Facebook 提供的 [Visdom](https://github.com/facebookresearch/visdom) 可以很方便管理很多模型，但是不能导出数据。另外 PyTorch 还可以通过 [tensorboardX](https://github.com/lanpa/tensorboardX) 调用 Tensorboard 进行可视化（虽然不能使用所有的功能）。

# 总体来说 TensorFlow 在这方面要稍强于 PyTorch。

# #### 部署

# 无论是在服务器部署还是移动端部署，PyTorch 都几乎是完败的，毕竟 Google 在 AI 领域的地位是独一无二的。

# 服务器部署 TensorFlow 有 [TensorFlow Serving](https://github.com/tensorflow/serving)，移动端部署有 [TensorFlow Lite](https://www.tensorflow.org/lite/) 将模型压缩到极致，运行速度也非常快，支持 Android 和 iOS，还有 [TensorFlow.js](https://js.tensorflow.org/) 直接调用 WebGL 预测模型。

# 当然 PyTorch 也一直在补足这方面的缺点，1.0 版本之后将 Caffe2 合并至 PyTorch，引入 JIT 等等，使 PyTorch 也有了部署能力，可以查看 [The road to 1.0: production ready PyTorch](https://pytorch.org/blog/the-road-to-1_0/#other-changes-and-improvements) 进一步了解。当然还有其他的如将 PyTorch Model 转换成其他框架的模型进行部署，也可以自己用 Flask、Django 等部署，只是比较麻烦。

# #### 社区

# TensorFlow 的市场占有率非常高，用的人多，网络上有非常多的资源、博客等介绍 TensorFlow，而 PyTorch 相对来说比较新，所以可能更多的是英文，现在 PyTorch 也在慢慢的追赶。

# 另外很多云服务商都提供支持 TensorFlow 的机器学习平台运行云服务器训练模型，但是不一定支持 PyTorch。

# ### 框架选择

# 没有十全十美的框架，即使是 TensorFlow 发展这么多年也一直存在很多问题。在 [TensorFlow 有哪些令人难以接受的地方？](https://www.zhihu.com/question/63342728) 和 [PyTorch 有哪些坑/bug？](https://www.zhihu.com/question/67209417) 都有详细的讨论。只能根据自己的需求来选择合适的框架。

# 具体讨论到如何选择框架，我们根据不同人群介绍:

# #### 学生或者科研人员

# 学生以及科研人员时间比较多，可以花很多时间来学习。最好的选择是两个框架都熟悉，主要使用一个框架的同时对另外一个也有一定了解，至少看得懂代码。
# 
# 在学术领域用各种框架的都有，但是主要还是集中在 TensorFlow 和 PyTorch （Caffe 主要集中在计算机视觉）。如果是学生但是以后会工作，Researcher 但是偏向工程或需要工程方面的应用，那么可能需要偏重于 TensorFlow 的使用上。如果是学生但是偏向于科研，Researcher 希望能够快速实现自己的算法，跑出一个原型，那么灵活的 PyTorch 会更适合。

# #### 已经工作

# 对于已经工作的来说，时间可能并不充裕，所以根据自己的能力选择一门框架深入理解。
# 
# 来学习这门课程，肯定是基于对人工智能、机器学习感兴趣的。如果对以后希望从事这个行业，拿机器学习当饭碗，那么必然是选择 TensorFlow。如果仅仅是基于兴趣，希望接触新的行业，那么 PyTorch 更容易入门，是个更好的选择。

# 综上所述，可能需要根据自己的偏向来选择合适的框架。并不是按照上面所以理解的就一定要选择某一个。框架毕竟是框架，而不是基础知识，能理解框架底层是如何运行的，比如说 Batch Norm 是如何实现的能力，才是最重要的。希望可以意思到，框架只是一个工具，如果不理解原理，也就只能当一个外围人员，俗称调包侠。

# ## 实验总结

# 在本节实验中，我们了解了 PyTorch 中最重要的几个组件，它们分别是：Tensor，Autograd，nn 以及 Optimizer，了解并掌握这些组件之后就可以完成深度神经网络的搭建了，当然不同的深度神经网络结构我们将在之后的课程中详细了解。在本节实验中积累的经验非常重要，因为你在以后的实验中会一直用到今天学习的知识点。回顾这些知识点有：
# 
# - PyTorch 使用
# - Tensor 张量
# - Autograd 自动微分
# - nn 神经网络组件
# - Optimizer 优化器
# - 深度学习框架对比与选择

# ---

# ## 本周思维导图

# 学习完本周的内容，我们总结知识点并绘制思维导图。思维导图是一种非常高效的学习手段，我们非常推荐你在学习的过程中自行梳理知识点。可以通过新标签页面打开图片查看原图。

# ![image](https://doc.shiyanlou.com/document-uid214893labid7506timestamp1542013331542.png)

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
