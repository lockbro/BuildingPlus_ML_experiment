
# coding: utf-8

# <img style="float: right;" src="https://img.shields.io/badge/%E6%A5%BC+-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98-red.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>

# # 朴素贝叶斯

# ---

# ### 实验介绍
# 
# 在分类预测中，以概率论作为基础的算法比较少，而朴素贝叶斯就是其中之一。朴素贝叶斯算法实现简单，且预测分类的效率很高，是一种十分常用的算法。本实验主要从贝叶斯定理和参数估计两个方面讲解朴素贝叶斯算法的原理并结合数据进行实现，最后通过一个例子进行实战练习。

# ### 实验知识点
# 
# - 先验概率
# - 后验概率
# - 贝叶斯定理
# - 极大似然估计
# - 贝叶斯估计
# - 朴素贝叶斯实现
# - 使用 scikit-learn 完成朴素贝叶斯预测

# ### 实验目录
# 
# - <a href="#朴素贝叶斯基础">朴素贝叶斯基础</a>
# - <a href="#算法实现">算法实现</a>
# - <a href="#朴素贝叶斯预测实战">朴素贝叶斯预测实战</a>
# - <a href="#实验总结">实验总结</a>

# ---

# ## 朴素贝叶斯基础

# ### 基本概念

# 朴素贝叶斯的数学理论基础源于概率论。所以，在学习朴素贝叶斯算法之前，首先对本实验中涉及到的概率论知识做简要讲解。

# #### 条件概率
# 
# 条件概率就是指事件 $A$ 在另外一个事件 $B$ 已经发生条件下的概率。如图所示 ：

# <img src="https://doc.shiyanlou.com/document-uid214893labid6671timestamp1531710802977.png" width="370" height="370">

# 其中： 
# - $P(A)$ 表示 $A$ 事件发生的概率。
# - $P(B)$ 表示 $B$ 事件发生的概率。
# - $P(AB)$ 表示 $A, B$ 事件同时发生的概率。 
# 
# 而最终计算得到的 $P(A \mid B)$ 便是条件概率，表示在 $B$ 事件发生的情况下 $A$ 事件发生的概率。

# #### 贝叶斯定理

# 上面提到了条件概率的基本概念，那么当知道事件 $B$ 发生的情况下事件 $A$ 发生的概率 $P(A \mid B)$，如何求 $P(B \mid A)$ 呢？贝叶斯定理应运而生。根据条件概率公式可以得到:

# $$P(B \mid A)=\frac{P(AB)}{P(A)} \tag1$$  

# 而同样通过条件概率公式可以得到：

# $$P(AB)=P(A \mid B)*P(B) \tag2$$  
#   

# 将 (2) 式带入 (1) 式便可得到完整的贝叶斯定理：

# $$P(B \mid A)=\frac{P(AB)}{P(A)}=\frac{P(A \mid B)*P(B)}{P(A)} \tag{3}$$

# 以下，通过一张图来完整且形象的展示条件概率和贝叶斯定理的原理。

# ![image](https://doc.shiyanlou.com/document-uid214893labid6671timestamp1531710804013.png)

# #### 先验概率

# 先验概率（Prior Probability）指的是根据以往经验和分析得到的概率。例如以上公式中的 $P(A), P(B)$,又例如：$X$ 表示投一枚质地均匀的硬币，正面朝上的概率，显然在我们根据以往的经验下，我们会认为 $X$ 的概率 $P(X) = 0.5$ 。其中 $P(X) = 0.5$ 就是先验概率。

# #### 后验概率

# 后验概率（Posterior Probability）是事件发生后求的反向条件概率；即基于先验概率通过贝叶斯公式求得的反向条件概率。例如公式中的 $P(B|A)$ 就是通过先验概率 $P(A)$和$P(B)$ 得到的后验概率，其通俗的讲就是「执果寻因」中的「因」。

# ### 什么是朴素贝叶斯

# 朴素贝叶斯（Naive Bayes）就是将贝叶斯原理以及条件独立结合而成的算法，其思想非常的简单，根据贝叶斯公式：

# $$
# P(B \mid A)=\frac{P(A \mid B)*P(B)}{P(A)} \tag{4}
# $$ 

# 变形表达式为：

# $$
# P(类别 \mid 特征)=\frac{P(特征 \mid 类别) * P(类别)}{P(特征)} \tag{5}
# $$

# 公式（5）利用先验概率，即特征和类别的概率；再利用不同类别中各个特征的概率分布，最后计算得到后验概率，即各个特征分布下的预测不同的类别。

# 利用贝叶斯原理求解固然是一个很好的方法，但实际生活中数据的特征之间是有相互联系的，在计算 $P(特征|类别)$ 时，考虑特征之间的联系会比较麻烦，而朴素贝叶斯则人为的将各个特征割裂开，认定特征之间相互独立。

# 朴素贝叶斯中的「朴素」，即条件独立，表示其假设预测的各个属性都是相互独立的,每个属性独立地对分类结果产生影响，条件独立在数学上的表示为：$P(AB)=P(A)*P(B)$。这样，使得朴素贝叶斯算法变得简单，但有时会牺牲一定的分类准确率。对于预测数据，求解在该预测数据的属性出现时各个类别的出现概率，将概率值大的类别作为预测数据的类别。

# ## 朴素贝叶斯算法实现

# 前面主要介绍了朴素贝叶斯算法中几个重要的概率论知识，接下来我们对其进行具体的实现，算法流程如下：

# **第 1 步**：设 $X = \left \{ a_{1},a_{2},a_{3},…,a_{n} \right \}$ 为预测数据，其中 $a_{i}$ 是预测数据的特征值。
# 
# **第 2 步**：设 $Y = \left \{y_{1},y_{2},y_{3},…,y_{m} \right \}$ 为类别集合。
# 
# **第 3 步**：计算 $P(y_{1}\mid x)$, $P(y_{2}\mid x)$, $P(y_{3}\mid x)$, $…$, $P(y_{m}\mid x)$。
# 
# **第 4 步**：寻找 $P(y_{1}\mid x)$, $P(y_{2}\mid x)$, $P(y_{3}\mid x)$, $…$, $P(y_{m}\mid x)$ 中最大的概率 $P(y_{k}\mid x)$ ，则 $x$ 属于类别 $y_{k}$。

# ### 生成示例数据

# 下面我们利用 python 完成一个朴素贝叶斯算法的分类。首先生成一组示例数据：由 `A` 和 `B`两个类别组成，每个类别包含 `x`,`y`两个特征值，其中 `x` 特征包含`r,g,b`（红，绿，蓝）三个类别，`y`特征包含`s,m,l`（小，中，大）三个类别，如同数据 $X = [g,l]$。

# In[ ]:


"""生成示例数据
"""
import pandas as pd


def create_data():
    data = {"x": ['r', 'g', 'r', 'b', 'g', 'g', 'r', 'r', 'b', 'g', 'g', 'r', 'b', 'b', 'g'],
            "y": ['m', 's', 'l', 's', 'm', 's', 'm', 's', 'm', 'l', 'l', 's', 'm', 'm', 'l'],
            "labels": ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B']}
    data = pd.DataFrame(data, columns=["labels", "x", "y"])
    return data


# **☞ 动手练习：**

# In[1]:


import pandas as pd

def create_data():
    data = {
        "x": ['r', 'g', 'r', 'b', 'g', 'g', 'r', 'r', 'b', 'g', 'g', 'r', 'b', 'b', 'g'],
        "y": ['m', 's', 'l', 's', 'm', 's', 'm', 's', 'm', 'l', 'l', 's', 'm', 'm', 'l'],
        "labels": ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
    }
    data = pd.DataFrame(data, columns=["labels", "x", "y"])
    return data


# 在创建好数据后，接下来进行加载数据，并进行预览。

# In[2]:


"""加载并预览数据
"""
data = create_data()
data


# ### 参数估计

# 根据朴素贝叶斯的原理，最终分类的决策因素是比较 $\left \{ P(类别 1 \mid 特征),P(类别 2 \mid 特征),…,P(类别 m \mid 特征) \right \}$ 各个概率的大小，根据贝叶斯公式得知每一个概率计算的分母 $P(特征)$ 都是相同的，只需要比较分子 $P(类别)$ 和 $P(特征 \mid 类别)$ 乘积的大小。

# 那么如何得到 $P(类别)$,以及 $P(特征\mid 类别)$呢？在概率论中，可以应用**极大似然估计法**以及**贝叶斯估计法**来估计相应的概率。

# #### 极大似然估计

# 什么是极大似然？下面通过一个简单的例子让你有一个形象的了解：

# ![image](https://doc.shiyanlou.com/document-uid214893labid6671timestamp1531710804576.png)

# > **前提条件：**假如有两个外形完全相同箱子，甲箱中有 `99` 个白球，`1` 个黑球；乙箱中有 `99` 个黑球，`1` 个白球。
# 
# > **问题：**当我们进行一次实验，并取出一个球，取出的结果是白球。那么，请问白球是从哪一个箱子里取出的？

# 我相信，你的第一印象很可能会是白球从甲箱中取出。因为甲箱中的白球数量多，所以这个推断符合人们经验。其中「最可能」就是「极大似然」。而极大似然估计的目的就是利用已知样本结果，反推最有可能造成这个结果的参数值。

# 极大似然估计提供了一种给定观察数据来评估模型参数的方法，即：「模型已定，参数未知」。通过若干次试验，观察其结果，利用试验结果得到某个参数值能够使样本出现的概率为最大，则称为极大似然估计。

# 在概率论中求解极大似然估计的方法比较复杂，基于实验，我们将讲解 $P(B)$ 和 $P(B/A)$ 是如何通过极大似然估计得到的。$P(种类)$ 用数学的方法表示 ：

# $$
# P(y_{i}=c_{k})=\frac{\sum_{N}^{i=1}I(y_{i}=c_{k})}{N},k=1,2,3,…,m \tag{6}
# $$

# 公式(6)中的 $y_{i}$ 表示数据的类别，$c_{k}$ 表示每一条数据的类别。
# 
# 你可以通俗的理解为，在现有的训练集中，每一个类别所占总数的比例，例如:**生成的数据**中 $P(Y=A)=\frac{8}{15}$，表示训练集中总共有 15 条数据，而类别为 `A` 的有 8 条数据。  

# 下面我们用 Python 代码来实现先验概率 $P(种类)$ 的求解：

# In[6]:


"""P(种类) 先验概率计算
"""


def get_P_labels(labels):
    labels = list(labels)  # 转换为 list 类型
    P_label = {}  # 设置空字典用于存入 label 的概率
    for label in labels:
        P_label[label] = labels.count(
            label) / float(len(labels))  # p = count(y) / count(Y)
    return P_label


P_labels = get_P_labels(data["labels"])
P_labels


# In[5]:


def get_P_labels(labels):
    labels = list(labels)
    P_label = {}
    for label in labels:
        P_label[label] = labels.count(label) / float(len(labels))
    return P_label

P_labels = get_P_labels(data["labels"])
P_labels


# $P(特征 \mid 种类)$ 由于公式较为繁琐这里先不给出，直接用叙述的方式能更清晰地帮助理解：

# 实际需要求的先验估计是特征的每一个类别对应的每一个种类的概率，例如：**生成数据** 中 $P(x_{1}="r" \mid Y=A)=\frac{4}{8}$， `A` 的数据有 8 条，而在种类为 `A` 的数据且特征 `x` 为 `r`的有 4 条。

# 同样我们用代码将先验概率 $P(特征 \mid 种类)$ 实现求解：

# 首先我们将特征按序号合并生成一个 `numpy` 类型的数组。

# In[ ]:


"""导入特征数据并预览
"""
import numpy as np

train_data = np.array(data.iloc[:, 1:])
train_data


# In[7]:


import numpy as np

# 选出两列特征
train_data = np.array(data.iloc[:, 1:])
train_data


# 在寻找属于某一类的某一个特征时，我们采用对比索引的方式来完成。  
# 开始得到每一个类别的索引：

# In[ ]:


"""类别 A,B 索引
"""
labels = data["labels"]
label_index = []
for y in P_labels.keys():
    temp_index = []
    # enumerate 函数返回 Series 类型数的索引和值，其中 i 为索引，label 为值
    for i, label in enumerate(labels):
        if (label == y):
            temp_index.append(i)
        else:
            pass
    label_index.append(temp_index)
label_index


# In[10]:


labels = data["labels"]
label_index = []
for y in P_labels.keys():
    temp_index = []
    for i, label in enumerate(labels):
        if(label == y):
            temp_index.append(i)
        else:
            pass
    label_index.append(temp_index)
label_index


# 得到 `A` 和 `B` 的索引，其中是`A`类别为前 $8$ 条数据，`B`类别为后 $7$ 条数据。

# 在得到类别的索引之后，接下来就是找到我们需要的特征为 `r`的索引值。

# In[ ]:


"""特征 x 为 r 的索引
"""
x_index = [i for i, feature in enumerate(train_data[:, 0]) if feature == 'r']  # 效果等同于求类别索引中 for 循环
x_index


# In[12]:


x_index = [i for i, feature in enumerate(train_data[:, 0]) if feature == 'r']
x_index


# 得到的结果为 $x$ 特征值为 $r$ 的数据索引值。

# 最后通过对比类别为 `A` 的索引值，计算出既符合 `x = r` 又符合 `A` 类别的数据在 `A` 类别中所占比例。

# In[ ]:


x_label = set(x_index) & set(label_index[0])
print('既符合 x = r 又是 A 类别的索引值：', x_label)
x_label_count = len(x_label)
print('先验概率 P(r|A):', x_label_count / float(len(label_index[0])))


# In[13]:


x_label = set(x_index) & set(label_index[0])
print('既符合 x = r 又是 A 类别的索引值：', x_label)
x_label_count = len(x_label)
print("先验概率P(r|A):", x_label_count / float(len(label_index[0])))


# 为了方便后面函数调用，我们将求 $P(特征\mid 种类)$ 代码整合为一个函数。

# In[16]:


"""P(特征∣种类) 先验概率计算
"""


def get_P_fea_lab(P_label, features, data):
    P_fea_lab = {}
    train_data = data.iloc[:, 1:]
    train_data = np.array(train_data)
    labels = data["labels"]
    for each_label in P_label.keys():
        label_index = [i for i, label in enumerate(
            labels) if label == each_label]  # labels 中出现 y 值的所有数值的下标索引
        # features[0] 在 trainData[:,0] 中出现的值的所有下标索引
        for j in range(len(features)):
            feature_index = [i for i, feature in enumerate(
                train_data[:, j]) if feature == features[j]]
            # set(x_index)&set(y_index) 列出两个表相同的元素
            fea_lab_count = len(set(feature_index) & set(label_index))
            key = str(features[j]) + '|' + str(each_label)
            P_fea_lab[key] = fea_lab_count / float(len(label_index))
    return P_fea_lab


features = ['r', 'm']
get_P_fea_lab(P_labels, features, data)


# 可以得到当特征 `x` 和 `y` 的值为 `r` 和 `m` 时，在不同类别下的先验概率。

# #### 贝叶斯估计

# 在做极大似然估计时，若类别中缺少一些特征，则就会出现概率值为 `0` 的情况。此时，就会影响后验概率的计算结果，使得分类产生偏差。而解决这一问题最好的方法就是采用贝叶斯估计。  

# 在计算先验概率 $P(种类)$ 中，贝叶斯估计的数学表达式为：

# $$
# P(y_{i}=c_{k})=\frac{\sum_{N}^{i=1}I(y_{i}=c_{k})+\lambda }{N+k\lambda} \tag{8}
# $$

# 其中 $\lambda \geq 0$ 等价于在随机变量各个取值的频数上赋予一个正数，当 $\lambda=0$ 时就是极大似然估计。在平时常取 $\lambda=1$，这时称为拉普拉斯平滑。例如：**生成数据** 中，$P(Y=A)=\frac{8+1}{15+2*1}=\frac{9}{17}$,取 $\lambda=1$ 此时由于一共有 `A`，`B` 两个类别，则 `k` 取 2。

# 同样计算 $P(特征 \mid 种类)$ 时，也是给计算时的分子分母加上拉普拉斯平滑。例如：**生成数据** 中，$P(x_{1}="r" \mid Y=A)=\frac{4+1}{8+3*1}=\frac{5}{11}$ 同样取 $\lambda=1$ 此时由于 `x` 中有 `r`, `g`, `b` 三个种类，所以这里 k 取值为 3。

# ### 朴素贝叶斯算法实现

# 通过上面的内容，相信你已经对朴素贝叶斯算法原理有一定印象。接下来，我们对朴素贝叶斯分类过程进行完整实现。其中，参数估计方法则使用极大似然估计。
# * 注：分类器实现的公式，请参考《机器学习》- 周志华 P151 页*

# In[21]:


"""朴素贝叶斯分类器
"""


def classify(data, features):
    # 求 labels 中每个 label 的先验概率
    labels = data['labels']
    P_label = get_P_labels(labels)
    P_fea_lab = get_P_fea_lab(P_label, features, data)

    P = {}
    P_show = {}  # 后验概率
    for each_label in P_label:
        P[each_label] = P_label[each_label]
        for each_feature in features:
            key = str(each_label)+'|'+str(features)
            P_show[key] = P[each_label] *                 P_fea_lab[str(each_feature) + '|' + str(each_label)]
            P[each_label] = P[each_label] *                 P_fea_lab[str(each_feature) + '|' +
                          str(each_label)]  # 由于分母相同，只需要比较分子
    print(P_show)
    features_label = max(P, key=P.get)  # 概率最大值对应的类别
    return features_label


# In[28]:


def classify(data, features):
    # 先验概率：P（种类） P（特征|种类）
    labels = data['labels']
    # {'A': 0.5333333333333333, 'B': 0.4666666666666667}
    P_label = get_P_labels(labels)
    # {'r|A': 0.5, 'm|A': 0.375, 'r|B': 0.14285714285714285, 'm|B': 0.42857142857142855}
    P_fea_lab = get_P_fea_lab(P_label, features, data)
    
    # 后验概率
    P = {}
    P_show ={}
    for each_label in P_label:
        P[each_label] = P_label[each_label]
        for each_feature in features:
            key = str(each_label) + '|' + str(features)
            P_show[key] = P[each_label] * P_fea_lab[str(each_feature) + '|' + str(each_label)]
            P[each_label] = P[each_label] * P_fea_lab[str(each_feature) + '|' +str(each_label)]
            # 由于分母相同，只需要比较分子
    print(P_show)
    print(P)
    features_label = max(P, key=P.get)
    return features_label


# In[29]:


classify(data, ['r', 'm'])


# 对于特征为 `[r,m]` 的数据通过朴素贝叶斯分类得到不同类别的概率值，经过比较后分为 `A` 类。

# ### 朴素贝叶斯的三种常见模型

# 了解完朴素贝叶斯算法原理后，在实际数据中，我们可以依照特征的数据类型不同，在计算先验概率方面对朴素贝叶斯模型进行划分，并分为：**多项式模型**，**伯努利模型**和**高斯模型**。

# #### 多项式模型

# 当特征值为离散时，常常使用多项式模型。事实上，在以上实验的参数估计中，我们所应用的就是多项式模型。为避免概率值为 0 的情况出现，多项式模型采用的是贝叶斯估计。

# #### 伯努利模型

# 与多项式模型一样，伯努利模型适用于离散特征的情况，所不同的是，伯努利模型中每个特征的取值只能是 `1` 和 `0`（以文本分类为例，某个单词在文档中出现过，则其特征值为 `1`，否则为 `0`）。

# 在伯努利模型中，条件概率 $P(x_{i} \mid y_{k})$ 的计算方式为：
# 
# - 当特征值 $x_{i}=1$ 时，$P(x_{i} \mid y_{k})=P(x_{i}=1 \mid y_{k})$;  
# 
# - 当特征值 $x_{i}=0$ 时，$P(x_{i} \mid y_{k})=P(x_{i}=0 \mid y_{k})$。

# #### 高斯模型

# 当特征是连续变量的时候，在不做平滑的情况下，运用多项式模型就会导致很多 $P(x_{i} \mid y_{k})=0$，此时即使做平滑，所得到的条件概率也难以描述真实情况。所以处理连续的特征变量，采用高斯模型。高斯模型是假设连续变量的特征数据是服从高斯分布的，高斯分布函数表达式为：

# $$P(x_{i}|y_{k})=\frac{1}{\sqrt{2\pi}\sigma_{y_{k},i}}exp(-\frac{(x-\mu_{y_{k},i}) ^{2}}{2\sigma ^{2}_{y_{k}},i})$$

# 其中：
# 
# - $\mu_{y_{k},i}$ 表示类别为 $y_{k}$ 的样本中，第 $i$ 维特征的均值。  
# - $\sigma ^{2}_{y_{k}},i$ 表示类别为 $y_{k}$ 的样本中，第 $i$ 维特征的方差。  

# 高斯分布示意图如下：

# ![image](https://doc.shiyanlou.com/document-uid214893labid6671timestamp1531710804887.png)

# ## 朴素贝叶斯分类预测

# 接下来，我们应用朴素贝叶斯算法模型对真实数据进行分类预测。

# ### 数据集介绍及预处理

# 本次应用到的数据集为企业运营评估数据集 `course-10-company.csv`，总共有 `250` 条数据，每条数据包括 6 个特征值，分别为：

# - `industrial_risk`: 产业风险
# - `management_risk`: 管理风险
# - `finacial_flexibility`: 资金灵活性
# - `credibility`: 信用度
# - `competitiveness`: 竞争力
# - `operating_risk`: 操作风险

# 其中，每条特征值包括 `3` 个等级，分别为：
# 
# - `Positive`: 好
# - `Average`: 中
# - `Negative`: 坏 
# 
# `3` 个等级分别用 `P`, `A`, `N` 代替。

# 通过这些特征对该企业是否会破产进行分类预测，其中：
# 
# - `NB`: 表示不会破产
# - `B`: 表示会破产

# 下面，我们导入数据集并预览：

# In[31]:


get_ipython().system('wget -nc http://labfile.oss.aliyuncs.com/courses/1081/course-10-company.csv')


# In[39]:


"""导入数据集并预览
"""
import pandas as pd

enterprise_data = pd.read_csv('course-10-company.csv')
enterprise_data.head()


# 由于本实验中使用 `scikit-learn` 模块，为遵循 `scikit-learn` 函数的输入规范，需要将数据集进行预处理。

# In[40]:


enterprise_data = enterprise_data.replace(
    {"P": 1, "A": 2, "N": 3, "NB": 0, "B": 1})  # 对元素值进行替换
enterprise_data


# ### 数据集划分

# 加载好数据集之后，为了实现朴素贝叶斯算法，同样我们需要将数据集分为 **训练集**和**测试集**，依照经验：**训练集**占比为 70%，**测试集**占 30%。
# 
# 同样在此我们使用 `scikit-learn` 模块的 `train_test_split` 函数完成数据集切分。  

# ```python
# from sklearn.model_selection import train_test_split
# 
# x_train,x_test, y_train, y_test =train_test_split(train_data,train_target,test_size=0.4, random_state=0)
# ```

# 其中：
# 
# - `x_train`,`x_test`, `y_train`, `y_test` 分别表示，切分后的特征的训练集，特征的测试集，标签的训练集，标签的测试集；其中特征和标签的值是一一对应的。  
# - `train_data`,`train_target`分别表示为待划分的特征集和待划分的标签集。
# - `test_size`：测试样本所占比例。
# - `random_state`：随机数种子,在需要重复实验时，保证在随机数种子一样时能得到一组一样的随机数。

# In[ ]:


"""数据集划分
"""
from sklearn.model_selection import train_test_split

# 得到企业运营评估数据集中 feature 的全部序列: industrial_risk, management_risk 等特征
feature_data = enterprise_data.iloc[:, :-1]
label_data = enterprise_data["label"]  # 得到企业运营评估数据集中 label 的序列
x_train, x_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.3, random_state=4)

x_test  # 输出企业运营评估数据测试集查看


# In[41]:


from sklearn.model_selection import train_test_split

# 除了label以外的所有特征
features_data = enterprise_data.iloc[:, :-1]
label_data = enterprise_data["label"]
x_train, x_test, y_train, y_test = train_test_split(
    features_data, label_data, test_size=0.3, random_state=4
)


# ### 建立分类预测模型

# 数据集表现出离散型的特征。所以，根据上文中提到的模型选择经验，这里选用多项式模型。在前面的实验中我们采用 `python` 对朴素贝叶斯算法进行实现，下面我们通过 `scikit-learn` 来对其进行实现。 
# 
# 在 `scikit-learn` 朴素贝叶斯类及参数如下：

# ```python
# MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# ```

# 其中：
# 
# - `alpha` 表示平滑参数，如拉普拉斯平滑则 `alpha=1`。
# - `fit_prior` 表示是否使用先验概率，默认为 `True`。
# - `class_prior` 表示类的先验概率。

# 常用方法:  
# 
# - `fit(x,y)`选择合适的贝叶斯分类器。
# - `predict(X)` 对数据集进行预测返回预测结果。

# In[ ]:


"""利用 scikit-learn 构建多项式朴素贝叶斯分类器
"""
from sklearn.naive_bayes import MultinomialNB


def sk_classfy(x_train, y_train, x_test):
    sk_clf = MultinomialNB(alpha=1.0, fit_prior=True)  # 定义多项式模型分类器
    sk_clf.fit(x_train, y_train)  # 进行模型训练
    return sk_clf.predict(x_test)


y_predict = sk_classfy(x_train, y_train, x_test)
y_predict


# In[43]:


from sklearn.naive_bayes import MultinomialNB

def sk_classfy(x_train, y_train, x_test):
    sk_clf = MultinomialNB(alpha=1, fit_prior=True)
    sk_clf.fit(x_train, y_train)
    return sk_clf.predict(x_test)

y_predict = sk_classfy(x_train, y_train, x_test)
y_predict


# ### 分类准确率计算

# 当我们训练好模型并进行分类预测之后，可以通过比对预测结果和真实结果得到预测的准确率。

# $$accur=\frac{\sum_{i=1}^{N}I(\bar{y_{i}}=y_{i})}{N}\tag{11}$$

# 公式(11)中 $N$ 表示数据总条数，$\bar{y_{i}}$ 表示第 $i$ 条数据的种类预测值，$y_{i}$ 表示第 $i$ 条数据的种类真实值，$I$ 同样是指示函数，表示 $\bar{y_{i}}$ 和 $y_{i}$ 相同的个数。

# In[ ]:


"""准确率计算
"""


def get_accuracy(test_labels, pred_labels):
    correct = np.sum(test_labels == pred_labels)  # 计算预测正确的数据个数
    n = len(test_labels)  # 总测试集数据个数
    accur = correct/n
    return accur


get_accuracy(y_test, y_predict)


# In[45]:


def get_accuracy(test_labels, pred_labels):
    correct = np.sum(test_labels == pred_labels)
    n = len(test_labels)
    accur = correct / n
    return accur
get_accuracy(y_test, y_predict)


# 可以看到，通过朴素贝叶斯算法进行分类可以得到准确率为 0.78。

# ## 实验总结

# 本节实验从概率论的相关概念讲起，阐述了朴素贝叶斯算法的核心定理，即贝叶斯定理。同时，实验应用理论结合代码的方式讲解了朴素贝叶斯的原理以及实现过程。特别注意的是，朴素贝叶斯算法中涉及到的概率论知识点容易混淆，建议通过结合实际的示例进行区分。

# 最后，回顾本实验的知识点有：
# 
# - 先验概率
# - 后验概率
# - 贝叶斯定理
# - 极大似然估计
# - 贝叶斯估计
# - 朴素贝叶斯实现
# - 使用 scikit-learn 完成朴素贝叶斯预测

# 关于贝叶斯定理，这里有一个有趣的视频，希望能加深大家对该定理的理解。

# <center><video width='800px' controls src="http://labfile.oss.aliyuncs.com/courses/1081/beyes_video.mp4" /></center>

# <div style="color: #999;font-size: 12px;text-align: center;">如何用贝叶斯方法帮助内容审核 | 视频来源：[回形针PaperClip](https://weibo.com/u/6414205745?is_all=1)</div>

# ---

# <img src="https://img.shields.io/badge/%E5%AE%9E%E9%AA%8C%E6%A5%BC-%E7%89%88%E6%9D%83%E6%89%80%E6%9C%89-lightgrey.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iOXB4IiBoZWlnaHQ9IjE0cHgiIHZpZXdCb3g9IjAgMCA5IDE0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1Mi4yICg2NzE0NSkgLSBodHRwOi8vd3d3LmJvaGVtaWFuY29kaW5nLmNvbS9za2V0Y2ggLS0+CiAgICA8dGl0bGU+c2hpeWFubG91X2xvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZyBpZD0iUGFnZS0xIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0ic2hpeWFubG91X2xvZ28iIGZpbGw9IiMwOEJGOTEiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxnIGlkPSLlvaLnirZfMTBf5ou36LSdXzNfMzRfIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjUyNTU4MTQsMC4yNjA0NjUxMTYgQzQuNTI1NTgxNCwwLjI2MDQ2NTExNiAxLjQ5NzY3NDQyLDQuMiAwLjU1MzQ4ODM3Miw1Ljk5MDY5NzY3IEMtMC40MjMyNTU4MTQsNy44MTM5NTM0OSAxLjU5NTM0ODg0LDEwLjA5MzAyMzMgMS41OTUzNDg4NCwxMC4wOTMwMjMzIEMxLjU5NTM0ODg0LDEwLjA5MzAyMzMgNC44MTg2MDQ2NSw1LjE3Njc0NDE5IDUuMjA5MzAyMzMsNC42MjMyNTU4MSBDNi4zNDg4MzcyMSwyLjk2Mjc5MDcgNC41MjU1ODE0LDAuMjYwNDY1MTE2IDQuNTI1NTgxNCwwLjI2MDQ2NTExNiBaIE03LjQyMzI1NTgxLDMuOTA2OTc2NzQgQzcuNDIzMjU1ODEsMy45MDY5NzY3NCA0LjM5NTM0ODg0LDcuODQ2NTExNjMgMy40NTExNjI3OSw5LjYzNzIwOTMgQzIuNDc0NDE4NiwxMS40Mjc5MDcgNC41MjU1ODE0LDEzLjczOTUzNDkgNC41MjU1ODE0LDEzLjczOTUzNDkgQzQuNTI1NTgxNCwxMy43Mzk1MzQ5IDcuNzQ4ODM3MjEsOC44MjMyNTU4MSA4LjEzOTUzNDg4LDguMjY5NzY3NDQgQzkuMjEzOTUzNDksNi42MDkzMDIzMyA3LjQyMzI1NTgxLDMuOTA2OTc2NzQgNy40MjMyNTU4MSwzLjkwNjk3Njc0IFoiIGlkPSLlvaLnirYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+&longCache=true&style=flat-square"></img>
