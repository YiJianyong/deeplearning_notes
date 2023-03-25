# 第三章   线性神经网络

## 1.线性回归

### 1.1 线性回归的基本元素

+ 1.线性模型
  + ![image](https://user-images.githubusercontent.com/78517435/226922757-d397173d-8bcf-4df7-9103-182225746ee4.png)
  + 其中w为权重，权重决定了每个特征对我们预测值的影响。b为偏置，即当所有特征都取值为零时，预测值应该为多少
  + 在机器学习领域，我们使用的常是高维数据集，我们将所有特征放到向量X中，将所有权重放到向量w中，那我们可以用向量的点积形式来表达模型
  + ![image](https://user-images.githubusercontent.com/78517435/226924290-ff36e4f9-751a-473a-a407-b5ab0d24f0b4.png)

+ 2.损失函数
  + 损失函数能够量化目标的实际值和预测值之间的差距
  + 回归问题中最常用的损失函数是平方误差函数

![image](https://user-images.githubusercontent.com/78517435/226925709-e629cda6-2f7a-4d4c-8451-37e965737815.png)

+ 3.解析解
  + 回归模型的解可以用一个简单的公式表示出来，这类解叫做解析解，如下：

![image](https://user-images.githubusercontent.com/78517435/226926707-3129fe95-a6ae-4651-834e-86bdf4311a01.png)

 
 + 4.随机梯度下降
   + 这种方法几乎可以优化所有深度学习模型。 它通过不断地在损失函数递减的方向上更新参数来降低误差。
   + 梯度下降最简单的用法就是计算损失函数关于模型参数的导数
   + 为提高速度，每次计算随机抽取一小批样本进行更新
 
![image](https://user-images.githubusercontent.com/78517435/227081480-45bc1d73-0388-4ce2-bee1-41ec2015cd35.png)
   + 步骤：（1）初始化模型参数，如随机初始化（2）从数据集中随机抽取小批量样本且在负梯度方向上更新参数

![image](https://user-images.githubusercontent.com/78517435/227081838-46250129-707f-465d-899c-b4f7e84cde51.png)


### 1.2矢量化加速

+ 我们经常希望能够同时处理整个小批量的样本，可以利用线性代数库，而不是在Python中编写开销高昂的for循环
+ 我们通过比较，矢量化代码通常会带来数量级的加速

### 1.3正态分布和平方损失

+ 均方误差损失函数可以用于线性回归的一个原因是： 我们假设了观测中包含噪声，其中噪声服从正态分布。 噪声正态分布如下式:

![image](https://user-images.githubusercontent.com/78517435/227082925-1420caaa-e9a5-4a58-8f9a-987c6db07491.png)
![image](https://user-images.githubusercontent.com/78517435/227082982-6b0c7a2a-2dd9-4875-a70b-6a03a7be85ca.png)
+ 我们现在可以写出通过给定x的观测到特定y的似然

![image](https://user-images.githubusercontent.com/78517435/227083212-cc7d265a-8fab-442a-9e2b-a5cf5cef81f6.png)
+ 根据极大似然估计法，参数w和b的最优值是使整个数据集的似然最大的值：

![image](https://user-images.githubusercontent.com/78517435/227083388-7738e530-647d-4109-bf7d-8f2aec8698eb.png)


## 2.softmax回归

### 2.1分类问题
+ 如何表示标签
  + 多标签时：我们可能想到用1，2，3等来表示不同类别（适用于一些类别间存在自然顺序）
  + 不存在自然顺序我们一般使用**独热编码**，它是一个向量，分量与类别一样多，类别对应的分量设置为1，其他分类设置为0

### 2.2网络架构

+ 为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。
+ 为了解决线性模型的分类问题，我们需要和输出一样多的仿射函数。 每个输出对应于它自己的仿射函数。

![image](https://user-images.githubusercontent.com/78517435/227682802-a1ccfcb9-3744-4df3-a2c5-c8bdf507b332.png)

![image](https://user-images.githubusercontent.com/78517435/227683642-9764cd82-44c7-4bd3-a2b6-85b749efa796.png)
+ 由于我们有4个特征和3个可能的输出类别， 我们将需要12个标量来表示权重， 3个标量来表示偏置


### 2.3softmax运算
+ 我们希望模型的输出可以视为属于某一类别的概率。要将输出视为概率，我们必须保证在任何数据上的输出都是非负的且总和为1。
+ softmax函数能够将未规范化的预测变换为非负数并且总和为1，同时让模型保持 可导的性质。

![image](https://user-images.githubusercontent.com/78517435/227685412-3f74be4a-e391-494a-a08a-754274728628.png)

+ 尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。 因此，softmax回归是一个线性模型


### 2.4小批量样本的矢量化

+ 为了提高计算效率并且充分利用GPU，我们通常会对小批量样本的数据执行矢量计算。
+ softmax的矢量计算表达式为：

![image](https://user-images.githubusercontent.com/78517435/227686039-29eb7dcb-1525-48e8-8088-e9d7e4968937.png)

+ 小批量样本的矢量化加快了X和W的矩阵-向量乘法















