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


















