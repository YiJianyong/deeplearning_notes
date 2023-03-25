# 多层感知机模型

## 1.多层感知机

### 1.1隐藏层

+ softmax中的仿射变换的线性是一个很强的假设，我们的标签通过仿射变换后确实与我们的输入数据相关，那么这种方法确实足够。
+ 线性意味着***单调***假设，任何特征的改变都会造成模型输出的改变。在一些问题中我们可以对数据进行预处理，使线性变得更加合理。
+ 例子：我们想要根据体温预测死亡率。 对体温高于37摄氏度的人来说，温度越高风险越大。 然而，对体温低于37摄氏度的人来说，温度越高风险就越低。我们可以使用**与37摄氏度的距离**作为特征。

+ 我们可以在网络中加入**一个或多个隐藏层**来克服线性模型的限制， 使其能处理更普遍的函数关系类型。
![image](https://user-images.githubusercontent.com/78517435/227703609-c870e7c5-9d9b-4af7-8467-2edd555e30e6.png)


+ 上述多层感知机模型中输入层不涉及计算，只需要完成隐藏层和输出层的计算，因此这个感知机模型的层数为2
+ 可以注意上面的感知机模型是全连接的，具有全连接的多层感知机模型的参数开销会很大。在不改变输入输出大小的情况下可以在参数节约和模型有效性之间权衡


### 1.2激活函数
+ 通过计算加权和并加上偏置来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算。 大多数激活函数都是非线性的。

+ 1.ReLU激活函数
  + 最受欢迎的激活函数，因为实现简单，表现良好
![image](https://user-images.githubusercontent.com/78517435/227706425-c947d5d7-18ac-475b-9dda-f48b43786864.png)

```python
#relu函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```
  + 当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。
 ```python
 #relu函数的导数图
 y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
 ```
   + relu函数表现良好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题

+ 2.sigmoid函数

  + sigmoid函数将一个定义域在R的输入变换为区间（0，1）上的输出，称为压缩函数

![image](https://user-images.githubusercontent.com/78517435/227706600-14372487-1765-4d63-ae9b-76fa647de564.png)

   + sigmoid在隐藏层中已经较少使用， 它在大部分时候被更简单、更容易训练的ReLU所取代。
  ```python
  #绘制sigmoid函数
 y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
  ```

```python
#sigmoid函数的导数
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```
+ 3.tanh函数
  + tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上
 ![image](https://user-images.githubusercontent.com/78517435/227706757-112a48e3-7c75-4af1-a738-6c86b6b7a6de.png)

 ```python
 #绘制tanh函数
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
 ```

```python
# 清除以前的梯度
#导数
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```


## 2.多层感知机的从零实现






## 3.多层感知机的简洁实现




## 4.模型选择、欠拟合和过拟合




## 5.权重衰减





## 6.暂退法





## 7.前向传播、反向传播和计算图




## 8.数值稳定性和模型初始化






