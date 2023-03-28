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

### 2.1初始化模型参数
+ 用到的数据集是784个像素值和10个类别组成，我们可以将每个图像看成具有784个输入特征和10个类别的简单分类数据集

```python
#设置模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

### 2.2激活函数
+ 为了确保我们对模型的细节了如指掌， 我们将实现ReLU激活函数， 而不是直接调用内置的relu函数。
```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

```

### 2.3模型

```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
```

### 2.4损失函数
+ 我们直接使用高级API中的内置函数来计算softmax和交叉熵损失。
```python
loss = nn.CrossEntropyLoss(reduction='none')

```
### 2.5训练

+ 可以直接调用d2l包的train_ch3函数,将迭代周期数设置为10，并将学习率设置为0.1
```python
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

```
+ 评估模型

```python
d2l.predict_ch3(net, test_iter)
```

## 3.多层感知机的简洁实现
+ 通过高级API更加简洁地实现多层感知机模型

```python
import torch
from torch import nn
from d2l import torch as d2l
```
### 3.1 模型

+ 我们添加了2个全连接层（之前我们只添加了1个全连接层）。 第一层是隐藏层，它包含256个隐藏单元，并使用了ReLU激活函数。 第二层是输出层。

```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

```
+ 训练过程的实现

```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

```


## 4.模型选择、欠拟合和过拟合

### 4.1训练误差和泛化误差

+ 训练误差：模型在训练集上计算得到的误差
+ 泛化误差：模型应用在同样从**原始样本的分布中**抽取**无限多**的数据样本时，模型误差的期望
  + 我们永远不能准确计算泛化误差，只能将模型应用于一个独立的测试集来估计泛化误差
  + 我们假设训练数据和测试数据都是从相同的分布中独立提取的，即不违背独立性假设
  + 有时候我们轻微违背独立同分布假设，模型可以依旧运行地很好，但是有些违背独立同分布的假设会引起问题

+ 模型复杂性
  + 通常对于神经网络，我们认为需要更多训练迭代的模型更加复杂，则需要早停的模型就不那么复杂

+ 影响模型泛化的因素
  + 可调整参数的数量：可调整的参数数量（自由度）很大时往往更容易过拟合
  + 参数采用的值：当权重的取值范围较大时，模型可能更容易过拟合
  + 训练样本的数量：样本数量较小时容易过拟合


### 4.2模型选择

+ 验证集
  + 不能依靠测试数据进行模型选择：容易发生数据泄露，导致测试数据发生过拟合
  + 因此常见解决方法：将数据分成三份，训练集、验证集、测试集


+ K折交叉验证
  + 训练数据集稀缺，无法提供足够发数据构成一个合适的验证集
  + 将原始训练数据分成K个不重叠的子集，执行K次模型训练和验证，取k次实验平均结果估计训练和验证误差


### 4.3欠拟合和过拟合
+ 模型训练误差和验证误差都很大
  + 如果模型不能降低训练误差，说明模型过于简单，无法学到东西
  + 如果可以用一个更加复杂的模型降低模型训练误差，称原模型欠拟合

+ 训练误差明显低于验证误差，这表明出现了严重的过拟合


+ 模型复杂度对欠拟合和过拟合的影响

![image](https://user-images.githubusercontent.com/78517435/228162262-58215053-0d0e-45c9-9a9a-886fb4b50485.png)

+ 训练数据集对欠拟合和过拟合的影响
  + 训练集中的样本越少，我们就越有可能过拟合
  + 随着数据量的增加，泛化误差通常会减少。当不减少时即达到了模型的最优


### 4.4多项式回归

+ 通过多项式拟合来探索这些概念

```python
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

```

+ 生成数据集

```python
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

```

```python
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]
```
+ 对模型进行训练和测试


```python
#实现一个函数来评估模型在给定数据集上的损失
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

```

```python
#定义训练函数
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```python
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])

```

```python
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])

```


```python
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)

```

## 5.权重衰减
+ 当我们不能获得大量的数据集，但是我们已经获得了尽可能多的高质量数据，我们可以将重点放在正则化技术上
+ 在训练参数化机器学习模型时，权重衰减是最广泛使用的正则化技术之一



+ 通过例子演示权重衰减
  + 1.生成数据

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

```

```python
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

```


  + 2.从零开始实现：将L2的平方惩罚添加到原始目标函数中
  + 3.初始化模型参数


```python
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

  + 4.定义L2范数惩罚

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

  + 5.定义训练代码

```python
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

```

  + 6.忽略正则化直接训练

```python
train(lambd=0)
```

  + 7.使用权重衰减

```python
train(lambd=3)
```

+ 权重衰减的简洁实现：深度学习框架将权重衰减集成到优化算法中，以便与任何损失函数结合使用。
  + 下面代码中实例化优化器时直接通过weight_decay来指定weight_decay超参数

```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
```

```python
train_concise(0)

```
```python
train_concise(3)
```

## 6.暂退法

+ 在训练过程中，在计算后续层之前向网络的每一层注入噪声，注入噪声会在输入-输出上增强平滑性
+ 暂退法：在整个训练过程的每一次迭代中，在计算下一层之前将当前层的一些节点置零

![image](https://user-images.githubusercontent.com/78517435/228184386-ed9a428f-06b8-43d7-b2e0-0d65a6577ccb.png)


+ 从零开始实现暂退法

```python
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
``

+ 测试上面的函数

```python
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

``

+ 定义模型参数
```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

+ 定义模型

```python
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```


+ 训练模型

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

```

+ 简洁实现暂退法：利用深度学习框架的高级API，只需在每个全连接层之后添加一个Dropout层，将暂退概率作为唯一参数传递给它的构造函数

```python
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);


```

+ 训练模型

```python
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

```


## 7.前向传播、反向传播和计算图






## 8.数值稳定性和模型初始化






