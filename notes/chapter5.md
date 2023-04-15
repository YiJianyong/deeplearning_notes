# 深度学习计算

## 1.层和块

+ 从编程角度看，块可以由类表示，它的子类都必须定义一个将其输入转换成输出的前向传播函数

### 1.1自定义块

+ 每个块须提供的功能
  + 将输入数据作为其前向传播函数的参数
  + 通过前向传播函数来生成输出（输出的形状可能与输入的形状不同）
  + 计算其输出关于输入的梯度
  + 存储和访问前向传播计算所需的参数
  + 根据需要初始化模型参数

+ 下面这段代码实现了一个块：它包含一个多层感知机，具有256个隐藏单元的隐藏层和一个10维输出层

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

```
### 1.2顺序块

+ 顺序块中两个关键函数
  + 一种将块逐个追加到列表中的函数
  + 一种前向传播函数，将输入按追加块的顺序传递给块组成的“链条”

+ 下面是顺序块的一个简单实现
  + __init__函数将每个模块追加到有序字典modules中

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

+ 使用我们上述定义的顺序块类重新实现多层感知机

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```



## 2.参数管理




## 3.延后初始化




## 4.自定义层





## 5.读写文件



## 6.GPU













