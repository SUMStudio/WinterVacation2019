# Week4 神经网络

## 0. 神经网络

在神经网络中，使用多层激活函数来模拟神经元。输入节点称为输入层，最终输出的假设函数称为输出层，可以在输入和输出层中间建立隐藏层的中间节点层。

在本次编程练习中，使用神经网络来完成手写数字的分类识别

### 数据集

吴恩达编程练习 ex3 中，提供了 5000 个手写数字样本，每个样本是一个 20*20 的灰度图像。

```
% Load saved matrices from file
load('ex3data1.mat');
% The matrices X and y will now be in your MATLAB environment
```

导入数据集之后，在工作区中可以看到引入的训练样本，y 是一个 5000 维的列向量，是对应的手写样本的标签。

## 1. 向量化逻辑回归

因为给定的数据集有 10 个不同的类别，因此，按照 1 vs all 逻辑模型，来构建多分类器，需要构建 10 个分类器

首先，我们要实现向量化计算代价函数 J 以及代价函数的偏导数。编辑函数 `lrCostFunction.m`

加入以下代码

```
h = sigmoid(X * theta);
J = (sum(-y.* log(h)-(1-y).* log(1-h)))/m + lambda/(2*m)* sum(theta(2:end,:).^2);
grad = X' * (h-y)/m;
grad(2:end, :) = grad(2:end,:) + lambda/m * theta(2:end,:);
```

实际上，这些代码在 ex2 中就已经完成，几乎无改动，可以直接用。

![mark](http://media.sumblog.cn/blog/20190130/b3rQu6EfemTE.png?imageslim)

可以看到，我们的 `lrCostFunction` 函数可以正确计算代价函数值以及偏导数

### 1 vs all 分类器

在之前的编程练习中，分类器代码都由文档给出，这一次，我们要自己实现分类器。对于 k 分类问题来说，使用 1 vs all 方法进行分类需要 k 个分类器。

编写 oneVsAll 函数。其传入的参数为：`X, y num_labels, lambda`

其中：

- X：训练集
- y：训练集的标签，列向量，每一行的值为标签标签号 （1至num_labels)
- num_labels：标签个数
- lambda：正则化参数

按照文档的思路，编写 for 循环，循环 k 次，每次训练一个分类器。由于这次参数较多，文档推荐使用 fmincg 函数来实现参数代价函数的最小化

```matlab
for c = 1:num_labels
    % Set Initial theta
     initial_theta = zeros(n + 1, 1);
     % Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter', 50);
     % Run fmincg to obtain the optimal theta
    [theta] =  fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
    all_theta(c,:) = theta;
```

实际上，这次的分类器训练代码，文档也已经给出，大部分代码直接从注释中就可得到

编写完训练函数之后，运行训练代码：

![mark](http://media.sumblog.cn/blog/20190130/qTOntKbQXygY.png?imageslim)

可以观察到随着迭代次数增加，代价值逐渐减小。至此，提交一次代码进行评测：

![mark](http://media.sumblog.cn/blog/20190130/rSVMnB85Wk9J.png?imageslim)



## 2. 预测 OneVsAll

编写 `predictOneVsAll` 函数，实现对给定输入样本的测试

- [Y,U]=max(A)：返回行向量Y和U，Y向量记录A的每列的最大值，U向量记录每列最大值的行号。
- max(A,[],dim)：dim取1或2。dim取1时，该函数和max(A)完全相同；dim取2时，该函数返回一个列向量，其第i个元素是A矩阵的第i行上的最大值。 

```
A = sigmoid( X * all_theta');
[~,p] = max(A,[],2);
```

运行函数，可以看到其对训练集数据预测的正确率为 94.6%

![mark](http://media.sumblog.cn/blog/20190130/OqY7emPDQOjn.png?imageslim) 

## 3. 神经网络

本次编程练习并不要求实现一个神经网络，本次练习已经给定一个训练好的神经网络，该神经网络共三层：

![mark](http://media.sumblog.cn/blog/20190130/MM0lHvxMnlj0.png?imageslim)

神经网络的参数已经在 `ex3weights.mat` 中给出，在文档中直接加载即可：

![mark](http://media.sumblog.cn/blog/20190130/iupUcnwH7qd5.png?imageslim)

编写神经网络的预测函数 `predict.m`

```
% Add ones to the X data matrix
X = [ones(m, 1) X];
a2 = sigmoid(X*Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
[~,p] = max(a3,[],2);
```

这是一个两层的神经网络，在进行计算时，每一次需要增加一列 1。最终还是取每一行中最大列的索引作为预测的分类。

在训练集上的准确度为 97.52% 比我们之前的线性分类器准确度提高了 3 个百分点。

![mark](http://media.sumblog.cn/blog/20190130/QwPiW27hm7Va.png?imageslim)

![mark](http://media.sumblog.cn/blog/20190130/WwUDlMeL85Iz.png?imageslim)

## 4. 拓展： Matlab 的深度学习包

在课程的拓展文档中对 MATLAB 的深度学习包进行了简要的介绍。

首先，使用 Matlab 打开给定的神经网络：

![mark](http://media.sumblog.cn/blog/20190130/msDjjPchHgtC.png?imageslim)

将样例输入网络，就可以获得预测的输出：

![mark](http://media.sumblog.cn/blog/20190130/080dRbJxMGsh.png?imageslim)

