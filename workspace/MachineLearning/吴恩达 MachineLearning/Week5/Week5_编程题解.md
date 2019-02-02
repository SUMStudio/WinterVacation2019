# Week5 编程题解

## 0. 神经网络

本周在上一周的基础上，仍然是对神经网络进行学习。这次的编程练习，主要是要自己实现反向传播算法。来对神经网络的参数进行优化。

## 1. 代价函数

没有正则化项的代价函数如下：
$$
J ( \theta ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sum _ { k = 1 } ^ { K } \left[ - y _ { k } ^ { （i ） } \log \left( \left( h _ { \theta } \left( x ^ { （i ） } \right) \right) _ { k } \right) - \left( 1 - y _ { k } ^ { ( i ) } \right) \log \left( 1 - \left( h _ { \theta } \left( x ^ { ( i ) } \right) \right) _ { k } \right]\right.
$$
按照文档的要求，文档要求将所有的样本输入进行拆分，在 for 循环中，对所有样本输入逐一计算代价，并进行累加。最后取平均值，得到代价函数值。这里我们先实现一个没有正则化项的代价函数来简化计算。

注意，在进行计算时，需要使用向量化的输出结果 $y$ 来简化计算
$$
y = \left[ \begin{array} { c } { 1 } \\ { 0 } \\ { 0 } \\ { \vdots } \\ { 0 } \end{array} \right] , \left[ \begin{array} { c } { 0 } \\ { 1 } \\ { 0 } \\ { \vdots } \\ { 0 } \end{array} \right] , \ldots \text { or } \left[ \begin{array} { c } { 0 } \\ { 0 } \\ { 0 } \\ { \vdots } \\ { 1 } \end{array} \right]
$$

```matlab
% Add ones to the X data matrix
X = [ones(m, 1) X];

for t = 1:m
    x_t = X(t,:);
    y_t = (y(t,:) == 1:num_labels);
    a2 = [1 sigmoid(x_t*Theta1');];
    a3 = sigmoid(a2 * Theta2');
    h = a3;
    J = J + (-y_t*log(h') - (1-y_t)*log(1-h'));
end

J = J /m;
```

在这里我们的代码只能处理三层的神经网络，按照上次编程练习的思路，逐层对参数值进行求解，并在每层加入偏置项1.

### 1.2 正则化代价方程

因为在练习中，我们的神经网络只有两层。因此，我们可以使用下面的公式来计算正则化之后的代价方程
$$
\begin{aligned} J ( \boldsymbol { \theta } ) & = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sum _ { k = 1 } ^ { K } \left[ - y _ { k } ^ { ( i ) } \log \left( \left( h _ { \theta } \left( x ^ { ( i ) } \right) _ { k } \right) - \left( 1 - y _ { k } ^ { ( i ) } \right) \log \left( 1 - \left( h _ { \theta } \left( x ^ { ( i ) } \right) \right) _ { k } \right) \right] \right. \\ & + \frac { \lambda } { 2 m } \left[ \sum _ { j = 1 } ^ { 25 } \sum _ { k = 1 } ^ { 400 } \left( \Theta _ { j , k } ^ { ( 1 ) } \right) ^ { 2 } + \sum _ { j = 1 } ^ { 10 } \sum _ { k = 1 } ^ { 25 } \left( \Theta _ { j , k } ^ { ( 2 ) } \right) ^ { 2 } \right] \end{aligned}
$$
对应的 matlab 代码为：

```matlab
% Add ones to the X data matrix
X = [ones(m, 1) X];

for t = 1:m
    x_t = X(t,:);
    y_t = (y(t,:) == 1:num_labels);
    a2 = [1 sigmoid(x_t*Theta1');];
    a3 = sigmoid(a2 * Theta2');
    h = a3;
    J = J + (-y_t*log(h') - (1-y_t)*log(1-h'));
end

re = (sum(sum(Theta1(:,2:end).^ 2)) + sum(sum(Theta2(:,2:end).^ 2))) * lambda / (2*m);

J = J /m + re;
```

需要注意的是，计算正则项时，不应该包括偏置项的参数，因此需要使用索引来对theta矩阵进行切分。因为 theta 矩阵是一个二维矩阵，需要使用两次 sum 函数来进行求和。

![mark](http://media.sumblog.cn/blog/20190201/vmCDiT1Byod9.png?imageslim)

计算文档中给出的样例，可以观察到代价输出符合预期，可以进行提交。

## 2. 反向传递

### 2.1 Sigmoid 梯度

首先，需要实现 sigmoid 函数的导数的计算。

SIgmoid 函数为：
$$
\operatorname { sigmoid } ( z ) = g ( z ) = \frac { 1 } { 1 + e ^ { - z } }
$$
其导数为：
$$
g ^ { \prime } ( z ) = \frac { d } { d z } g ( z ) = g ( z ) ( 1 - g ( z ) )
$$
完成 sigmoidGradient.m 函数

```matlab
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).


g = sigmoid(z).*(1-sigmoid(z));
```

### 2.2 随机初始化

为了避免全部初始值相同，造成的反向传递的对称效应。需要将参数初始化为较小的随机值。

文档中已经给出了构建随机初始化矩阵的代码。

```matlab
 % Randomly initialize the weights to small values
    epsilon_init = 0.12;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
```

### 2.3 反向传播

使用反向传播来计算偏导数的步骤：

- $\Delta^{(l)}_{i,j} := 0$ 令初始所有的反向误差值为 0

循环遍历所有给定的样本 （ t 的值从 1 到 m）

1. 令 $a^{(1)} = x^{(t)}$

2. 使用前向传播，计算每一层的参数值 $a^{l}$

3. 使用 $y^{(t)}$ 反向计算最后一层的误差值 $\delta^{(L)} = a^{(L)}-y^{(t)}$ 

   其中L是我们的总层数，而$a^{(L)}$是最后一层的激活单元的输出矢量。所以我们最后一层的“误差值”只是我们在最后一层的实际结果与y中正确输出的差异。要获得最后一层之前的图层的delta值，我们可以使用一个从右到左的步骤：

4. 使用公式：
   $$
   \delta ^ { ( l ) } = \left( \left( \Theta ^ { ( l ) } \right) ^ { T } \delta ^ { ( l + 1 ) } \right) .* a ^ { ( l ) } .* \left( 1 - a ^ { ( l ) } \right)
   $$
   计算 $\delta ^ { ( L - 1 ) } , \delta ^ { ( L - 2 ) } , \ldots , \delta ^ { ( 2) }$

5. $$
   \Delta _ { i , j } ^ { ( l ) } : = \Delta _ { i , j } ^ { ( l ) } + a _ { j } ^ { ( l ) } \delta _ { i } ^ { ( l + 1 ) }
   $$

   向量化的表示：
   $$
   \Delta ^ { ( l ) } : = \Delta ^ { ( l ) } + \delta ^ { ( l + 1 ) } \left( a ^ { ( l ) } \right) ^ { T }
   $$
   
6. 计算偏导数：
   $$
   \begin{array} { l } { D _ { i , j } ^ { ( l ) } : = \frac { 1 } { m } \left( \Delta _ { i , j } ^ { ( l ) } + \lambda \Theta _ { i , j } ^ { ( l ) } \right) , \text { if } j \neq 0 } \\ { D _ { i , j } ^ { ( l ) } : = \frac { 1 } { m } \Delta _ { i , j } ^ { ( l ) }, i f = 0 } \end{array}
   $$

$$
\frac { \partial } { \partial \Theta _ { i j } ^ { ( l ) } } J ( \Theta ) = D _ { i j } ^ { ( l ) }
$$

**在 nnCostFunction 函数中补充完成反向传播的代码：**

```matlab
% Add ones to the X data matrix
X = [ones(m, 1) X];

for t = 1:m
    a1 = X(t,:);
    y_t = (y(t,:) == 1:num_labels);
    z2 = a1*Theta1';
    a2 = [1 sigmoid(a1*Theta1');];
    a3 = sigmoid(a2 * Theta2');
    h = a3;
    J = J + (-y_t*log(h') - (1-y_t)*log(1-h'));
    
    delta3 = a3 - y_t;
    delta2 =delta3 * Theta2 .* sigmoidGradient([1 z2]);
    Theta2_grad = Theta2_grad + delta3'* a2;
    Theta1_grad = Theta1_grad + delta2(2:end)'* a1;
end

re = (sum(sum(Theta1(:,2:end).^ 2)) + sum(sum(Theta2(:,2:end).^ 2))) * lambda / (2*m);

J = J /m + re;
    
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
```

### 2.4 梯度检查

为了检查我们的反向传播算法是否正确，我们运行文档中给出的梯度检查算法，梯度检查会通过近似法计算出给定点的梯度，再和我们使用反向传播计算出的梯度进行对比：

![](http://media.sumblog.cn/img/20190202181337.png-min_pic)

可以看到，最终的误差之和在 $10^{-11}$ 数量级。这个误差可以接受。反向传播能够正确的算出函数梯度。

之后提交代码进行检查：

![](http://media.sumblog.cn/img/20190202181714.png-min_pic)

可以看到通过了上述监测点的检测

### 2.5 梯度正则化

在使用反向传播算法完成了梯度的计算后，我们需要将得到的梯度加入正则化项：
$$
\begin{aligned} \frac { \partial } { \partial \Theta _ { i j } ^ { ( l ) } } J ( \Theta ) = D _ { i j } ^ { ( l ) } & = \frac { 1 } { m } \Delta _ { i j } ^ { ( l ) } \text { for } j = 0 \\ \frac { \partial } { \partial \Theta _ { i j } ^ { ( l ) } } J ( \Theta ) = D _ { i j } ^ { ( l ) } & = \frac { 1 } { m } \Delta _ { i j } ^ { ( l ) } + \frac { \lambda } { m } \Theta _ { i j } ^ { ( l ) } \text { for } j \geq 1 \end{aligned}
$$
和之前的正则化类似，偏置项的梯度不需要正则化。

观察所加入的正则化项，可以发现，我们只需要在 nnCostFunction 函数的末尾加入下面两行：

```matlab
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end) * lambda / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end) * lambda / m;
```

![](http://media.sumblog.cn/img/20190202182721.png-min_pic)

加入正则化项之后再进行梯度检查，最后的误差符合要求。

![](http://media.sumblog.cn/img/20190202182837.png-min_pic)

至此，我们便通过了所有的编程任务。

### 2.6 进行参数学习

使用 MATLAB 的 `fmincg` 函数，优化代价函数，进行 50 次迭代，完成参数学习。

```
options = optimset('MaxIter', 50);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
```

完成参数学习后，神经网络对训练集的输出结果正确率达到了 95.06 %

![](http://media.sumblog.cn/img/20190202183449.png-min_pic)

## 3. 参数矩阵可视化

![](http://media.sumblog.cn/img/20190202184320.png-min_pic)

![](http://media.sumblog.cn/img/20190202184531.png-min_pic)

下图是选用了一个较大的正则化系数后的结果。可以看到，可视化之后的参数图像更为平滑，忽视了更多的图像细节，转而提取更一般的特征。



## 附加：使用 nprtool 创建模式识别网络

MATLAB 的神经网络工具箱中，有 GUI 工具用来快速生成和训练神经网络

- 聚类：nctool
- 回归：nftool
- 时间序列：ntstool
- 模式识别：nprtool

在命令行中运行 `nnstart` 来启动常规神经网络程序

![](http://media.sumblog.cn/img/20190202185332.png-min_pic)

启动 nprtool 进行模式识别分类

![](http://media.sumblog.cn/img/20190202185408.png-min_pic)

选择合适的数据和类型进行导入：

![](http://media.sumblog.cn/img/20190202185603.png-min_pic)

按照文档的要求，选择百分之 5 的验证集和测试集

![](http://media.sumblog.cn/img/20190202185655.png-min_pic)

隐藏层的神经节点数设置为 25：

![](http://media.sumblog.cn/img/20190202185802.png-min_pic)

点击 Train 进行模型的训练:

![](http://media.sumblog.cn/img/20190202185918.png-min_pic)

![](http://media.sumblog.cn/img/20190202190005.png-min_pic)

可以绘制 ROC 曲线进行观察：

![](http://media.sumblog.cn/img/20190202190136.png-min_pic)

**导出神经网络**

选择将网络保存到 MATLAB 工作区：

![](http://media.sumblog.cn/img/20190202190353.png-min_pic)

**使用训练的神经网络进行预测：**

![](http://media.sumblog.cn/img/20190202190539.png-min_pic)

