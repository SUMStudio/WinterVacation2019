## 代价方程

**符号表示：**

1. L ：神经网络的总层数
2. sl：第一层（输入层）的单元数
3. K：输出层的分类数

代价方程 J：
$$
J ( \Theta ) = - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sum _ { k = 1 } ^ { K } \left[ y _ { k } ^ { ( i ) } \log \left( \left( h _ { \Theta } \left( x ^ { ( i ) } \right) \right) _ { k } \right) + \left( 1 - y _ { k } ^ { ( i ) } \right) \log \left( 1 - \left( h _ { \Theta } \left( x ^ { ( i ) } \right) \right) _ { k } \right) \right] + \frac { \lambda } { 2 m } \sum _ { l = 1 } ^ { L - 1 } \sum _ { i = 1 } ^ { s _ { l } } \sum _ { j = 1 } ^ { s _ { l 1 } } \left( \Theta _ { j , i } ^ { ( l ) } \right) ^ { 2 }
$$
当前theta矩阵中的列数等于当前层中的节点数（包括偏置单元）。当前theta矩阵中的行数等于下一层中的节点数（不包括偏置单元）。与逻辑回归一样，我们对每个项都进行平方。

## 反向传播

- $\Delta^{(l)}_{i,j} := 0​$ 令初始所有的反向误差值为 0

循环遍历所有给定的样本 （ t 的值从 1 到 m）

1. 令 $a^{(1)} = x^{(t)}​$

2. 使用前向传播，计算每一层的参数值 $a^{l}​$

3. 使用 $y^{(t)}$ 反向计算最后一层的误差值 $\delta^{(L)} = a^{(L)}-y^{(t)}​$ 

   其中L是我们的总层数，而$a^{(L)}​$是最后一层的激活单元的输出矢量。所以我们最后一层的“误差值”只是我们在最后一层的实际结果与y中正确输出的差异。要获得最后一层之前的图层的delta值，我们可以使用一个从右到左的步骤：

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

## 梯度检查

对于多分类问题，可以使用下式来近似计算偏导数
$$
\frac { \partial } { \partial \Theta _ { j } } J ( \Theta ) \approx \frac { J \left( \Theta _ { 1 } , \ldots , \Theta _ { j } + \epsilon , \ldots , \Theta _ { n } \right) - J \left( \Theta _ { 1 } , \ldots , \Theta _ { j } - \epsilon , \ldots , \Theta _ { n } \right) } { 2 \epsilon }
$$
Matlab 语言如下：

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```



## 随机初始化

**对称现象**

进行随机初始化的目的就是打破对称效应

```
Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

这将随机初始化所有的 $\Theta _ { i j } ^ { ( l ) }$ 为介于 $[-\epsilon,\epsilon]$ 的随机值



## 总结

**如何训练神经网络来完成分类问题**

1. 随机初始化权值
2. 使用前向传播来获取预测输出
3. 计算代价函数
4. 使用反向传播来计算偏导数
5. 使用梯度检查来确保正确性
6. 使用梯度下降算法来最小化参数值

