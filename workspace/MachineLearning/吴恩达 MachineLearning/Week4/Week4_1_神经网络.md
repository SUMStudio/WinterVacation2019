# Neural Networks 神经网络

## 模型表示

在神经网络中，使用多层激活函数来模拟神经元。输入节点称为输入层，最终输出的假设函数称为输出层，可以在输入和输出层中间建立隐藏层的中间节点层。

我们把每一层中的节点，叫做激活单元
$$
\begin{array} { l } { a _ { i } ^ { ( j ) } = ^ { \prime \prime } \text { activation } ^ { \prime \prime } \text { of unit } i \text { in layer } j } \\ { \Theta ^ { ( j ) } = \text { matrix of weights controlling function mapping from layer } j \text { to layer } j + 1 } \end{array}
$$

对于有一个隐藏层的神经网络
$$
\left[ \begin{array} { l } { x _ { 0 } } \\ { x _ { 1 } } \\ { x _ { 2 } } \\ { x _ { 3 } } \end{array} \right] \rightarrow \left[ \begin{array} { c } { a _ { 1 } ^ { ( 2 ) } } \\ { a _ { 2 } ^ { ( 2 ) } } \\ { a _ { 3 } ^ { ( 2 ) } } \end{array} \right] \rightarrow h _ { \theta } ( x )
$$
每一个激活单元的假设函数为：
$$
\begin{array} { r } { a _ { 1 } ^ { ( 2 ) } = g \left( \Theta _ { 10 } ^ { ( 1 ) } x _ { 0 } + \Theta _ { 11 } ^ { ( 1 ) } x _ { 1 } + \Theta _ { 12 } ^ { ( 1 ) } x _ { 2 } + \Theta _ { 13 } ^ { ( 1 ) } x _ { 3 } \right) } \\ { a _ { 2 } ^ { ( 2 ) } = g \left( \Theta _ { 20 } ^ { ( 1 ) } x _ { 0 } + \Theta _ { 21 } ^ { ( 1 ) } x _ { 1 } + \Theta _ { 22 } ^ { ( 1 ) } x _ { 2 } + \Theta _ { 23 } ^ { ( 1 ) } x _ { 3 } \right) } \\ { a _ { 3 } ^ { ( 2 ) } = g \left( \Theta _ { 30 } ^ { ( 1 ) } x _ { 0 } + \Theta _ { 31 } ^ { ( 1 ) } x _ { 1 } + \Theta _ { 32 } ^ { ( 1 ) } x _ { 2 } + \Theta _ { 33 } ^ { ( 1 ) } x _ { 3 } \right) } \\ { h _ { \Theta } ( x ) = a _ { 1 } ^ { ( 3 ) } = g \left( \Theta _ { 10 } ^ { ( 2 ) } a _ { 0 } ^ { ( 2 ) } + \Theta _ { 11 } ^ { ( 2 ) } a _ { 1 } ^ { ( 2 ) } + \Theta _ { 12 } ^ { ( 2 ) } a _ { 2 } ^ { ( 2 ) } + \Theta _ { 13 } ^ { ( 2 ) } a _ { 3 } ^ { ( 2 ) } \right) } \end{array}
$$

## 前向传播

从输入层的激励开始前项传播给隐藏层再计算输出层的激励，

**向量化：**
$$
a ^ { ( j ) } = g \left( z ^ { ( j ) } \right)
$$

$$
z ^ { ( j + 1 ) } = \Theta ^ { ( j ) } a ^ { ( j ) }
$$

## 多分类问题

