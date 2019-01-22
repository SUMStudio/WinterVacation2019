# 多特征量

多元线性回归的假设方程：
$$
h _ { \theta } ( x ) = \theta _ { 0 } + \theta _ { 1 } x _ { 1 } + \theta _ { 2 } x _ { 2 } + \cdots + \theta _ { n } x _ { n }
$$
假设方程也可以表示为矩阵的乘积形式：
$$
h _ { \theta } ( x ) = \left[ \begin{array} { c c c c } { \theta _ { 0 } } & { \theta _ { 1 } } & { \dots } & { \theta _ { n } } \end{array} \right] \left[ \begin{array} { c } { x _ { 0 } } \\ { x _ { 1 } } \\ { \vdots } \\ { x _ { n } } \end{array} \right] = \theta ^ { T } x
$$

# 梯度下降法解决多元线性回归问题

多元线性回归的代价函数：
$$
J ( \theta ) = \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left( \theta ^ { T } x ^ { ( i ) } - y ^ { ( i ) } \right) ^ { 2 }
$$
梯度下降算法优化参数：
$$
\begin{array} { l } { \text { repeat until convergence: } \{ } \\ { \theta _ { j } : = \theta _ { j } - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) \cdot x _ { j } ^ { ( i ) } \quad \text { for } j : = 0 \ldots n } \end{array}
$$

# 梯度下降的技巧

## 特征放缩

如果特征的取值范围极度不同，那么其代价函数图像将会呈现非常偏斜的形式。

将特征的定义域进行放缩，梯度下降算法将会更快收敛

我们通常将特征的取值范围约束到 $-1<x<1$ 的范围

## 均一化

将 xi 替换为 xi 减去均值，这样均值将会变为 0
$$
x _ { i } : = \frac { x _ { i } - \mu _ { i } } { s _ { i } }
$$
这里 $\mu_i$ 表示该特征量的均值，$s_i$ 代表该特征量的取值范围 （最大值减去最小值）

# 非线性回归

当我们的回归方程需要是非线性拟合时，我们可以创建基于已有特征 $x$ 的高次特征 $x^p$

当产生了高次特征之后，特征的归一化就十分重要



# 正规方程

一步就可以得到最优值
$$
\theta = \left( X ^ { T } X \right) ^ { - 1 } X ^ { T } y
$$
