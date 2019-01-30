# 过拟合 Overfiting

![mark](http://media.sumblog.cn/blog/20190126/nCHA4WPKB50v.png?imageslim)

解决过拟合的方法：

1. 减少特征的数量

   - 人工判断哪些变量更重要
   - 使用模型选择算法

2. 正则化

   

## 正则化

修改成本函数，添加额外的正则化项
$$
\min _ { \theta } \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) ^ { 2 } + \lambda \sum _ { j = 1 } ^ { n } \theta _ { j } ^ { 2 }
$$
对于常规的线性回归，正则化之后的梯度下降算法如下：
$$
\begin{array} { l } { \text { Repeat } \{ } \\ { \theta _ { 0 } : = \theta _ { 0 } - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { 0 } ^ { ( i ) } } \\ { \quad \theta _ { j } : = \theta _ { j } - \alpha \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { j } ^ { ( i ) } \right) + \frac { \lambda } { m } \theta _ { j } \right] } & { j \in \{ 1,2 \ldots n \} } \end{array}
$$
也可以表达为：
$$
\theta _ { j } : = \theta _ { j } \left( 1 - \alpha \frac { \lambda } { m } \right) - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { j } ^ { ( i ) }
$$
其中，前面的$1-\alpha \frac{\lambda}{m}$ 保证了其值总是小于1，这样，每次梯度下降，都会将 $\theta$ 值向 0 的方向压缩

**最小二乘法**

正则化之后的最小二乘方程为：
$$
\theta = \left( X ^ { T } X + \lambda \cdot L \right) ^ { - 1 } X ^ { T } y
$$
其中：
$$
L = \left[ \begin{array} { c c c c } { 0 } & { } & { } & { } \\ { } & { 1 } & { } & { } \\ { } & { } & { 1 } & { } \\ { } & { } & { } & { \ddots } & { } \\ { } & { } & { } & { } & { 1 } \end{array} \right]
$$

加入正则化项之后，还能够解决 $X^TX$ 不可逆的问题

## 逻辑回归正则化

增加正则化项之后的代价函数：
$$
J ( \theta ) = - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left[ y ^ { ( i ) } \log \left( h _ { \theta } \left( x ^ { ( i ) } \right) \right) + \left( 1 - y ^ { ( i ) } \right) \log \left( 1 - h _ { \theta } \left( x ^ { ( i ) } \right) \right) \right] + \frac { \lambda } { 2 m } \sum _ { j = 1 } ^ { n } \theta _ { j } ^ { 2 }
$$
