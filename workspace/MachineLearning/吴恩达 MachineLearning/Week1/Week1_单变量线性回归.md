# 线性回归

回归问题是一种监督学习

回归是指根据之前的数据预测出现在的值

课程中使用的符号：

- m ：训练样本的数目
- x：输入变量，特征量
- y：数据结果
- (x,y)：一个训练样本
- h：hypothesis 假设，h表示一个函数，从 x 到 y 的映射

**单变量线性回归**

![mark](http://media.sumblog.cn/blog/20190120/0HWYcB5UW1QD.png?imageslim)

当我们试图预测的目标量是连续值时，我们将问题成为回归问题，当输出的目标量只是少量的离散值时，我们将其称为分类问题

## 代价函数

我们的假设函数（用来预测的函数）是这样的线性函数的形式：
$$
h _ { \theta } ( x ) = \theta _ { 0 } + \theta _ { 1 } x
$$
我们把 $\theta​$ 称为模型参数 

如何得出模型参数的值？我们要选择参数值，使得尽可能多的训练样本在预测函数输出结果附近

这也叫做 **最小化问题** h(x) 和 y 的差距 （平方误差和）尽量最小

**代价函数：**
$$
J \left( \theta _ { 0 } , \theta _ { 1 } \right) = \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left( \hat { y } _ { i } - y _ { i } \right) ^ { 2 } = \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x _ { i } \right) - y _ { i } \right) ^ { 2 }
$$
这种代价函数有时又称为 **平方误差代价函数**，对于大多数线性回归问题非常合理

> 平均值减半，是为了便于计算梯度下降，因为平方的导数项将会抵消掉前面的$\frac{1}{2}$ 系数

## 代价函数的直观含义

$$
\underset { \theta _ { 0 } , \theta _ { 1 } } { \operatorname { minimize } } J \left( \theta _ { 0 } , \theta _ { 1 } \right)
$$

![mark](http://media.sumblog.cn/blog/20190120/5PaaxjD3cAtB.png?imageslim)

右侧的图是轮廓图，每一个圈上的点代表了 J 值相同的点的集合（等高线）

我们需要的是编写程序，自动的最小化代价函数



# 梯度下降 Gradient descent

对于给定的方程：
$$
J(\theta_0,\theta_1)
$$
最小化 J 的值
$$
\underset { \theta _ { 0 } , \theta _ { 1 } } { \operatorname { minimize } } J \left( \theta _ { 0 } , \theta _ { 1 } \right)
$$

- 开始给定特定的参数值 $\theta_0, \theta_1$ （通常将这些值初始化为 0）

- 让参数的值一点点不断变化，来减小 J 的值，直到 J 达到最小，或者是局部最小

  正确的更新是同时更新所有的参数
  $$
  \begin{array} { l } { \operatorname { temp } 0 : = \theta _ { 0 } - \alpha \frac { \partial } { \partial \theta _ { 0 } } J \left( \theta _ { 0 } , \theta _ { 1 } \right) } \\ { \operatorname { temp } 1 : = \theta _ { 1 } - \alpha \frac { \partial } { \partial \theta _ { 1 } } J \left( \theta _ { 0 } , \theta _ { 1 } \right) } \\ { \theta _ { 0 } : = \operatorname { temp } 0 } \\ { \theta _ { 1 } : = \operatorname { temp } 1 } \end{array}
  $$
  

![mark](http://media.sumblog.cn/blog/20190120/g5onI5R47b1B.png?imageslim)

梯度下降的方向是函数导数的防线，参数值变化的大小由 $\alpha$ （学习率）来确定， 来控制以多大的幅度来更新参数

重复同时改变所有参数的值，直到函数值达到最小：
$$
\theta _ { j } : = \theta _ { j } - \alpha \frac { \partial } { \partial \theta _ { j } } J \left( \theta _ { 0 } , \theta _ { 1 } \right)
$$
当参数的初始值位于局部最优点时，梯度下降算法将无法工作，所有参数的值不再改变

当我们快要到达最优点时，梯度下降算法会自动采取较小的幅度





