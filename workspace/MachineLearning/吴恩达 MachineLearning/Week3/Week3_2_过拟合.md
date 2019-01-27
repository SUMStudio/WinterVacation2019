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
