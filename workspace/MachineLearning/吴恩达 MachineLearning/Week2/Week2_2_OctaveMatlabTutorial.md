# Octave 移动数据

pwd 命令显示 Octave 当前所处的路径



# Octave 可视化

1. 函数绘制

   - 设置横坐标扫描精度

     ```matlab
     t = [0:0.01:0.98]
     ```

   - 生成对应纵坐标

     ```matlab
     y1 = sin(2*pi*4*t)
     ```

   - plot 函数进行绘图

     ```
     plot(t,y1)
     ```

   - 更多绘图命令

     - 叠加绘图

       ```matlab
       hold on;
       plot (t,y2);
       ```

     - 横纵坐标标签

       ```matlab
       xlabel('time')
       ylabel('value')
       ```

     - 图例

       ```matlab
       legend('sin','cos') 
       title('my plot') %标题
       ```

     - 多图绘制

       ```matlab
       figure(1); plot(t,y1);
       ```

       ```
       subplot(1,2,1) %将图像分为两个格子，并使用第一个格子
       ```

     - 坐标轴

       ```
       axis([0.5 1 -1 1]) %调整坐标轴的范围
       ```

2. 矩阵可视化

   - 展示矩阵彩图

     ```matlab
     imagesc(A)
     colorbar
     colormap gray
     ```



# Octave 流程控制

- For 循环

  ```
  for i=1:10,
  	v(i) = 2^i;
  end;
  ```

- while 循环

  ```
  while i <= 5,
  	v(i) = 100;
  	i+=1;
  end;
  ```

- if

  ```
  if v ==1,
  	disp ('the value is one');
  else
  	disp ('not');
  end;
  ```

**定义函数**

创建一个文件，并使用 函数名.m 命名

```
function A = functionname();
%%%%%%%%%%%%%%%%
% set a to the return value
%%%%%%%%%%%%%%%%%%%%
end
```



# 向量化

两个向量逐项求积并求和，等于向量的转置再相乘
$$
\begin{aligned} h _ { \theta } ( x ) & = \sum _ { j = 0 } ^ { n } \theta _ { j } x _ { j } \\ & = \theta ^ { T } x \end{aligned}
$$

$$
\begin{array} { l } { \theta _ { 0 } : = \theta _ { 0 } - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { 0 } ^ { ( i ) } } \\ { \dot { \theta } _ { 1 } : = \theta _ { 1 } - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { 1 } ^ { ( i ) } } \\ { \theta _ { 2 } : = \theta _ { 2 } - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { 2 } ^ { ( i ) } } \\ { ( n = 2 ) } \end{array}
$$

