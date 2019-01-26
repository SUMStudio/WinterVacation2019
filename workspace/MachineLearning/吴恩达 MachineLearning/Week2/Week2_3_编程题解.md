# 机器学习 编程练习1 ： 线性回归

这是 Coursera 上吴恩达的 Machine Learning 的第一次编程作业。通过两周课程的讲解，我们基本了解了机器学习中第一个最简单的算法 ： **线性回归 Linear Regression** 的基本原理和解决办法



## 0. 线性回归

所谓线性回归，是指用线性方程来对输入的特征值，以及输出的变量建立线性方程模型，进行拟合的过程。

线性回归的样本中，每一组给定的输入，都存在着唯一确定的输出，因此，线性回归是一种有监督学习。学习完成后，我们便得到了学习出的模型。在这里，得到的线性模型由一个 $n$ 维向量表示，因为额外引入了一个与特征无关的常数项，所以，$n​$ 的大小是特征的数量 + 1. 

![mark](http://media.sumblog.cn/blog/20190120/0HWYcB5UW1QD.png?imageslim)

在线性回归中，我们把输入的特征值，均看做是连续的数值，得到预测模型 h 之后，将任意给出的一组输出特征 x 带出方程 h，便可解出预测值。



在高中，我们便学习过使用最小二乘法来对线性方程进行拟合，看起来所谓机器学习中的线性回归也不过是求解最小二乘法的参数而已。

诚然，在特征个数有限的时候，我们的确可以使用最小二乘法对回归模型进行求解。
$$
\theta = \left( X ^ { T } X \right) ^ { - 1 } X ^ { T } \vec { y }
$$

但是，由于方程中出现了矩阵的逆，因此，当特征的个数过多（超过上万个）时，使用最小二乘法进行拟合会耗费过多的计算资源。因此，在机器学习中，着重讲授了 **梯度下降法**，对预测方程中的参数 $\theta$ 进行优化。

> Tom Mitchell在他的《[Machine Learning](http://www.amazon.com/dp/0070428077?tag=job0ae-20)（中文版：[计算机科学丛书:机器学习](http://www.amazon.cn/gp/product/B002WC7NH2/ref=as_li_qf_sp_asin_il_tl?ie=UTF8&camp=536&creative=3200&creativeASIN=B002WC7NH2&linkCode=as2&tag=vastwork-23) ）》一书的序言开场白中给出了一个定义：
>
> “机器学习这门学科所关注的问题是：计算机程序如何随着经验积累自动提高性能。”
>
> 他在引言中多次重复提到一个简短的形式体系：
>
> “对于某类任务T和性能度量P，如果一个计算机程序在T上以P衡量的性能随着经验E而自我完善，那么我们称这个计算机程序在从经验E学习。”

而梯度下降法便是提现了机器学习定义的一种算法。

梯度下降法在求解过程中引入了一个关键的度量方法：**代价函数**

给定一组参数 $\theta$ ，可以通过代价函数求解出其代价值，而这里的  「代价」，刻画了训练集对当前模型的满意程度，代价值越低，当前模型就越令人满意，因此，我们可以不断的调整参数的值，让其代价函数最小，这样就得到了最终的预测模型参数。

如何调整参数的值，使得代价函数值尽快下降？这便是梯度下降方法：代价函数下降最快的方向，就是其梯度方向。因此，我们给代价函数求其关于每个变量的偏导数，偏导数乘以步长，确定下一个参数的值。再计算新的参数值的偏导数，进行下一轮更新，直到代价函数的偏导数为0，代价函数值不再下落，这样就得到了全局最优，或是局部最优解。



## 1. 编程环境

吴恩达在机器学习课程中，多次强烈推荐使用 Octave\ MATLAB 作为编程练习的环境。的确，在完成这一次的编程练习之后，的确可以感受到其便利性。

在进行编程练习之前，先下载 Coursera 提供的作业包，作业包中包括了整个课程的所有编程题目，一次下载之后便不用再下载。

为了完成编程练习，我安装了 MATLAB R2018b 编程环境。打开 MATLAB 之后，首先在MATLAB中将 ex1 文件夹设置为工作区。这样，便可以运行实验1 中预先编辑好的各种函数。

与我们学校的上机练习相比，吴恩达机器学习的编程练习显得格外用心。在本次练习的文件夹中，有预先编辑好的实验导引 `ex1.mlx`

与一般的 md 或 pdf 说明文档不同，mlx 文档是 MATLAB 的专用格式，让我感到最方便的一点是，文档里面编写的代码可以直接运行，并在文档中实时显示结果。

![mark](http://media.sumblog.cn/blog/20190124/DkBQV2dtQfxn.png?imageslim)

因此，要获得最佳的练习体验，请使用 MATLAB，并完整的按照 `ex1.mlx` 中的说明逐步实现所有编程任务。

### 1.1 warmUp 热身

课程为了让我们尽快进入编程状态，特意的编写了 `warmUpExercise()` 函数，类似于学习编程语言时的 helloword， `warmUpExercise()` 函数只是返回了一个 5 阶单位矩阵。

打开并编辑 `warmUpExercise.m`  文件，可以发现，课程已经为我们编写好了函数的框架，我们只需要编写关键代码对函数进行补完，免去语法细节和其他不必要问题对编程练习的干扰。

在函数的注释下面编写我们的实现：

```matlab
function A = warmUpExercise()
%WARMUPEXERCISE Example function in octave
%   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

A = [];
% ============= YOUR CODE HERE ==============
% Instructions: Return the 5x5 identity matrix 
%               In octave, we return values by defining which variables
%               represent the return values (at the top of the file)
%               and then set them accordingly. 

A = eye(5);

% ===========================================

end
```

运行文档中的代码观察输出：

![mark](http://media.sumblog.cn/blog/20190124/NWIMwYQGrPw5.png?imageslim)

## 2.  Linear regression with one variable 单变量线性回归

### 2.1 Computing the cost 编写代价函数

编写代价函数的计算程序，按照文档中给出的公式
$$
J ( \theta ) = \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) ^ { 2 }
$$
其中：
$$
h _ { \theta } ( x ) = \theta ^ { T } x = \theta _ { 0 } + \theta _ { 1 } x _ { 1 }
$$
在这里，我们使用向量化的方法来简化求和过程

```
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


h = X * theta;
J = sum((h - y) .^ 2) / (2 * m); 


% =========================================================================

end
```

编写过程中需要注意的是，输入变量的矩阵形状，必须首先明确给定的参数究竟是行向量还是列向量，我们可以在代码编辑器窗口点击行号右侧插入断点，在运行过程中进行观测和调试。

按照文档中的流程对编写的函数进行测试，如果测试无误，在命令行窗口中 运行 `submit()` 函数。将自己的代码进行提交测试：

![mark](http://media.sumblog.cn/blog/20190124/lUufXKihta1c.png?imageslim)



## 2.2  gradientDescent 梯度下降算法

梯度下降的核心是对参数值的更新，对代价函数进行求导，可以得到更新方程：
$$
\theta _ { j } : = \theta _ { j } - \alpha \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( h _ { \theta } \left( x ^ { ( i ) } \right) - y ^ { ( i ) } \right) x _ { j } ^ { ( i ) } \quad \text { (simultaneously update } \theta _ { j } \text { for all } j )
$$

```matlab
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X * theta;
    theta = theta - ((alpha * (h-y)'*X)/m)';


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

```

绘制出最后求得的模型，可以观察到对给定的数据集有较好的拟合效果

![mark](http://media.sumblog.cn/blog/20190124/VYJtfB1VlbVK.png?imageslim)

![mark](http://media.sumblog.cn/blog/20190124/5XqHhivtNeGJ.png?imageslim)

### 2.3 可视化

为了加深对梯度下降的理解，课程文档中还给出了对代价函数可视化的要求。运行实验代码，可以观察到代价函数 J 的三维图形，并绘制出等高线，观察梯度下降得到了全局最优解。

![mark](http://media.sumblog.cn/blog/20190124/6aWz25gE1Fr3.png?imageslim)

## 3. 拓展：多变量线性回归

以下的内容是课程的拓展部分，不计入最后的总分。

### 3.1 Feature Normalization 特征归一化

在多变量问题中，因为每一个输入特征的取值范围差异过大，直接使用梯度下降算法会导致代价函数的等高线过于「狭长」，梯度下降在等高线之间波动，造成算法的性能下降，因此，在数据处理之前，必须要进行输入特征的归一化处理 （均值为0，方差为1）

```matlab
function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ sigma;


% ============================================================

end

```

### 3.2 代价函数和梯度下降

因为之前编写的梯度下降算法，使用了向量化的编程思路，对给定数据集的维度无关，可以在这里直接使用，不需要做任何改动。

### 3.3 输入测试数据进行预测

在这里，文档中让我们实现对 1650 sq-ft ，3 个房间的住宅的房间进行预测。sigma 是我们之前使用梯度下降算法求得的优化参数。补全 price 的计算方法：

```matlab
% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

price = []; % Enter your price formula here
price = [1, ([1650, 3] - mu)./ sigma] * theta;

% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);
```

![mark](http://media.sumblog.cn/blog/20190124/KT9m8gIroxRR.png?imageslim)

### 3.4 可视化调节参数

文档中让我印象最深刻的地方就是这里，可以直接在代码中嵌入下拉按钮和滑块，进行参数的调整，调整之后，会在下方动态展示出代价函数的下降过程，便于观察参数值是否合理

![mark](http://media.sumblog.cn/blog/20190124/nX6Uq60fsMmn.png?imageslim)



### 3.5 Normal Equations 使用最小二乘法拟合

最小二乘法求解线性回归问题参数的方程为
$$
\theta = \left( X ^ { T } X \right) ^ { - 1 } X ^ { T } \vec { y }
$$
使用最小二乘法求解不需要归一化，也不需要多次迭代的过程

```matlab
function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------


theta = pinv((X'* X)) * X'* y;

% -------------------------------------------------------------


% ============================================================

end

```

同样的进行预测时也不需要考虑归一化的影响，可以直接向量乘法进行运算，求得最终预测结果

![mark](http://media.sumblog.cn/blog/20190124/m3VuvgSMlzSQ.png?imageslim)

可以观察到使用最小二乘法求得的最优解和之前使用梯度下降方法求解结果基本吻合。证明了梯度下降方法的正确和合理性。



## 3. 提交结果

![mark](http://media.sumblog.cn/blog/20190124/9cguSlr6oQDl.png?imageslim)



![mark](http://media.sumblog.cn/blog/20190124/e7daKlhvQAJJ.png?imageslim)