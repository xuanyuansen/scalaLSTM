###深入理解LSTM的BPTT算法
####LSTM网络结构
关于LSTM网络的结构可以阅读这篇文章：http://colah.github.io/posts/2015-08-Understanding-LSTMs/

这里需要注意文章最后提及的LSTM两种变形，第一种是加入peephole，使得gate layer能够回溯前一个cell的状态，这增加了一些复杂度；第二种是GRU，将gate layer和forget layer合并为一个update layer，降低了复杂度。
####LSTM网络的训练
LSTM的训练使用了BPTT算法，需要重要理解的一点是BPTT算法相当于BP算法扩展到序列（时序）数据，另一个需要理解的点是LSTM是recurrent neural network（这里注意理解recurrent neural network和recursive neural network的区别），BPTT算法在计算中要注意这一点。

####LSTM的计算图Compute Graph
*	LSTM的BPTT算法可以参考这篇文章http://nicodjimenez.github.io/2014/08/08/lstm.html
*	讲述很清晰，注意这篇文章里面最后的输出h(t)没有加入tanh变换。

*	为了理解LSTM的recursive特性，可以参考下图。
*	从LSTM的结构可以看到，当前cell的状态会受到前一个cell状态的影响，这体现了LSTM的recursive特性。同时在误差反向传播计算时，可以发现h(t)的误差不仅仅包含当前时刻T的误差，也包括T时刻后所有时刻的误差，即back propagation through time的含义。这样每个时刻变量的误差都可以经由h(t)和c(t+1)迭代计算。
*	![](https://github.com/xuanyuansen/scalaLSTM/blob/master/image/LSTM%20understanding.png)


*	为了使整个直观计算过程，在参考神经网络计算图分解的基础上，LSTM的计算图如下图所示，从计算图上面可以直观地看出LSTM的forward propagation和back propagation过程。
*	从图中可以看出，H(t-1)的误差由H(t)决定，且要对所有的gate layer求和，c(t-1)由c(t)决定，而c(t)的误差由两部分，一部分是h(t)，领一部分是c(t+1)。
*	![](https://github.com/xuanyuansen/scalaLSTM/blob/master/image/LSTM%20Computation%20Graph.png)
*	如果所示，在计算的时候，需要传入h(t)和c(t+1)，h(t)在更新的时候需要加上h(t+1)。

####SCALA实现
* breeze库

####利用SPARK实现minibatch方式的训练

####几种常见的LSTM结构
*	1、原始LSTM
*	2、peephole LSTM
*	3、GRU
