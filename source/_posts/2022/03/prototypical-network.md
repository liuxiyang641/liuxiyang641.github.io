---
title: prototypical-network
notshow: false
date: 2022-03-10 16:40:44
categories:
- Paper
tags:
- theory
---

# Prototypical Networks for Few-shot Learning

作者为少次学习和零次学习提出了一种新的网络Prototypical network。核心思想是为不同的class定义不同的prototype的表示。这个prototype是有相同class下的所有实例求平均得到的。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20220310164356012.png" alt="image-20220310164356012" style="zoom:50%;" />

<!--more-->

直接看核心公式，

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20220310164441739.png" alt="image-20220310164441739" style="zoom:50%;" />

$S_k$就是class $k$下的所有实例。$f_{\phi}$是某种编码函数，或者叫embedding function，可以为任意合适的方法来产生最后的向量。例如作者就使用了CNN，作为在图像数据集下，few-shot的编码函数。

因此，如果要求某个新的实例$x$是否属于class $k$，通过定义距离函数$d(\cdot, \cdot)$，经过$softmax$就可求出：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20220310164703633.png" alt="image-20220310164703633" style="zoom:50%;" />

距离越大，当然成为class $k$的概率就越小。

作者在训练的时候，使用了之前工作采用的采样batch的方法，叫做*episodes*，核心思想是模拟少次学习在test时候的情况，每次train的时候，也只采样几个class，几个shot。具体作者的做法如下：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20220310170614007.png" alt="image-20220310170614007" style="zoom:40%;" />

作者额外的证明了一些其它的性质，比如如果距离函数是属于regular Bregman divergences（布雷格曼发散），推测一个点属于class的概率就是上面的softmax结果。简单查了一下，这个Bregman divergences的含义是说，它满足空间中距离所有点最小“距离”的点，就是所有点的平均值。这个条件是当且仅当的。

作者还证明了，如果使用欧式距离作为距离函数的话，求解属于哪个class的公式就等价于一个线性的模型：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20220310170123470.png" alt="image-20220310170123470" style="zoom:50%;" />

上面公式中的第一项对于不同的class都是固定的，而对于后面两项：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20220310170201612.png" alt="image-20220310170201612" style="zoom:50%;" />

求$x$属于class $k$的概率就等价于一个拥有参数$w_k$和$b_k$的线性模型。
