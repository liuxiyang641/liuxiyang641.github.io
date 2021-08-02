---
title: HyperNetworks
notshow: false
date: 2021-07-24 22:07:31
categories:
- Paper
tags:
---

# HYPERNETWORKS

ICLR 2017

核心贡献是将Hypernetwork扩展到了convolutional networks和long recurrent networks，证明其在使用更少的参数情况下，在序列模型和卷积网络的多个预测任务下都达到了不错的训练结果。

<!--more-->

## Introduction

Hypernetwork是一个能够为另一个更大的网络产生weight的较小的网络。示例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725110111444.png" alt="image-20210725110111444" style="zoom:50%;" />

> In this work, we consider an approach of using a small network (called a “hypernetwork") to generate the weights for a larger network (called a main network).

> hypernetwork takes a set of inputs that contain information about the structure of the weights and generates the weight for that layer

HyperNEAT是一个使用hypernetwork的实例，输入时weight的virtual coordinates。

这篇文章的hypernetwork直接接收一个描述weight的embedding vector。同时设计了CNN和RNN的两种变体。

## Method

### Static Hypernetwork : A Weight Factorization Approach For Deep Convolutional Networks

$K^j$是第$j$​层的卷积核，一共有$D$层

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111008315.png" alt="image-20210725111008315" style="zoom:50%;" />

它由一个hypernetwork产生，每层接收一个描述weight的embedding $z^j$

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111037575.png" alt="image-20210725111037575" style="zoom:50%;" />

具体产生方法，一个静态的hypernetwork，简单看了下实验，$z^j$是一个比较小的embedding，甚至只有4。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111153567.png" alt="image-20210725111153567" style="zoom:50%;" />

### Dynamic Hypernetwork : A Daptive Weight Generation For Recurrent Networks

一个新的RNN，weight是生成的：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111430981.png" alt="image-20210725111430981" style="zoom:50%;" />

这个hypernetwork同样是用另一个小的RNN产生。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111544605.png" alt="image-20210725111544605" style="zoom:50%;" />

示例图：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111631387.png" alt="image-20210725111631387" style="zoom:50%;" />

实际上，作者使用了另一种简化的版本，每一层定义了一个weight scaling vector $d$​​，不再是完成生成weight matrix，而是生成weight vector。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111733324.png" alt="image-20210725111733324" style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210725111717560.png" alt="image-20210725111717560" style="zoom:50%;" />

