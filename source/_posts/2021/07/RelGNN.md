---
title: RelGNN
notshow: false
date: 2021-07-02 11:38:37
categories:
- Paper
- GNN
tags:
- KRL
- GNN
---

# Relation-aware Graph Attention Model With Adaptive Self-adversarial Training

AAAI 2021

作者提出了RelGNN方法处理带attribute的heterogeneous graph，核心部分包括一个采取了注意力机制的R-GCN方法以及能够降低负采样false negative sample的自对抗负采样方法adaptive self-adversarial (ASA) negative sampling。

个人认为最大的创新点是这种负采样的方法ASA。负采样的核心思路是如何寻找最难区分discriminative的样本。而ASA方法的核心思想是计算正样本和负样本得分score之间的绝对差距，采样使这种差距最小的负样本。作者认为如果一个构造的负样本计算的得分比正样本的得分还要大，那么这样的负样本更有可能是false negative，因此不能直接选择这样最难区分的负样本，而是考虑正样本的预测值。

<!--more-->

## Introduction

**motivation**：两个问题

1. 作者认为目前处理heterogeneous graph的GNN在聚合邻居信息的时候没有考虑边的语义信息，知识在进行meta-path遍历或者消息构造函数时起到作用。
2. 目前训练采用的负采样方法，无法考虑false negatives的问题。

**method**：提出RelGNN

1. 在聚合消息时，使用注意力机制，同时考虑node state和edge state。
2. 提出ASA采样方法，使用每一次训练好的模型为下一次训练寻找negative samples。思路是认为一个positive sample的confidence level应该和它衍生的negative sample的概率是匹配的。

## Method

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702160109326.png" style="zoom:50%;" />

### attribute embedding

在Attributed Heterogeneous Graph中，对于不同node type，有不同的attribute schema，有不同的attribute。使用不同的方法处理这些特征，然后拼接，投影至相同空间中，获得node $v$的attribute embedding $h_{v}^{(0)}$。

### Message Passing

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702161233932.png" style="zoom:50%;" />

上面就是R-GCN。实际上，为了避免过度参数化，使用了R-GCN的*basis-decomposition*。

接下来，是使用了edge embedding的attention：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702161403878.png" style="zoom:50%;" />

采用多头机制，同时要注意这个attention是不包括self-loop传递过来的信息的。

最后，为了融合attribute embedding $h_{v}^{(0)}$以及graph embedding $h_{v}^{last}$，使用attention来融合

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702161953963.png" style="zoom:50%;" />

使用了*averaging*的multi-attention。

### ASA negative sampling

以前的自对抗采样方法self-adversarial negative sampling，寻找最难预测的负样本。

> The core idea is to use the model itself to evaluate the hardness of negative samples,

RelGNN预测三元组存在的概率，使用了DistMult来打分：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702163057832.png" style="zoom:50%;" />

之前的自对抗负采样方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702163147312.png" style="zoom:50%;" />

改进后的负采样方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702163241877.png" style="zoom:50%;" />

加入$u$之后，实际上是减小了小于正样本得分的负样本和正样本之间的差距，扩大了大于正样本得分的负样本和正样本之间的差距，让模型倾向于选择小于正样本得分的负样本。随着模型训练，模型越来越“正确”，可以考虑减小$u$的值，让模型去选择更难预测的负样本。

$u$如果太小，会让模型倾向选择更hard的负样本，增大false negative的概率。

$u$如果太大，会倾向于选择那些trivial samples，不够discriminate。

## Experiment

主要的结果忽略，可以学习的是它对于attention的可视化，计算每个node的领奖attention的熵entropy，计算不同节点的注意力的熵，熵约低，表示这个节点的邻居注意力差异越小，约不混沌，值约集中，越关注某些特定的邻居。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210702163855948.png" style="zoom:50%;" />
