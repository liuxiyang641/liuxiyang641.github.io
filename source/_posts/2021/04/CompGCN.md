---
title: CompGCN
date: 2021-04-15 16:27:01
categories:
- Paper
- GNN
tags:
---

# COMPOSITION-BASED MULTI-RELATIONAL GRAPH CONVOLUTIONAL NETWORKS

## 1 INTRODUCTION

原来的CNN, RNN等方法不能直接应用到graph上，因此最近GCN被提出来了。

但是初始的GCN方法主要集中与无向图，最近的针对有向图的方法类如R-GCN，存在over-parameterization问题。

> there is a need for a framework which can utilize KG embedding techniques for learning task-speciﬁc node and relation embeddings.
>
> COMPGCN addresses the shortcomings of previously proposed GCN models by jointly learning vector representations for both nodes and relations in the graph

<!--more-->

## 2 RELATED WORK

两个方面叙述

- GCN: 原始的GCN，之后的各种拓展，都在MPNN框架下，本论文提出的也是这样，但是专门为relational data设计过
- KE: translational、semantic matching based、neural network based

## 3 BACKGROUND

### GCN on undirected graph

图的表示形式：
$$
G=(\cal{V},\cal{E},\cal{X})
$$
其中$\cal{X}$表示所有entity的初始feature，$\cal{X}\in \cal{R}^{ |\cal{V}|\times d_0 }$。

获取归一化后的self-connection邻接矩阵：
$$
\hat{A}=\tilde{D}^{-\frac{1}{2}}(A+I)\tilde{D}^{-\frac{1}{2}} \\
\tilde{D}_{ii}=\sum_j{(A+I)_{ij}}
$$
某一层的GCN：
$$
H^{k+1}=f(\hat{A}H^kW^k) \\
H^0=\cal{X}
$$

### GCN on directed graph

图的表示形式：
$$
G=(\cal{V},\cal{R},\cal{E},\cal{X})
$$
$\cal{R}$表示relation的集合。

这种情况下对于关系数据的处理就存在区别了，基于Encoding sentences with graph convolutional networks for semantic role labeling 中提出的假设，

> information in a directed edge ﬂows along both directions

因此构造出反向关系inverse relation:
$$
(u,v,r)\in \cal{E}\ \ and\ \ (v,u,r^{-1})\in \cal{E^{-1}}
$$
此时的GCN：
$$
H^{k+1}=f(\hat{A} H^k W^k_r) \\
H^0=\cal{X}
$$

## 4 CompGCN DETAILS

图：
$$
G=(\cal{V},\cal{R},\cal{E},\cal{X},\cal{Z}) \\
\cal{X}\in \cal{R}^{ |\cal{V}|\times d_0 } \\
\cal{Z}\in \cal{R}^{ |\cal{R}|\times d_0 }
$$
构造关系：
$$
\cal{E^{'}} = \cal{E}\ \cup\ \{ (v,u,r^{-1}) | (u,v,r)\in \cal{E}\ \}\ \cup\ \{ (u,u,T) | u\in \cal{V} \}
$$
embedding更新方式：
$$
h_v^{k+1}=f(\sum_{(u,r)\in \cal{N}_v} W_{\lambda(x)^k}\phi(x_u^k, z_r^k))
$$
其中$\phi$函数，为了减少参数，可以为下面的三种方式，当然可以拓展为更多的方式：
$$
Sub: \ \phi(x_u, z_r) = x_u - z_r \\
Mult:\ \phi(x_u, z_r) = x_u * z_r \\
Circular-correlation:\ \phi(x_u, z_r) = x_u \star z_r
$$
其中的关系权值矩阵：
$$
W_{dir(r)}= \begin{cases} W_o,\ r\in \cal{R} \\ W_i,\ r\in \cal{R}_{inv} \\ W_S\ r\in \cal{T}(self-loop) \end{cases}
$$
对于关系relation的处理，与KBGAT一样：
$$
z_r^{k+1} = W_{rel}z_r^k \\
W_{rel}\in R^{d_1\times d_0}
$$
在第一层初始的时候，对于relation的定义是bias-vector。
$$
Z_r = \sum_b^B \alpha_{br}\bold{v}_b \\
\{ \bold{v}_1, \bold{v}_2,\cdots \bold{v}_B \}
$$

## 5 EXPERIMENTAL SETUP

进行了下面三个任务：

- Link Prediction：FB15k-237，WN18RR
- Node Classiﬁcation：MUTAG (Node) ， AM
- Graph Classiﬁcation：bioinformatics dataset：MUTAG (Graph) ， PTC

## 6 RESULTS

研究了下面四个方面的问题：

1. 在link prediction上的效果
2. 选择不同的composite operation效果
3. 模型对于不同数量的relation的数据集的效果
4. 在node和graph classiﬁcation的效果

> We ﬁnd that with DistMult score function, multiplication operator (Mult) gives the best performance while with ConvE, circular-correlation surpasses all other operators.

具体结果略

