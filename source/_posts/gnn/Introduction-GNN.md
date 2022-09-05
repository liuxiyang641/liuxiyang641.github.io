---
title: Introduction-GNN
date: 2021-04-15 16:20:44
categories:
- Paper
- GNN
tags:
---

# Introduction to Graph Neural Networks

Note of book 《Introduction to Graph Neural Networks》

<!--more-->

## Introduction

所谓的图graph就是一种数据结构，由节点集合与边集合组成。具有图结构的数据存在于众多的领域中，包括社交网络、化学分子、知识图谱等。最近图神经网络的出现使得在图上的深度学习成为了可能，图神经网络也受到了大量的研究关注。

图神经网络起源于两大部分，卷积神经网络（Convolutional Neural Network）与图嵌入（Graph Embedding）。

CNN在图像（image）上的成功在于它能够导出多尺度（multi-scale）局部化（localized）的空间特征。应用CNN的三个关键点在于局部连接（local connection）、共享参数（shared weight）以及多层结构（multi-layer）。这三个点对于图同样是很重要的，首先图中的绝大多数结构都是局部连接的；其次，共享参数能够帮助减少传统的基于谱图理论的图算法计算复杂度；最后，多层的结构能够用来捕获图中的层级结构信息。但是，由于图像是规则的欧几里得域（Euclidean domian）下的数据结构，而图是非欧几里得数据的（non-Euclidean），因此我们无法直接将传统的CNN应用到图上。我们需要一种新的模型，在保留CNN的三个关键特征的同时，能够处理非欧式空间下的图结构。

> 简单辨析下欧几里得的数据以及非欧几里得下的数据：
>
> 在机器学习中处理的数据可以分为欧几里得和非欧几里得两类。两者的核心区别在于是否“排列整齐”，前者对于数据，可以使用$\mathbb{R}^n$的空间进行描述，不丢失任何的原始信息。这一类数据的代表是图像、文字、一般的数值数据等。非欧几里得的数据实际广泛存在，主要有图（graph）和流型（mainfold）两类。对于这一类的数据，比如graph中的每个节点的邻居节点、邻居边都各不相同，数量不一，无法使用一个$n$维的空间来完全描述此类数据。

图嵌入的发展收到词嵌入（word embedding），Deepwalk被认为是首个学习图嵌入的方法，它将图中的每个节点都表示为了$n$维的嵌入向量。Deepwalk首先在图上进行随机游走采样，之后应用SkipGram方法学习到合适的嵌入表示。在Deepwalk后，node2Vec，LINE，TADW这些方法逐步发展。但是这些方法存在两方面的缺点，一个是由于需要随机游走，学习到的方法很难直接用到新数据上，泛化性较差；另一方面是没有参数共享，导致随着图节点增多，模型参数也不断增加。

图神经网络在前两者的基础上，首先尝试将整个图都嵌入到欧式空间中，之后应用CNN的思想增强模型的表达能力，让嵌入能够表示更多的原始信息。

在这本书中，对于GNN的分类为：

- 循环图神经网络（recurrent graph neural networks）
- 卷积图神经网络（convolutional graph neural networks）
- 图自编码器（graph auto-encoders）
- 时空图神经网络（spatial-temporal graph neural networks）

## Vanilla Graph Neural Networks

关于图神经网络GNN的概念实际上在2005年前后就已经有对应概念的提出[The graph neural network model; Graphical-based learning environments for pattern recognition]。

接下来介绍一个原始的模型vanilla GNN。它针对的是无向同质的图。

核心包括两个函数，局部转移函数（local transition function）以及局部输出函数（local output function）。
$$
\mathbf{h}_v = f(\mathbf{x}_v,\mathbf{x}_{co[v]}, \mathbf{x}_{ne[v]}, \mathbf{h}_{ne[v]},) \\
\mathbf{o}_v = g(\mathbf{h}_v, \mathbf{x}_v)
$$
其中，$\mathbf{x}_{co[v]}$是邻居边的特征，$\mathbf{x}_{ne[v]}$是邻居节点的特征，$\mathbf{h}_{ne[v]}$是邻居节点的隐藏状态。

vanilla GNN不断迭代更新函数$h$ $T$步，直到到达固定点，使得$\mathbf{h}_v^T\approx \mathbf{h}_v^{T-1}$。这一操作的原理是Banach’s ﬁxed point theorem[An Introduction to Metric Spaces and Fixed Point Theory]。到达不动点之后，再进行梯度下降。

vanilla GNN的几个缺点：

- 计算不够有效，每次都需要不断迭代T步之后才能进行梯度下降，实际上一般的神经网络是直接产生一个输出后就可进行梯度更新。
- 在T步的迭代中，vanilla GNN一直使用的是一样的参数，这就导致图的层级结构信息没有能够被显式的学习。实际上我们可以让模型的每次迭代都学习不同的参数。
- vanilla GNN没有有效的建模边的信息。
- 如果图中的节点数量很多，使用不动点这种原理，可能会导致各个节点过平滑，差异性不够

## Graph Convolutional Networks

主要有两类在图上进行卷积操作的GNN，一类是谱空间下卷积操作，一类是空间领域下的卷积操作。

### Spectral Methods

#### Spectral GNN

2014年。Spectral networks and locally connected networks on graphs

通过对图拉普拉斯矩阵进行特征分解，定义了在傅里叶域下的卷积操作。
$$
\mathbf{g}_\theta \star \mathbf{x} = \mathbf{U} \mathbf{g}_\theta \mathbf{U}^T \mathbf{x}
$$
详细的可以看之前的GCN笔记。

#### ChebNet

2011年。Convolutional neural networks on graphs with fast localized spectral ﬁltering.

对上面公式中的$\mathbf{g}_\theta$使用切比雪夫多项式进行估计，从而无需计算特征向量$\mathbf{U}$。

#### GCN

2017年。Semi-supervised classiﬁcation with graph convolutional networks.

出现了著名的图卷积算子，它实际是在CheNet进一步的简化，约定了切比雪夫多项式只包括前两步。
$$
\mathbf{Z}=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} \mathbf{X} \Theta
$$

#### AGCN

2018年，Adaptive graph convolutional neural networks.

AGCN假设除了图的结构能够表现图中节点的关系外，它们之间应该还存在某种隐式的联系。

### Spatial  Methods

基于空间域的方法是直接在graph上进行操作，不需要计算特征向量。

Spatial methods 和 Spectral methods的对比：

由于效率、通用性和灵活性问题，空间模型优于光谱模型。首先，光谱模型的效率低于空间模型。谱模型要么需要进行特征向量计算，要么需要同时处理整个图。由于空间模型通过信息传播直接在图域中执行卷积，因此空间模型更适合于大图形。计算可以在一批节点中进行，而不是整个图。其次，依赖于图Fourier基的谱模型很难推广到新的图。它们假设一个固定的图。对图的任何扰动都会导致特征基的变化。另一方面，基于空间的模型在每个节点上执行局部的图形卷积，在这些节点上可以很容易地在不同的位置和结构之间共享权重。第三，基于谱的模型仅适用于无向图。基于空间的模型更灵活地处理多源图形输入

#### Neural FPS

2015年，Convolutional networks on graphs for learning molecular ﬁngerprints

核心思想是对于具有不同度数的节点学习不同的权值矩阵。缺点是很难直接应用到大规模的图上。

#### PATCHY-SAN

2016年，Learning convolutional neural networks for graphs.

核心思想是对于图中的每个节点，基于广度优先搜索选择固定$k$个邻居作为接受域，这样就将图处理问题转化为了传统的欧几里得数据问题，最后输入到CNN中进行处理。

#### DCNN

2016年，Diﬀusion-convolutional neural networks.

扩散卷积神经网络

#### DGCN

2018年，Dual graph convolutional networks for graph-based semisupervised classiﬁcation

重点读：双向卷积神经网络，既考虑了局部一致性，也考虑了全局一致性

#### LGCN

2018年，Large-scale learnable graph convolutional networks.

#### MONET

2017年，Geometric deep learning on graphs and manifolds using mixture model CNNs.

#### GraphSAGE

2017年

## Graph Attention Network

### GAT

2018年，Graph attention networks

使用多头注意力机制。

### GaAN

2018年，GaAN: Gated attention networks for learning on large and spatiotemporal graphs.

同样使用多头注意力机制，但是使用了key-value的机制。

## Graph Recurrent Networks

融合了gate机制的模型，核心思想是提升图信息long term聚和的效果。

### GGNN

2016年，Gated graph sequence neural networks.

利用GRU，每次聚合$T-1$步下的邻居信息，输入GRU，得到最后的输出

### Tree-LSTM

2015年，Improved semantic representations from treestructured long short-term memory networks.

利用LSTM

### Graph LSTM

2017年，Cross-sentence N-ary relation extraction with graph LSTMs.

### Sentence-LSTM

2018年，Sentence-state LSTM for text representation.

将文本直接转换为graph，然后使用GNN进行学习，在很多NLP任务上表现出了很好的性能。



## Graph Redidual Networks

使用残差链接，为了缓解在GCN中聚合多层信息效果反而下降的情况。

### Highway GCN

2018年，Semi-supervised user geolocation via graph convolutional networks.

在更新节点状态时，使用了门机制。

### Jump Knowledge Network

2018年，Representation learning on graphs with jumping knowledge networks.

每一层都会直接连接向最后一层。

### DEEPGCNS

2019年，DeepGCNs: Can GCNs go as deep as CNNs?

使用skip connection和dense connection解决两个问题：GNN中的梯度消失以及过度平滑。

## Heterogeneous Graph

- HAN：
- PP-GCN：Fine-grained event categorization with heterogeneous graph convolutional networks.
- ActiveHNE：Activehne: Active heterogeneous network embedding.

## Multi-Dimensional Graph

- Multi-dimensional graph convolutional networks

## Sampling

- GraphSAGE（2017）: 邻居节点随机采样
- PinSage（2018）: 基于邻居节点重要性采样，Graph convolutional neural networks for web-scale recommender systems.
- FastGCN：FastGCN: Fast learning with graph convolutional networks via importance sampling
- Adaptive sampling towards fast graph representation learning：参数化，可训练的采样器
- SSE：Learning steady-states of iterative algorithms over graphs.

## Graph Auto Encoder

- GAE: Variational graph auto-encoders.
- ARGA: Adversarially regularized graph autoencoder for graph embedding.
- DGI: Deep graph infomax.

## General Framework

- MPNN：
- NLNN：Non-local neural networks.
- GN：Relational inductive biases, deep learning, and graph networks.

