---
title: KGE-Collection
date: 2021-04-17 19:32:13
categories:
- Paper
- KGE
tags:
- Collection
---

# Collection of KGE Papers Volume 1

Collection of KGE papers, volume 1. 

Now it contains models of:

- HoLE (2016)
- TACT(2021)

<!--more-->

## HoLE: Holographic Embeddings of Knowledge Graphs

AAAI 2016

这篇文章提出了holographic embeddings (HOLE)，来学习KG的compositional vector space representations。

**motivation**：However, existing embedding models that can capture rich interactions in relational data are often limited in their scalability. Vice versa, models that can be computed efﬁciently are often considerably less expressive.

**methods**：直接从subject entity embedding和object entity embedding中，使用circular correlation获得新的embedding，称作holograph embedding，然后使用这个holograph embedding与relation embedding做点积，得到预测概率。

![](KGE-Collection/image-20210418184909978.png)

一个图示：

![](KGE-Collection/image-20210418181121701.png)

从这个图能够看出来，Circular Correlation可以看做是tensor dot的一种压缩方式，它的输出结果的每一维都是tensor dot结果的一部分。

## Topology-Aware Correlations Between Relations for Inductive Link Prediction in Knowledge Graphs

AAAI 2021

[TACT](https://github.com/MIRALab-USTC/KG-TACT)，作者主要考虑的是inductive link prediction，使用gnn，捕获relation之间的语义上的关联性，即semantic correlation。作者认为relation之间的关联性通过relation的拓扑结构得到体现，因此，作者将所有的relation之间相连的拓扑结构分为7种，在relation形成的graph中进行学习，提出了RCN。

然后看一下整体结构：

![](KGE-Collection/image-20210510170730416.png)

