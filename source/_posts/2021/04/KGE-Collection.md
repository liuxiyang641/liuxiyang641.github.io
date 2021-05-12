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

- HoLE (AAAI 2016)
- TACT(AAAI 2021)
- TransF(ICPR 2018)
- TransCoRe(JCST 2018)

<!--more-->

## HoLE

**Holographic Embeddings of Knowledge Graphs**

AAAI 2016

这篇文章提出了holographic embeddings (HOLE)，来学习KG的compositional vector space representations。

**motivation**：However, existing embedding models that can capture rich interactions in relational data are often limited in their scalability. Vice versa, models that can be computed efﬁciently are often considerably less expressive.

**methods**：直接从subject entity embedding和object entity embedding中，使用circular correlation获得新的embedding，称作holograph embedding，然后使用这个holograph embedding与relation embedding做点积，得到预测概率。

![](KGE-Collection/image-20210418184909978.png)

一个图示：

![](KGE-Collection/image-20210418181121701.png)

从这个图能够看出来，Circular Correlation可以看做是tensor dot的一种压缩方式，它的输出结果的每一维都是tensor dot结果的一部分。

## TACT

**Topology-Aware Correlations Between Relations for Inductive Link Prediction in Knowledge Graphs**

AAAI 2021

[TACT](https://github.com/MIRALab-USTC/KG-TACT)，作者主要考虑的是inductive link prediction，使用gnn，捕获relation之间的语义上的关联性，即semantic correlation。作者认为relation之间的关联性通过relation的拓扑结构得到体现，因此，作者将所有的relation之间相连的拓扑结构分为7种，在relation形成的graph中进行学习，提出了RCN。

然后看一下整体结构：

![](KGE-Collection/image-20210510170730416.png)

## TransF

**Knowledge Graph Embedding with Multiple Relation Projections**

ICPR 2018

基于翻译的方法，在TransR的思想上的改进。考虑了每个relation不是独立的，而是具有Correlation，比如关系*“/people/person/place_of_birth*和*/people/person/nationality*就有较强的相关性，比如居住在纽约的人大概率是美国人。为了解决这个问题，作者直接将每个relation独立的matrix分为一系列的basis space的组合，对于不同relation有不同的组合系数。

![](KGE-Collection/image-20210511102258964.png)

公式：

![](KGE-Collection/image-20210511102445139.png)

在实验中，在FB15k-237数据集上，作者使用了维度100，s数量5；在WN18RR数据集上，维度50，s数量5。最后使用TransR的方法投影：

![](KGE-Collection/image-20210511102512774.png)

## TransCoRe

作者考虑了关系之间的correlation，首先利用SVD和PCC方法分析了TransE这些方法学习到的relation embedding之间的相关性，然后发现在所有relation组成的matrix中，存在low-rank的structure。因此，作者直接将relation matrix拆分为两个矩阵的乘积，一个是通用矩阵，一个是关系矩阵，每一列对应不同的relation。
$$
\mathbf{R}=\mathbf{U}\mathbf{V}
$$
![](KGE-Collection/image-20210511110706718.png)

在这种情况下，矩阵$\mathbf{U}$的列是关系空间的basis

![](KGE-Collection/image-20210511110904912.png)

最后

![](KGE-Collection/image-20210511110958068.png)

