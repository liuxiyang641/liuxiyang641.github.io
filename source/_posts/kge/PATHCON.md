---
title: PATHCON
notshow: false
date: 2021-08-06 17:33:59
categories:
- Paper
- GNN
tags:
- GNN
- KG
---

# Relational Message Passing for Knowledge Graph Completion

在这篇论文中，作者只考虑了KG中的relation embedding，没有学习entity embedding。更具体的说，学习两个方面的结构信息，relational context和relation paths。前者是头/尾实体的邻居relation，后者是头尾实体在KG中相连的relational path。提出了[PATHCON](https://github.com/hwwang55/PathCon)

作者预测的是relation prediction，$<h,?,t>$，区别于常见的head/tail prediction，这样情况下relation prediction的候选项是所有的relation，会少很多候选项。这篇文章，作者还提出了一个新的数据集，DDB14，基于医药和疾病的一个知识图谱。

<!--more-->

## 1 Introduction

**motivation**：作者认为实体的周围关系有很丰富的信息，可以使用GNN来学习这样的邻居结果。但是一般的KG上的GNN是迭代的将消息从实体传递到另外的实体。作者认为KG中的relation应该起到更大的作用。

**method**：作者提出了一种relational message passing的方法，只考虑relation embedding，然后让messages在relation之间传播。同时，为了降低计算复杂度，作者提出了改进版alternate relational message passing，让relation先传递给entity，再传递给relation。

作者认为重点建模relation而不是entity有三方面的好处：

> (1) it is inductive, since it can handle entities that do not appear in the training data during inference stage; 
>
> (2) it is storage-efficient, since it does not calculate embeddings of entities; and 
>
> (3) it is explainable, since it is able to provide explainability for predicted results by modeling the correlation strength among relation types.

这篇文章，作者还提出了一个新的数据集，DDB14。

## 2 Method

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807105609410.png" alt="image-20210807105609410" style="zoom:50%;" />

**Alternate relational message passing** 

为了降低以edge为底的聚合方法的计算复杂度（作者提供了计算复杂度的计算公式，没有细看），提出了交替的关系消息传递函数。让relation先传递给entity，再传递给relation。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807103941361.png" alt="image-20210807103941361" style="zoom:50%;" />

注意，这个公式里面的，$e$和$v$表示边relation和节点entity。$N(v)$不是表示实体$v$的邻居实体，而是实体$v$的邻居关系。$A_1,\ A_2$是两个构造函数。

接下来解释作者使用到的两个KG上的关系信息

### Relational Context

头尾实体的周围关系。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807104339704.png" alt="image-20210807104339704" style="zoom:50%;" />

### Relational Paths

头尾实体在KG上可以相连的路径，注意，为了限制路径数量，该路径被限制为各个节点的实体在这个路径上是唯一的，这样避免出现循环圈这样的情况。

对于一个路径$p$​​，作者只保留中间的各个relation，然后给这样的relation赋予一个独立的embedding $s_p$​，而不是去用某种方式产生。

虽然这种做法看起来会导致参数量爆炸，实际上作者发现出现的relational path数量并不多，单独赋予embedding，是可以接受的。

### Combining Relational Context and Paths

对于要预测的三元组$<h,?,t>$，首先产生一个relation context

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807105123264.png" alt="image-20210807105123264" style="zoom:50%;" />

之后，基于注意力聚合Relational Paths

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807105237434.png" alt="image-20210807105237434" style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807105253024.png" alt="image-20210807105253024" style="zoom:50%;" />

最终的预测输出

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210807105324839.png" alt="image-20210807105324839" style="zoom:50%;" />

