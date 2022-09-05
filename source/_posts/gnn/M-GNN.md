---
title: M-GNN
notshow: false
date: 2021-08-26 14:25:30
categories:
- Paper
- GNN
tags:
- KGE
- GNN
---

# Robust Embedding with Multi-Level Structures for Link Prediction

这篇文章提出了一种multi-level graph neural network，M-GNN。使用GIN中的MLP学习结构信息，然后提出了一种基于KG中图的不同粒度进行建模的方法。它会从原始的KG出发，不断合并邻居节点，合并边，构造出一系列不同粒度的graph，在这些graph上进行图卷积操作，得到最后的输出。除了一般的链路预测实验，作者还进行了在不同稀疏度以及加入noising edges的实验。

<!--more-->

## Graph Coarsening

和一般的GNN消息聚合方式不同，M-GNN希望能够建模KG中不同尺度中的信息。

首先构造k个Coarsened graph：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210826143747570.png" alt="image-20210826143747570" style="zoom:50%;" />

核心是两种合并结构的方法：**edge coarsening**和**neighbor coarsening**。

M-GNN考虑在图中的不同relation包括不同的结构信息：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210826144601695.png" alt="image-20210826144601695" style="zoom:50%;" />

对于1-1 structure，使用**edge coarsening**，1-1 structure是指两个edge没有相连的相同实体。edge coarsening直接把这样的edge分别看做是新的super node，对edge进行了coarsening。如下图a所示。edge coarsening使得节点包括了更多阶邻居的信息。

对于1-n或者n-1 structure，同一关系下的不同邻居实体共享某种相同的信息，可以进行聚合，把两个邻居实体合并为新的super node，叫做**neighbor coarsening**。如下图b, c所示。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210826145202651.png" alt="image-20210826145202651" style="zoom:50%;" />

对于n-n structure，可以看做是多个1-n或者n-1结构，不需要单独处理。

通过先neighbor coarsening压缩实体，然后edge coarsening进一步压缩graph中实体和关系数量。

## Multi-Level GNN

与产生不同粒度的graph的顺序相反，在进行GNN的消息传递时，先从最粗粒度的graph进行学习，然后到更细粒度的graph，每个graph对应一层GNN。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210826150307208.png" alt="image-20210826150307208" style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210826150339407.png" alt="image-20210826150339407" style="zoom:50%;" />

公式中的$S$是指实体到实体的对应矩阵。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210826150505564.png" alt="image-20210826150505564" style="zoom:50%;" />

