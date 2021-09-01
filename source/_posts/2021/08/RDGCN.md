---
title: RDGCN
notshow: false
date: 2021-08-27 16:10:58
categories:
- Paper
- GNN
tags:
- GNN
- KGE
---

# Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs

IJCAI 2019

[**RDGCN**](https://github.com/StephanieWyt/RDGCN) (Relation-aware Dual-Graph Convolutional Network)，预测任务是KG的实体对齐，主要是为了捕获更多的在dual KG中的relation的信息。核心创新点是对于dual KG（即要对齐的两个KG），构造了Dual Relation Graph，捕获relation和relation之间的联系。之后在这个Dual Relation Graph上学习relation的表示，融入到original KG中进行entity的表示学习，最终用于entity之间的对齐。

<!--more-->

## Method

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827161832004.png" alt="image-20210827161832004" style="zoom:50%;" />

### Constructing the Dual Relation Graph

有两个KG，$G_1$和$G_2$，然后这两个图看做是一个大的graph，$G_e$。注意，两个KG没有相连。

构造relation graph，relation作为node，如果两个relation具有相同的头/尾实体，那么两个relation node构造一条边。

更进一步，为这个边赋值一个权重，表示两个关系相连的紧密程度。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827162305649.png" alt="image-20210827162305649" style="zoom:50%;" />

$H$和$T$是所有的头/尾实体。

### Dual Attention Layer

这一层是用来捕获更复杂的关系信息，从而辅助下面的实体表示的学习。

在dual realtion graph中，一个关系node的表示为：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827193821221.png" alt="image-20210827193821221" style="zoom:50%;" />

需要注意，这里没有给关系赋予一个独立的表示，而是直接使用头尾实体的平均表示。

之后，基于gat进行relation之间的聚合。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827193932914.png" alt="image-20210827193932914" style="zoom:50%;" />

### Primal Attention Layer

利用上面学习到的relation的表示，在original graph中进行实体的表示学习。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827194018172.png" alt="image-20210827194018172" style="zoom:50%;" />

随后，使用残差

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827194112479.png" alt="image-20210827194112479" style="zoom:50%;" />

### Incorporating Structural Information

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827194151191.png" alt="image-20210827194151191" style="zoom:50%;" />

使用highway gnn更新，保留上一步的信息

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827194230407.png" alt="image-20210827194230407" style="zoom:50%;" />

最后，进行实体对齐

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210827194312252.png" alt="image-20210827194312252" style="zoom:50%;" />
