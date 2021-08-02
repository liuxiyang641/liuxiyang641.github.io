---
title: GNN-Collection
notshow: false
date: 2021-07-08 19:48:05
categories:
- Ppaer
- GNN
tags:
- Collection
---

# Collection of GNN papers

- Highway GNN（ACL 2018）
- HGAT（EMNLP 2019）
- HetGNN（KDD 2019）



<!--more-->



## Highway GNN

[**Semi-supervised User Geolocation via Graph Convolutional Networks**](https://github.com/ afshinrahimi/geographconv) ACL 2018

应用场景是社交媒体上的用户定位。单纯的在GNN上的创新点是使用Gate机制来控制传入的邻居的信息。

在每一层，借鉴Highway networks的思路，计算一个门weight

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210708195301778.png" style="zoom:50%;" />

## HGAT

**Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classiﬁcation** EMNLP 2019

为短文本分类任务（semi-supervised short text classiﬁcation）设计了一个异质图神经网络HGAT。

首先是利用原始文本构造一个异质图（HIN），把不同来源的文本组合到一起。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180026169.png" alt="image-20210728180026169" style="zoom:50%;" />

重点在于，其中的node type各不相同，各自具有差异性很大的特征。

然后是设计的网络结构，重点在于设计了一个两层的attention。

不同type的node有不同的卷积核：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180219517.png" alt="image-20210728180219517" style="zoom:50%;" />

然后，type-level的attention，聚合邻居下所有相同type的node embedding，然后计算attention weight。这样同一type下的所有neighbor node共享一个type level的weight。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180348109.png" alt="image-20210728180348109" style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180319175.png" alt="image-20210728180319175" style="zoom:50%;" />

不同type之间softmax。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180440793.png" alt="image-20210728180440793" style="zoom:50%;" />

然后是node-level的attention，不同邻居node，计算attention。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180538577.png" alt="image-20210728180538577" style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180555435.png" alt="image-20210728180555435" style="zoom:50%;" />

最后结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210728180643981.png" alt="image-20210728180643981" style="zoom:50%;" />

## HetGNN

[Heterogeneous Graph Neural Network](https://github.com/chuxuzhang/KDD2019_HetGNN) KDD 2019

作者提出了一种同时处理node content和heterogeneous graph structure的GNN，HetGNN。

看一下整体结构：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210802205726358.png" alt="image-20210802205726358" style="zoom:50%;" />

核心模块有三方面：

**Sampling Heterogeneous Neighbors**：使用了random walk with restart (RWR)的邻居采样策略，需要注意的是这个采样策略保证对于node $v$，能够采样到所有不同类型的邻居。然后相同类型的邻居聚合到一起。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210802205956510.png" alt="image-20210802205956510" style="zoom:50%;" />

**Encoding Heterogeneous Contents**：对于不同格式的content，使用不同的网络进行处理，然后使用Bi-LSTM进行融合，不同type的node有自己的Bi-LSTM网络。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210802210216977.png" alt="image-20210802210216977" style="zoom:50%;" />

**Aggregating Heterogeneous Neighbors**：对于相同类型的邻居，先基于Bi-LSTM进行聚合。然后不同类型的邻居基于attention进行聚合。

