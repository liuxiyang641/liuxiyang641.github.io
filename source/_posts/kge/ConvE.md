---
title: ConvE
date: 2021-04-15 17:12:23
categories:
- Paper
- KGE
tags:
---

# Convolutional 2D Knowledge Graph Embeddings

2018-7-4

## 1 Introduction

第一个利用CNN学习KGE的方法。

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20200314161529853.png)

<!--more-->

现在的knowledge graph会存在很多missing links，比如在Freebase和DBpedia中，超过66%的person实体没有到出生地的link。由于在知识图谱当中存在上百万的facts，所以模型的效率和计算代价就需要特别的考虑。

CNN具有能够快速计算的特性，因此可以应用与knowledge graph embedding。

## 2 Convolutional 2D Knowledge Graphs Embeddings

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20200314161529853.png)

首先对$e_1\in R^k$和$r_r\in R^k$的一维的embedding进行转换为2维的形式：
$$
[\bar{e_s}; \bar{r_r}] \\
\bar{e_s},\ \bar{r_r}\in R^{k_w\times k_h} \\
k=k_w\times k_h
$$
即：
$$
\begin{pmatrix} 
a & a & a\\ 
b & b & b\\
a & a & a\\
b & b & b\\
\end{pmatrix}
$$
改变为这种两个embedding相间的格式。

之后进行卷积操作：
$$
relu([\bar{e_s}; \bar{r_r}] \star \cal{w})
$$
然后变回一维矩阵
$$
vec(relu([\bar{e_s}; \bar{r_r}] \star \cal{w}))
$$
过一个全连接层，
$$
relu(vec(relu([\bar{e_s}; \bar{r_r}] \star \cal{w}))W)
$$
最后和目标embedding相乘，就得到了score。
$$
\psi_r(e_s,e_o)=relu(vec(relu([\bar{e_s}; \bar{r_r}] \star \cal{w})) W)e_o
$$
训练loss
$$

$$