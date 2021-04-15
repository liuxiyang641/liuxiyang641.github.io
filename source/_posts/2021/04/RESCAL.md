---
title: RESCAL
date: 2021-04-15 17:15:27
categories:
- Paper
- KGE
tags:
---

# A Three-Way Model for Collective Learning on Multi-Relational Data

2011

> we propose the relational learning approach RESCAL which is based on a tensor factorization that is related to DEDICOM but does not exhibit the same constraints.

<!--more-->

## 1 Introduction

说明最近tensor decomposition方法逐渐被应用在relational learning，原因：

- 从建模的角度讲，tensor decomposition更直接，各种relation可以直接表示为high-order tensor。同时，没有先验知识需要
- 从learning的角度讲，关系型的数据通常是高维并且稀疏的，适用于tensor decomposition。

关系型数据的重要特征是相关性能够通过各种相连的node产生，但是目前的模型都无法很好的满足要求。

> we propose the relational learning approach RESCAL which is based on a tensor factorization that is related to DEDICOM but does not exhibit the same constraints.

## 2 Modelling and Notation

看一下对于relational data如何表示：

<img src="wimage-20200223224511662.png" alt="image-20200223224511662" style="zoom:50%;" />

整体的数据被表示为张量$\cal{X}$，$\cal{X}_{ijk}=1$，表示fact存在。

## 4 Methods and Theoretical Aspects

文中定义了collective learning，大概含义是通过利用相关node的信息perform task。

> We will refer to the mechanism of exploiting the information provided by related entities regardless of the particular learning task at hand as collective learning.

### 4.1 A Model for Multi-Relational Data

核心在于：
$$
{\cal{X}_k} \approx AR_kA^T,\ k = 1,2,\dots,m,\\
A \in R^{n\times r}, R_k\in R^{r\times r}
$$
对于该式子的理解是，$R_k$表示关系$r$的转换，$R_kA^T$将$A$转换到了$R_k$表示的向量空间当中，通过与$A$乘积，最终得到的对于$(h,r,t)$，通过点积表示fact。

通过最小化得到最终的embedding：
$$
min\ f(A,R_k)+g(A,R_k) \\
f(A,R_k)=\frac{1}{2}(\sum_k {\lVert {\cal{X}}-AR_kA^T \rVert}_F^2) \\
g(A,R_k)=\frac{1}{2}\lambda({\lVert A \rVert}^2_F + \sum_k{\lVert R_k \rVert}^2_F)
$$

## 5 Evaluation

进行了四方面的比较，

- Collective Classiﬁcation
- Collective Entity Resolution
- Kinships, Nations and UMLS
- Runtime Performance and Technical Considerations