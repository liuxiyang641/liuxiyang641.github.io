---
title: CapsE
date: 2021-04-15 17:01:21
categories:
- Paper
- KGE
tags:
---

# A Capsule Network-based Embedding Model for Knowledge Graph Completion and Search Personalization

2019-6-2s

## 1 Introduction

常用的KE模型，比如TransE，Complex，DISTMULT等模型，它们只捕获了三元实体之间的线性联系，没有捕获非线性的联系。

本论文的基础是在capsule networks（CapsNet）Dynamic routing between capsules的基础上，直接应用到knowledge graph triplet上。CPasNet原来是作用于图片上。

论文的理论是处在相同维度下的triplet，同一纬度下的embedding可以通过capsule（each capsule is a group of neurons） network捕获不同的变体。

<!--more-->

## 2 The proposed CapsE

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20200217182118840.png" style="zoom:50%;" />

特点：

1. 三元组直接作为一个矩阵进行训练
2. 最后的score function是向量的长度

具体请看论文描述

## 3 Knowledge graph completion evaluation

数据集：

- WN18RR
- FB15k-237

评估指标：

- Mean rank（MR）
- Mean reciprocal rank (MRR)
- Hits@10

Embedding 初始化：

- ConvKB和CapsE都使用了TransE训练好之后的embedding来初始化
- 对于TransE，在WN18RR数据集下，使用了100-dimensional Glove word embeddings初始化

参数设置：

- 初始的KE维度为100
- 过滤器filter数量的设置在{50，100, 200, 400}

关于关系r的分类：

> Following Bordes et al. (2013), for each relation r in FB15k-237, we calculate the averaged number $\eta_s$ of head entities per tail entity and the averaged number $\eta_o$ of tail entities per head entity. If $\eta_s$ <1.5 and $\eta_o$ <1.5, r is categorized one-to-one (1-1). If $\eta_s$ <1.5 and $\eta_o$ ≥ 1.5, r is categorized one-to-many (1-M). If $\eta_s$ ≥ 1.5 and $\eta_o$ <1.5, r is categorized many-to-one (M-1). If $\eta_s$ ≥ 1.5 and $\eta_o$ ≥ 1.5, r is categorized many-to-many (M-M)

最后得到的结果显示M-M的关系总是最多的

使用filtered设置进行训练