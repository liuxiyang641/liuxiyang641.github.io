---
title: GRL
notshow: false
date: 2021-06-29 15:26:35
categories:
- Paper
- KRL
tags:
- KG
---

# Generalized Relation Learning with Semantic Correlation Awareness for Link Prediction

AAAI 2021

作者提出了一种能够捕获KG中relation的semantic correlations的方法，叫做GRL（Generalized Relation Learning）。这个方法在一般的embedding方法之后，利用输出的embedding评估relation之间的相似程度。

<!--more-->

## Introduction

**motivation**：作者认为目前用于link prediction的基于embedding方法存在两个问题：

1. 忽略了对于few-shot relation的学习，大多数方法假设不同relation有足够的实例进行学习
2. 无法学习zero-shot relation

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629162808164.png)

根据调研的情况来看，大多数的relation是few-shot relation。

**method**：作者利用many-shot relation来为相似的few-shot和zero-shot relation提供信息。主要做法是提出GRL，学习relation之间的相关性correlation。

## Method

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629163143120.png)

首先，通过一个base model获取embedding，比如利用ConvE或者DistMult。

GRL详细的说有三个module，

在Attention module中，首先捕获头实体与尾实体之间可能存在的潜在relation信息，即学习一个头尾实体的联合表示joint vector $\mathbf{j}$

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629163754371.png)

然后，作者使用了一个Relation Memory Block保存所有的relation信息，$K$就是所有relation的数量。

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629163911233.png)

之后，作者希望从这个Relation Memory Block导出能够丰富$\mathbf{j}$的信息，

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629164240642.png)

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629164355894.png)

这一步实际就是捕获了不同relation之间的correlation，需要注意的是对于在$\mathbf{M}$中的预测目标$\mathbf{r}$，会被mask为0。$\alpha_{sim}$就是joint vector $\mathbf{j}$和不同relation之间的相似程度。这样，利用$\alpha_{sim}$，在遭遇zero-shot relation时，可以选择最相似的relation来替代zero-shot relation。

在Fusion module中，为了确定如何自适应的混合$\mathbf{j}$和$\mathbf{rk}$，使用了一个类似GRU的方法，计算一个weight scalar。

![、](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629164658959.png)

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629164715132.png)

最后，在classifier module，预测真实的relation：

![](../../../../../../../Library/Application Support/typora-user-images/image-20210629165004731.png)

其中，$W_c\in \mathbb{R}^{dim\times K}$，计算loss

![、](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629165133409.png)

最终，这个loss和base model的loss混合到一起

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210629165254620.png)

