---
title: Re-eval-KGC
date: 2021-04-16 15:30:05
categories:
tags:
---

# A Re-evaluation of Knowledge Graph Completion Methods

2019-11-10 ACL 2020

重新发现目前的KGC方法中存在的问题，提出了一个RANDOM的评估策略。

<!--more-->

## 1 Introduction

最近出现的nn的model for Knowledge Graph Completion(KGC)，的效果存在问题：

> in ConvKB, there is a 21.8% improvement over ConvE on FB15k-237, but a degradation of 42.3% on WN18RR, which is surprising given the method is claimed to be better than ConvE.

它们在一个数据集(FB15K-237)上取得很好的结果，但是在另外的数据集(WRR18)上的效果反而下降了。本论文就针对这个问题进行了调查。发现是由于它们的评估策略的问题。

## 3 Observations

经过调查发现，在最后进行评估的时候，部分受到影响的模型如ConvKB，KBAT等，它们会对于很多的negative sample产生和valid triple一样的score。

> On average, ConvKB and CapsE have 125 and 278 entities with exactly same score as the valid triplet over the entire evaluation dataset of FB15k-237, whereas ConvE has around 0.002,

在这样的情况下，如果一开始的valid triple是作为评估triple的开头的话，效果就会虚假的高。

## 4 Evaluation Method

因此，论文就提出了一个评估的策略：*RANDOM*

> RANDOM:
>
> In this, the correct triplet is placed randomly in $\cal{T^{'}}$ .

其中，
$$
\cal{T^{'}} = \{ (h, r, t^{'})\ |\ t^{'} \in \cal{E} \}
$$

> RANDOM is the best evaluation technique which is both rigorous and fair to the model.

