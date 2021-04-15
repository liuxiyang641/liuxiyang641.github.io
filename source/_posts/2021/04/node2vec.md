---
title: node2vec
date: 2021-04-15 17:44:35
categories:
- Paper
- Graph Embedding
tags:
---

本文就提出了一种无监督的方法。核心思想：通过特定的游走方式进行采样，对于每个点都会生成 对应的序列。再将这些序列视为文本导入skip-gram模型，即可得 到每个节点的向量

<!--more-->

{% pdf node2vec-Scalable-Feature-Learning-for-Networks.pdf %}

