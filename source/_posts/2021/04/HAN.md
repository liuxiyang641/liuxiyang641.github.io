---
title: HAN
date: 2021-04-15 16:10:10
categories:
- Paper
- GNN
tags:
---

# Heterogeneous Graph Attention Network

2019-4-13

## 1 INTRODUCTION

HAN设计了两个层次的attention机制，

- Semantic-level attention：在不同的meta-path中选择weight
- Node-level attention：对于一个节点，它的邻居的weight

<!--more-->

## 2 RELATED WORK

> The graph convolutional neural work generally falls into two categories, namely spectral domain and non-spectral domain.

## 4 THE PROPOSED MODEL

总的来说，HAN包括了两个层次的attention，node-level和semantic-level，node-level attention的输出作为semantic-level层次的输入。

### 4.1 Node-level Attention

对于$(i,j,\Phi)$，$i,j$表示节点，$\Phi$表示meta-path。

node-level attention针对的目标是同一个meta-path下的nodes的weight。

首先根据node type确定投影embedding，
$$
h^{'}_i=M_{\phi_i}h_i \\
\phi_i: node \ i \ type
$$
计算attention value，
$$
\alpha_{ij}=\frac{exp(\sigma(a^T[h_i^{'}||h_j^{'}]))}{\sum_{k\in N_i^{\Phi}}  exp(\sigma(a^T[h_i^{'}||h_k^{'}]))}
$$
把所有node的embedding结合起来，
$$
z_i^\Phi = \sigma(\sum_{j\in N_i^{\Phi}} \alpha_{ij}^\Phi h_{ij}^{'} )
$$
类似于GAT，采用multi-head attention，
$$
z_i^\Phi = ||_{k=1}^K \sigma(\sum_{j\in N_i^{\Phi}} \alpha_{ij}^\Phi h_{ij}^{'} )
$$
这是一种meta-path下一个节点$i$的最终输出，对于所有的$\Phi$与全部的node，产生$\{z_i^{\Phi_0}, z_i^{\Phi_1},\dots z_i^{\Phi_p}\}$。

## 4.2 Semantic-level Attention

要计算各种类型的meta-path的weight，就要在全局的情况下计算，
$$
\omega_{\Phi_p}=\frac{1}{|V|}\sum_{i\in V}q^T tanh(Wz_i^{\Phi_p}+b) \\
\beta_{\Phi_p}=softmax(\omega_{\Phi_p})
$$
最后求和，得到最终的embedding，
$$
Z_i=\sum_{p=1}^P \beta_{\Phi_p}z_i^{\Phi_p}
$$
<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20200221215000895.png" alt="image-20200221215000895" style="zoom: 33%;" />

## 5 EXPERIMENTS

略