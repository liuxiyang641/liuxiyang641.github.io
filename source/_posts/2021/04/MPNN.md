---
title: MPNN
date: 2021-04-15 16:11:26
categories:
- Paper
- GNN
tags:
---

# Neural Message Passing for Quantum Chemistry

ICML 2017

Google Brain, Google DeepMind

本文就提出了一种图上进行监督学习的泛化框架Message Passing Neural Networks (MPNNs)

<!--more-->

## 1. Introduction

虽然化学家们已经尝试将机器学习应用到化学任务上，但是之前的很多工作还是在围绕特征工程打交道。虽然神经网络在其它很多领域已经很成功，但是在化学领域还处在很初始的阶段。

最近，随着high throughput experiments的进步，量子化学计算与分子动态模拟产生了大量的数据，导致之前的经典方法无法再处理这样数据量的数据，需要一种新的更灵活的方法。

而在化学分子上设计的神经网络需要满足图同构的情况下不变：

> The symmetries of atomic systems suggest neural networks that operate on graph structured data and are invariant to graph isomorphism might also be appropriate for molecules.

本文就提出了一种图上进行监督学习的泛化框架Message Passing Neural Networks (MPNNs)

预测任务是预测小型有机分子化学属性。

使用QM9数据集。

## 2. Message Passing Neural Networks

MPNN，泛化了至少之前的8种方法。分为两大阶段，message passsing phase和readout phase。

### message passsing phase

包括两个函数，消息函数Message Funciton和Update Function。

Message Function：用来产生消息，$M_t(h_v^t, h_w^t, e_{v,w})$
$$
m_v^{t+1}=\sum_{w\in N(v)} M_t(h_v^t, h_w^t, e_{vw})
$$
Update Function: 更新节点状态
$$
h_v^{t+1}=U_t(h_v^t, m_v^{t+1})
$$

### readout phase

这一阶段是针对图级别的任务。
$$
y^\prime=R(\{ h_v^T | v\in G \})
$$

## 3 MPNN Variants

接下来描述MPNN中具体实现的时候使用的结构。

基于GG-NN进行探究，

