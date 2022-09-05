---
title: ParamE
date: 2021-04-15 16:58:20
categories:
- Paper
- KGE
tags:
---

# ParamE

这篇文章提出了新的KGE方法叫做ParamE，既能够建模关系的翻译属性，又能够利用非线性的联系。

最大的创新点在于**将神经网络的参数看做是relation embedding**，在实验中，为了验证，设计了三个方法，ParamE-MLP，ParanE-CNN，ParamE-Gate

实际是为所有的relation都设计了单独的神经网络。

<!--more-->

最后得到的表示投影到tail entity embedding space中，类似于ConvE，与所有tail entity embedding相乘，经过sigmoid得到预测得分。

ParamE-MLP：三层，每一层都有W和b，维度是200，400，800，输入只有head entity embedding，激活函数relu；

ParamE-CNN：两层卷积层后变为vector，激活函数relu，第一层卷积核32个，第二层64个，都是3x3卷积核；

ParamE-Gate：使用了一个gate。

训练过程：每个epoch训练固定次数n，每次都按照比例抽取某一组关系，从中随机抽取batch size大小的triples。

在WN18RR和FB15k-237上实验，效果很好，在FB15k-237上的MRR达到了0.399，最好的模型是ParamE-Gate。

优点：将relation embedding作为network parameters，能够充分的捕获关系和实体之间的交互，也能够表示“翻译”这种属性。

缺点：如果模型很复杂，那肯定不能将所有网络的参数都作为relation embedding。可以尝试在哪些模型中哪一部分适合作为由relation embedding构建的参数，哪些不适合。