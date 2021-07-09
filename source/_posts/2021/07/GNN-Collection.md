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

- Highway GNN



<!--more-->



## Highway GNN

[**Semi-supervised User Geolocation via Graph Convolutional Networks**](https://github.com/ afshinrahimi/geographconv) ACL 2018

应用场景是社交媒体上的用户定位。单纯的在GNN上的创新点是使用Gate机制来控制传入的邻居的信息。

在每一层，借鉴Highway networks的思路，计算一个门weight

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210708195301778.png" style="zoom:50%;" />

