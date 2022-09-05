---
title: 14-transformers
date: 2021-04-26 22:16:00
categories:
- Class
- CS224N
tags:
---

# Transformers and Self-Attention

序列化的模型类似于RNN，存在几个问题：

- Sequential computation的计算限制了并行计算
- 没有对于short和long dependencies的显式建模
- 我们希望能够建模层级

<!--more-->

对于迁移不变性的解释。

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210329210505679.png)

## 