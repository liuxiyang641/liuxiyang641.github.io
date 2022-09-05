---
title: Independence Survey
notshow: false
date: 2021-07-07 10:39:25
categories:
- Note
tags:
---

# Independence Modeling Method

对目前接触的集中能够约束差异性/独立性的方法做个简单汇总。包括

- 互信息Mutual information.
- 距离相关性Distance correlation.
- Hilbert-Schmidt Independence Criterion (HSIC)

<!--more-->

Mutual information和Distance correlation.在推荐模型KGIN（Learning Intents behind Interactions with Knowledge Graph for Recommendation）中得到了使用，用于约束几个特定embedding之间的独立性。

Mutual information：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210709151338821.png" style="zoom:50%;" />

其中的函数$s()$是计算相似度的函数，在文章中使用了cosine similarity。

Distance correlation：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210709151449098.png" style="zoom:50%;" />

其中的$dCor()$是距离相关性distance correlation：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210709151520931.png" style="zoom:50%;" />

Hilbert-Schmidt Independence Criterion (HSIC)方法在AM-GCN中应用，

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20210709151711018.png" style="zoom:50%;" />

对于上述的集中方法还没有深入理解，先做一下记录。
