---
title: cross-domain-IE
published: true
date: 2023-05-25 22:55:52
categories:
- Paper
- IE
tags:
- IE
- Cross-domain
---

# Cross-domain IE

跨域信息抽取论文调研。

<!--more-->

## CP-NER

One Model for All Domains: Collaborative Domain-Preﬁx Tuning for Cross-Domain NER

IJCAI 2023，浙大ZJUNLP，[代码](https://github.com/zjunlp/DeepKE/tree/main/example/ner/cross)。

> Cross-domain NER is a challenging task to address the low-resource problem in practical scenarios. Previous typical solutions mainly obtain a NER model by pre-trained language models (PLMs) with data from a rich-resource domain and adapt it to the target domain. Owing to the mismatch issue among entity types in different domains, previous approaches normally tune all parameters of PLMs, ending up with an entirely new NER model for each domain. Moreover, current models only focus on leveraging knowledge in one general source domain while failing to successfully transfer knowledge from multiple sources to the target. To address these issues, we introduce Collaborative Domain-Preﬁx Tuning for cross-domain NER (Cp -NER) based on text-to-text generative PLMs. Speciﬁcally, we present textto-text generation grounding domain-related instructors to transfer knowledge to new domain NER tasks without structural modiﬁcations. We utilize frozen PLMs and conduct collaborative domain-preﬁx tuning to stimulate the potential of PLMs to handle NER tasks across various domains. Experimental results on the Cross-NER benchmark show that the proposed approach has flexible transfer ability and performs better on both one-source and multiple-source cross-domain NER tasks.

作者期望解决下面三个问题：

- 之前的跨域IE方法依赖于为不同的domain设计不同的architecture
- 大多数现有的方法需要tuning PLM的所有参数，计算代价较高
- 之前的方法只考虑单源的跨域IE

作者提出的方法

- 使用prefix-tuning，这样就不需要为不同的domain设计不同的architecture

- 使用frozen PLM parameters，这样就不需要更新模型的所有参数
- 考虑多源的跨域IE

下面是方法图：

![image-20230525230232411](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230525230232411.png)

作者方法基于T5-base（220M参数），重点在于prefix embedding的学习，prefix embedding会加入到输入的sentence tokens的左边（简单看了下代码，应该是长度为10）。按照prefix-tuning的原论文，prefix-tuning可以看做是continuous instruction，是用来学习和task相关的context信息，进而激发LM去执行特定的任务，特别是对于那些没有表现出利用人类自然语言描述的instruction去执行特定任务的小模型（BERT、GPT-1,2等）。

prefix embedding是要从每个领域都进行训练学习，prefix embedding在T5的每一层都有独立的表示。学习的方法就是在加入prefix embedding之后，再按照T5的训练loss，在domain data下进行训练，过程中保持T5原来的所有参数不变。（对应图中的domain-specific warm-up过程）

之后，作者因为要考虑多源IE，因此通过source domain的实体label和target domain的实体label计算相似度；domain prefix embedding之间也计算相似度这样来评估不同来源对于target domain的影响大小。（对应图中的dual-query domain selector过程）

最后，在前一步计算得到的相似度经过softmax之后和多源prefix embedding进行加权求和，再加入到target domain的prefix embedding上。作者最后还进行了再一次的微调，训练最终学习到的target domain prefix embedding。

实验结果，下面是单源IE，以CONLL2003作为source domain，CrossNER数据集下的多个子集作为target domain：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230525231607634.png"   style="zoom:40%;" />

下面是多源的跨域抽取，以CoNLL 2003, Politics, Literature, and Music作为source domain，使用Mit-Movie, AI, Science, and Mit-Restaurant作为target domain：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230525231642332.png"   style="zoom:40%;" />

## Fact-Mix

FactMix: Using a Few Labeled In-domain Examples to Generalize to Cross-domain Named Entity Recognition

COLING 2022，西湖大学 ，[代码](https://github. com/lifan-yuan/FactMix)。

> Few-shot Named Entity Recognition (NER) is imperative for entity tagging in limited resource domains and thus received proper attention in recent years. Existing approaches for few-shot NER are evaluated mainly under in-domain settings. In contrast, little is known about how these inherently faithful models perform in cross-domain NER using a few labeled in-domain examples. **This paper proposes a two-step rationale-centric data augmentation method to improve the model’s generalization ability.** Results on several datasets show that our model-agnostic method significantly improves the performance of crossdomain NER tasks compared to previous state-of-the-art methods, including the data augmentation and prompt-tuning methods. Our codes are available at https://github. com/lifan-yuan/FactMix.

作者主要从数据增强的角度解决跨域NER问题。作者认为跨域的NER任务要考虑两个核心问题：

- NER任务作为序列标注任务，它的label之间是相互依赖的，而不是相互独立的。不同领域这种label依赖不一样。it is essential to understand dependencies within the labels instead of classifying each token independently.
- 不同domain的文本中的non-entity tokens的语义是不一致的，这种不一致可能增大NER模型进行跨域NER的困难程度。non-entity tokens in NER do not hold unified semantic meanings, but they could become noisy when combined with entity tokens in the training set.

因此，作者认为NER模型学习到的non-entity token和要预测的label之间隐式联系可能影响跨域性能。比如在医学domain上的句子'Jane monitored the patient’s heart rate'，Jane是一个person，在医学domain上训练好的一个NER model可能学习到Jane和monitored之间的潜在关联。但是如果迁移到关于movie review的跨域数据集上，Jane和monitor之间的在医疗领域的潜在关联就不再合适了。

因此，作者提出了一种新的数据增强策略Context-level semi-fact generations：

- 随机使用MLM的[MASK] token代替source domain文本中的某个non-entity token，选择预测时概率最大的词进行替换。这样就引入了out-of-domain的context信息（被预训练model在预训练阶段学习到的信息）
- 为了避免替换后的词引起entity label标注的影响，作者只保留那些能够被NER模型正确预测所有token NER tag的替换后的样例

这种数据增强策略Context-level semi-fact generations和之前研究者提出的Entity-level semi-fact generations结合起来：

![image-20230530110812914](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230530110812914.png)

实验以CONLL2003作为source domain，CrossNER数据集下的多个子集作为target domain。数据增强策略应用到了fine-tuning based和prompt-tuning based两种方法上进行验证，具体参考论文。

## PromptNER

PromptNER: Prompt Locating and Typing for Named Entity Recognition

ACL 2023，浙大，[代码](https://github.com/ tricktreat/PromptNER)。

> Prompt learning is a new paradigm for utilizing pre-trained language models and has achieved great success in many tasks. To adopt prompt learning in the NER task, two kinds of methods have been explored from a pair of symmetric perspectives, populating the template by enumerating spans to predict their entity types or constructing type-specific prompts to locate entities. However, these methods not only require a multi-round prompting manner with a high time overhead and computational cost, but also require elaborate prompt templates, that are difficult to apply in practical scenarios. **In this paper, we unify entity locating and entity typing into prompt learning, and design a dual-slot multi-prompt template with the position slot and type slot to prompt locating and typing respectively.** Multiple prompts can be input to the model simultaneously, and then the model extracts all entities by parallel predictions on the slots. To assign labels for the slots during training, we design a dynamic template filling mechanism that uses the extended bipartite graph matching between prompts and the ground-truth entities. We conduct experiments in various settings, including resource-rich flat and nested NER datasets and low-resource indomain and cross-domain datasets. Experimental results show that the proposed model achieves a significant performance improvement, especially in the cross-domain few-shot setting, which outperforms the state-of-the-art model by +7.7% on average.

这篇paper准确的讲不应该出现在cross-domain IE方向内。它只是在实验部分做了跨域IE的实验。

主要创新点在于作者期望能够一次性的用prompt的方法抽取出句子中包括的所有实体span和实体type。这样能够提高模型的速度。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612162253219.png"   style="zoom:40%;" />

作者提出的方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612162443446.png" />

首先是提前构造$M$个prompt，实验中使用了$M=50$。所有的prompt里包括了一个position slot和type slot，最终和原始的句子一起输入到BERT中进行编码：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612162601219.png"   style="zoom:40%;" />

经过编码之后，作者还增加了一个不同prompt之间的interaction layer。即包括prompt之间的self-attention between slots with the same type，也包括从sentence to prompt slots的attention：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612162917336.png"   style="zoom:40%;" />

在进行解码阶段，不同prompt中的type slot的embedding直接拿出来进行softmax分类：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612163128048.png"   style="zoom:40%;" />

而position slot需要和不同位置的word embedding相加，然后判断是不是某个实体的左边界：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612163112798.png"   style="zoom:40%;" />

同样的可以获得右边界的预测值$\mathbf{p}^r_{ij}$。

现在我们要进行优化，那么就需要知道一个prompt输出的预测结果，真实期望是哪个entity，要不然无法优化。我们有很多个prompt，应该想一种办法对这些prompt的结果进行整合，最后和实际的entity对应上。

作者这里使用了一个bipartite graph matching的思路。就是期望那个最好的匹配结果，能够使prompts的预测结果和entities之间的cost最小：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612163636157.png"   style="zoom:40%;" />

上面$i$是指$i$-th entity，$\sigma(i)$是指$\sigma(i)$-th prompt。$Cost_{match}$的定义如下，实际上就是让一个entity去和最倾向于预测它的prompt匹配，方便最后计算loss：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612163724720.png" style="zoom:40%;" />

作者这里还提出了让一个entity能够和多个prompt匹配，具体做法就是重复多次相同的entity。作者使用Hungarian algorithm来求解最佳的匹配，然后计算loss：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612164233873.png"   style="zoom:40%;" />

实验：

![image-20230612164255376](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612164255376.png)

![image-20230612164314322](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230612164314322.png)
