---
title: IE-Collection
published: true
date: 2023-05-25 22:55:52
categories:
- Paper
- IE
tags:
- IE
---

信息抽取论文调研。

<!--more-->

# IE Papers

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

## TemplateNER

Template-Based Named Entity Recognition Using BART

ACL 2021 Findings, [代码](https://github.com/Nealcly/templateNER)。

> There is a recent interest in investigating fewshot NER, where the low-resource target domain has different label sets compared with a resource-rich source domain. Existing methods use a similarity-based metric. However, they cannot make full use of knowledge transfer in NER model parameters. To address the issue, **we propose a template-based method for NER, treating NER as a language model ranking problem in a sequence-to-sequence framework, ** where original sentences and statement templates filled by candidate named entity span are regarded as the source sequence and the target sequence, respectively. For inference, the model is required to classify each candidate span based on the corresponding template scores. Our experiments demonstrate that the proposed method achieves 92.55% F1 score on the CoNLL03 (rich-resource task), and significantly better than fine-tuning BERT 10.88%, 15.34%, and 11.73% F1 score on the MIT Movie, the MIT Restaurant, and the ATIS (low-resource task), respectively.

首个把template-based的方法应用到NER的序列标注任务。作者提出这种方法一开始的出发点是期望从模型结构的角度能够更好的解决低资源NER任务。之前的利用CRF或者一个线性层的序列标注方法严格的限制了能够预测和匹配的标签范围，不同数据集下的标签不同，必须重新训练新的预测层。

作者提出的方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630102935108.png"   style="zoom:50%;" />

整体上是一个编码器+解码器的结构，编码器把整个句子进行编码，随后编码的embedding输入到解码器，和对应的查询prompt的编码进行对比计算。

核心是查询时候的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630103433819.png"   style="zoom:50%;" />

作者采用效果最好的第一种prompt。除此之前还有对应非实体的prompt template：*<candidate_span > is not a named entity*。

在推理的时候，对于长度为$n$的句子，可能的span数量是$n(n-1)$，为了效率，作者设置一个span的最大长度为$8$。通过穷举span和对应的不同entity type，计算概率，然后排序。计算某个填充后的prompt是否成立的概率为：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630103737930.png"   style="zoom:50%;" />

实现的时候基于BART：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630104145213.png"   style="zoom:50%;" />

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630104223459.png"   style="zoom:40%;" />

## UIE

Uniﬁed Structure Generation for Universal Information Extraction

中国信息处理实验室，百度，ACL 2022, [代码](https://universal-ie.github.io/)。

> Information extraction suffers from its varying targets, heterogeneous structures, and demandspeciﬁc schemas. In this paper, **we propose a uniﬁed text-to-structure generation framework, namely UIE, which can universally model different IE tasks, adaptively generate targeted structures**, and collaboratively learn general IE abilities from different knowledge sources. Specifically, UIE uniformly encodes different extraction structures via a structured extraction language, adaptively generates target extractions via a schema-based prompt mechanism – structural schema instructor, and captures the common IE abilities via a large-scale pretrained text-to-structure model. Experiments show that UIE achieved the state-of-the-art performance on 4 IE tasks, 13 datasets, and on all supervised, low-resource, and few-shot settings for a wide range of entity, relation, event and sentiment extraction tasks and their uniﬁcation. These results verified the effectiveness, universality, and transferability of UIE.

UIE的模型图：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630104511551.png"   style="zoom:50%;" />

UIE从两个角度统一IE任务：

1. 结构化信息的格式SEL：包括三个描述元素，Spot Name用来描述span的类型（Entity/Event Type）；Asso Name用来描述和上级Spot关联的类型（Relation/Role Type）；Info Span对应的在原始text中的文本（Entity mention/Argument mention/Trigger word）；

   <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630104627488.png"   style="zoom:50%;" />

2. 统一的抽取prompt：指定要抽取的信息，通过加入特殊符号`[spot]+spot name`，`[asso]+Asso Name`和`[text]+source text`指定要抽取的信息。可以参考上面的方法图示例。

UIE的训练包括两个阶段，一个是在大规模的抽取语料中进行预训练（这一部分参考原始论文）；另一个是针对特定的任务进行快速的微调：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630105501812.png"   style="zoom:40%;" />

在微调阶段，有个需要特别注意的细节是，为了防止可能的exposure bias，作者引入了Rejection Mechanism。也就是说随机的插入负样例：（个人的理解就是让模型不要倾向于认为prompt中指定的要抽取的结构化信息是全部存在的，偏好给每一个要抽取的类型信息都要强制找到对应的span，而是能够学会判断各种结构化信息是否存在）

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630105730830.png"   style="zoom:40%;" />

这一点在实验部分有非常大的影响，能够有12-13点左右的变化。

UIE基于T5，也是编码器+解码器的结构：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630110116780.png"   style="zoom:30%;" />

从实验结果来看，即使是不经过微调，直接在不同数据集上进行推理时的信息抽取（对应表格中的SEL列），也能够达到比较好的效果。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630110209780.png"  style="zoom:50%;" />

## SuRE

Summarization as Indirect Supervision for Relation Extraction

EMNLP 2022，[代码](https://github.com/luka-group/SuRE)。

> Relation extraction (RE) models have been challenged by their reliance on training data with expensive annotations. Considering that summarization tasks aim at acquiring concise expressions of synoptical information from the longer context, these tasks naturally align with the objective of RE, i.e., extracting a kind of synoptical information that describes the relation of entity mentions. **We present SuRE, which converts RE into a summarization formulation.** SuRE leads to more precise and resource-efficient RE based on indirect supervision from summarization tasks. To achieve this goal, we develop sentence and relation conversion techniques that essentially bridge the formulation of summarization and RE tasks. We also incorporate constraint decoding techniques with Trie scoring to further enhance summarization-based RE with robust inference. Experiments on three RE datasets demonstrate the effectiveness of SuRE in both full-dataset and low-resource settings, showing that summarization is a promising source of indirect supervision signals to improve RE models.

作者认为如果直接把RE任务看做是多分类任务的话有两个缺点：

- 缺少对实体之间关系语义的理解。因为关系被转化为了logits，仅仅是个分类的匹配标签，实际的关系语义信息没有被学习到。
- 非常依赖于足够的RE标注数据来提供direct supervision。而在少资源的情况下，效果下降非常严重。

因此作者将RE任务和Summarization task关联起来，引入了Summarization相关的和RE任务不是直接关联的监督信号。

方法图：

![image-20230630164230877](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630164230877.png)

首先是如何处理输入的句子，能够进一步插入额外的实体信息，作者试验了两种方法：

1. Entity typed marker. 作者试了几种不同的标记方法，最终采用了下面的方法：

   <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630164850191.png"  style="zoom:40%;" />

2. Entity information verbalization. 直接在句子前面加入实体信息的描述：

   <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630165053978.png"   style="zoom:40%;" />

各种尝试的变化：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630165132956.png"   style="zoom:50%;" />

从实验效果来看，加入实体描述信息是有用的：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630165226787.png"   style="zoom:50%;" />

接下来的问题是怎么样描述relation，作者在前人的工作基础上，进行了微小的改动，获得了自己的模板：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630165334851.png"   style="zoom:50%;" />

作者在实现的时候，基于BART和Pegasus这两个已经在summarization相关数据上pre-train过的模型。然后在对应的RE数据集上按照sequence-to-sequence方法进行微调。

在推理阶段，作者将所有的relation template聚合在一起构造了一个Trie tree。通过在Trie树上计算每一个路径，得到最后的关系预测概率。编码器输入原始的句子，解码器不断输入对应的路径，具体可以参考方法图。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230630170430071.png"   style="zoom:50%;" />

# Cross-domain IE

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

> Few-shot Named Entity Recognition (NER) is imperative for entity tagging in limited resource domains and thus received proper attention in recent years. Existing approaches for few-shot NER are evaluated mainly under in-domain settings. In contrast, little is known about how these inherently faithful models perform in cross-domain NER using a few labeled in-domain examples. **This paper proposes a two-step rationale-centric data augmentation method to improve the model’s generalization ability.** Results on several datasets show that our model-agnostic method significantly improves the performance of crossdomain NER tasks compared to previous state-of-the-art methods, including the data augmentation and prompt-tuning methods. Our codes are available at https://github.com/lifan-yuan/FactMix.

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

