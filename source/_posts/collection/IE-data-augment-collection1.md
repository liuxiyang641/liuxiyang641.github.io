---
title: IE-data-augment-collection1
published: true
date: 2023-09-17 10:42:34
categories:
- Paper
- IE
- Data Augment
tags:
- IE
- Data Augment
- Collection
---

基于数据增强策略的信息抽取论文合集。

<!--more-->

# Data Augment for IE papers



## DAGA

DAGA: Data Augmentation with a Generation Approach for Low-resource Tagging Tasks

南洋理工与阿里达摩，EMNLP 2020，[代码](https://github.com/ntunlp/daga)。

> Data augmentation techniques have been widely used to improve machine learning performance as they enhance the generalization capability of models. In this work, to generate high quality synthetic data for low-resource tagging tasks, **we propose a novel augmentation method with language models trained on the linearized labeled sentences. Our method is applicable to both supervised and semi-supervised settings.** For the supervised settings, we conduct extensive experiments on named entity recognition (NER), part of speech (POS) tagging and end-to-end target based sentiment analysis (E2E-TBSA) tasks. For the semi-supervised settings, we evaluate our method on the NER task under the conditions of given unlabeled data only and unlabeled data plus a knowledge base. The results show that our method can consistently outperform the baselines, particularly when the given gold training data are less.

作者声称是首个在序列标注task上，引入LM做数据增强的文章。

数据增强是用来人造数据的一种在各个领域都被广泛应用的方法。NLP上的数据增强有它自己独特的特征：在image上简单的修改通常不会改变image本身的信息；但是在natural language上删除或替换一个词就可能完全改变整个sentence的意思。

而一般的NLP 数据增强方法包括synonym replacement, random deletion/swap/insertion, generation with VAE or pre-trained language models、back translation、systematically reordering the dependents of some nodes in gold data、leveraging knowledge base for question generation等等。

和上面的NLP任务相比，类似NER这类的token-level的sequence tagging任务对数据增强时引入的噪音更加敏感。序列标注有的3种尝试（2020年前）：

- Annotating unlabeled data with a weak tagger [*Automated phrase mining from massive text corpora. 2018*] 使用已有的标注工具直接进行标注，需要标注工具已经提前具备了相应的domain knowledge，否则面临domain-shift problem [*Multimix: A robust data augmentation framework for cross-lingual nlp. 2020*]
- leveraging aligned bilingual corpora to induce annotation [*Inducing multilingual text analysis tools via robust projection across aligned corpora. 2001*] 要求有额外的外语语料，很多情况下不实际
- synonym replacement [*Biomedical named entity recognition via reference-set augmented bootstrapping. 2019*] 需要WordNet这类外部知识和人工设计的规则，难以覆盖所有的低资源场景

因此，作者提出使用生成式的数据增强方法。作者首先训练一个LM学会现有gold data中语言的特征：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912161946176.png"   style="zoom:30%;" />

单层的LSTM作为语言模型，使用一般的单向language objectives进行优化。作者通过sentence linearization把所有的序列标注sentence都转换为带有tag的句子（NER任务中忽略tag $O$）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912162201735.png"   style="zoom:30%;" />

将tag放在对应的word前面，作者发现这样比tag在word后面效果好。推测原因是这样子可能更加符合一般的语言中形容词-名词的pattern（Modifier-Noun pattern）。

在生成的时候，输入是[BOS]，让LSTM LM直接输出各种不同的句子。对于输出的句子进行后处理，比如删除没有tag的句子、删除有错误tag的情况等。

除了上面直接在gold data上让LM学习特征外，作者还提出了conditional generation method让LM能够利用unlabeled data or knowledge bases。从外部的数据源中获取更多的knowledge：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912162711987.png"  style="zoom:50%;" />

conditional generation本质就是在sentence之前添加condition tags：$\{ [labeled], [unlabeled], [KB] \}$。

在实验中，作者的NER使用BiLSTM-CRF模型在gold data和生成的data上进行训练，然后评估。作者使用了过采样gold data的策略，采样1个generated data，过采样4个gold data。

在CoNLL2002/2003 NER数据集的多个语言子集（English, German, Dutch and Spanish）上进行验证：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912163003964.png"  style="zoom:30%;" />

实验中有一个可以注意的是作者如何评估生成数据的多样性，一个是用entity出现的周围token作为上下文；计算unique上下文token数量；一个是统计unique entity的数量：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912163929108.png"   style="zoom:25%;" /> <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912163954188.png" style="zoom:25%;" />

## MELM

MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER

阿里达摩与南洋理工，ACL 2022，[代码](https://github.com/RandyZhouRan/MELM/)。

> Data augmentation is an effective solution to data scarcity in low-resource scenarios. However, when applied to token-level tasks such as NER, **data augmentation methods often suffer from token-label misalignment, which leads to unsatsifactory performance.** In this work, **we propose Masked Entity Language Modeling (MELM) as a novel data augmentation framework for low-resource NER.** To alleviate the token-label misalignment issue, we explicitly inject NER labels into sentence context, and thus the fine-tuned MELM is able to predict masked entity tokens by explicitly conditioning on their labels. Thereby, MELM generates high-quality augmented data with novel entities, which provides rich entity regularity knowledge and boosts NER performance. When training data from multiple languages are available, we also integrate MELM with codemixing for further improvement. We demonstrate the effectiveness of MELM on monolingual, cross-lingual and multilingual NER across various low-resource levels. Experimental results show that our MELM presents substantial improvement over the baseline methods.

前人工作指出，增强上下文带来的提升比较少[*A rigorous study on named entity recognition: Can fine-tuning pretrained model lead to the promised land? EMNLP 2020*]。作者也发现，增强新entity多样性带来的效果要大于增强上下文patterns：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912200034085.png"  style="zoom:30%;" />

作者使用masked LM来给low-resource NER任务做数据增强。作者只会根据一定的概率mask entity的token。然后在mask data上fine-tuning pretrained MLM，让MLM学会根据context预测entity。

如果只是mask entity，然后让MLM预测，可能能够符合context，但是不一定符合原来的entity label。为了让生成的entity和原来的entity有相同的label，作者在原来的句子中插入entity type marker：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912200407792.png"   style="zoom:50%;" />

在进行数据生成的时候，输入masked sentence，为了增强生成数据的多样性。没有使用greedy decoding策略，而是在top-K的候选项上进行随机选择。

同时在生成的时候，采用了新的mask策略。每一次生成都有不同的mask阈值，这样进一步增大了mask结果的差异。

生成的数据需要经过处理以减低噪音，作者用一个训练好的NER模型，去处理增强的句子；只有NER model的标注和生成句子原来的entity label标注一致，才会被保留。

最后，作者在这篇论文中着重考虑多语言场景，引入code-mixing技术。随机从某个其它语言中，选择有相同label的entity作为候选项，之后选择在embedding space上余弦相似度的外语entity替换原来language entity（使用MUSE作为编码方法）。并且在替换后的entity前加入language tag表示替换后的entity原来的语言是什么。

增强的数据比例是3倍。作者实现中使用的LM是XLM-RoBERTa-base，使用的NER model是XLM-RoBERTa-Large+CRF。

在CoNLL数据集的不同语言子集上的实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230912201243434.png"   style="zoom:30%;" />

## GPDA

Improving Low-resource Named Entity Recognition with Graph Propagated Data Augmentation

ACL 2023 short paper，上海科技与阿里达摩，[代码](https://github.com/modelscope/AdaSeq/tree/master/examples/GPDA)。

> Data augmentation is an effective solution to improve model performance and robustness for low-resource named entity recognition (NER). **However, synthetic data often suffer from poor diversity, which leads to performance limitations. In this paper, we propose a novel Graph Propagated Data Augmentation (GPDA) framework for Named Entity Recognition (NER), leveraging graph propagation to build relationships between labeled data and unlabeled natural texts.** By projecting the annotations from the labeled text to the unlabeled text, the unlabeled texts are partially labeled, which has more diversity rather than synthetic annotated data. To strengthen the propagation precision, a simple search engine built on Wikipedia is utilized to fetch related texts of labeled data and to propagate the entity labels to them in the light of the anchor links. Besides, we construct and perform experiments on a real-world lowresource dataset of the E-commerce domain, which will be publicly available to facilitate the low-resource NER research. Experimental results show that GPDA presents substantial improvements over previous data augmentation methods on multiple low-resource NER datasets.

data augmentation对于sentence-level NLP task两大思路：

1. One is manipulating a few words in the original sentence, which can be based on synonym replacement (Zhang et al., 2015; Kobayashi, 2018; Wu et al., 2019; Wei and Zou, 2019), random insertion or deletion (Wei and Zou, 2019), random swap (¸Sahin and Steedman, 2018; Wei and Zou, 2019; Min et al., 2020). 修改原有句子的部分表述，获得新data。
2. The other is generating the whole sentence with the help of back-translation (Yu et al., 2018; Dong et al., 2017; Iyyer et al., 2018), sequence to sequence models (Kurata et al., 2016; Hou et al., 2018) or pre-trained language models (Kumar et al., 2020). 构造完全新的data。



作者认为之前的 Data Augmentation会使用人造的数据，这可能inevitably introduces incoherence, semantic errors and lacking in diversity. 

因此作者提出要直接使用已有的natural text作为辅助数据增强的来源。

方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230908201331433.png"  style="zoom:30%;" />

步骤：

- 从外部源如Wikipedia corpus中，通过BM25 sparse retrieval或者L2 dense retrieval的方法检索和句子相似的sentence
- 然后进行label propagation，在Wikipedia中带有链接的anchor text如果和有label的entity是完全匹配的，就赋值给anchor text对应的label。（但是完全一样text的entity就是相同的entity吗？）使用这样的新标注的数据和原有的有标注数据训练一个NER model
- 使用训练好的NER model，重新标注一次外部的text，然后使用重新标注后的数据和原有的有标注数据训练一个更好的NER model（Explored Entity Annotations，EEA）

实验：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230908201751825.png"   style="zoom:40%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230908201830377.png"   style="zoom:40%;" />

## GDA

GDA: Generative Data Augmentation Techniques for Relation Extraction Tasks

ACL 2023 Findings，清华与浙大，[代码](https://github.com/THU-BPM/GDA)。

> Relation extraction (RE) tasks show promising performance in extracting relations from two entities mentioned in sentences, given sufficient annotations available during training. Such annotations would be labor-intensive to obtain in practice. Existing work adopts data augmentation techniques to generate pseudo-annotated sentences beyond limited annotations. **These techniques neither preserve the semantic consistency of the original sentences when rule-based augmentations are adopted, nor preserve the syntax structure of sentences when expressing relations using seq2seq models, resulting in less diverse augmentations.** In this work, we propose a dedicated augmentation technique for relational texts, named GDA, which uses two complementary modules to preserve both semantic consistency and syntax structures. We adopt a generative formulation and design a multi-tasking solution to achieve synergies. Furthermore, GDA adopts entity hints as the prior knowledge of the generative model to augment diverse sentences. Experimental results in three datasets under a low-resource setting showed that GDA could bring 2.0% F1 improvements compared with no augmentation technique. Source code and data are available.

之前方法存在的问题：

- 之前的rule-based techniques的数据增强方法不能够保证构造出来的句子和原来的句子是语义一致的，并且由于忽略了语法结构还有可能扭曲原来的语义
- model-based techniques能够保持语义一致性 [*Data augmentation in natural language processing: a novel text generation approach for long and short text classifiers. 2022*]，但是不能够生成多样性的表达。the model generates less diverse sentences – it includes similar entities and identical relational expressions under the same relation.

生成的数据既需要多样性，又需要和原来句子的语义一致性。

作者基于多任务学习提出的数据增强方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230909170611490.png"  style="zoom:30%;" />

基于BART或T5这样的encoder+decoder结构，有两个decoder：

- Original sentence restructuring. 左侧的decoder，重建原来的sentence，让模型学会产生和原来句子语义一致的句子：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230909170816949.png"   style="zoom:30%;" />

- Original sentence pattern approximation. 右侧的decoder用来生成新的sentence。由于归纳偏执，seq2seq decoder总是会倾向高频率出现的pattern，就失去生成数据的多样性。因此作者限制生成的新句子的pattern和原来的句子一致。具体做法是使用两个entity之间的语法路径作为relation pattern，生成句子的relation pattern和原来句子的relation pattern要接近：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230909171308055.png"   style="zoom:30%;" />

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230917152104901.png"   style="zoom:40%;" />
  
  另外，为了进一步控制输出句子。作者会从数据集中同属于一个relation的样例中选择entity，输入解码器，让模型输出带有entity的句子。

训练的时候，先训练编码器和restructuring decoder；然后使用restructuring decoder的参数初始化pattern approximation decoder参数，和编码器一起训练；pattern approximation decoder参数继续用来初始化restructuring decoder。

两个decoder分别独立迭代优化；encoder一直进行优化。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230909171514198.png"   style="zoom:40%;" />

可以看到，利用作者的数据增强方法生成的数据来训练，能够有效提升Base model的效果。



## ENTDA

Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks

ACL 2023 Findings，清华与阿里达摩，{% post_link nlp/ENTDA  [详细博客] %}

> Data augmentation techniques have been used to alleviate the problem of scarce labeled data in various NER tasks (flat, nested, and discontinuous NER tasks). **Existing augmentation techniques either manipulate the words in the original text that break the semantic coherence of the text, or exploit generative models that ignore preserving entities in the original text, which impedes the use of augmentation techniques on nested and discontinuous NER tasks.** In this work, we propose a novel Entity-toText based data augmentation technique named ENTDA to add, delete, replace or swap entities in the entity list of the original texts, and adopt these augmented entity lists to generate semantically coherent and entity preserving texts for various NER tasks. Furthermore, we introduce a diversity beam search to increase the diversity during the text generation process. Experiments on thirteen NER datasets across three tasks (flat, nested, and discontinuous NER tasks) and two settings (full data and low resource settings) show that ENTDA could bring more performance improvements compared to the baseline augmentation techniques.

基于entity list生成对应的新data：

作者提出的方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230911191701967.png"   style="zoom:50%;" />

作者的生成data思路是根据entity list，让language model来直接生成相应的句子。

然后，让language model基于entity list生成对应的句子。为了提升生成句子的多样性diversity，作者提出了一种diversity beam search decoding策略：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230911191955856.png"  style="zoom:30%;" />

作者在flat, nested, and discontinuous NER tasks都进行了实验。在full data的情况下，提升不太大，但是在低资源的情况下提升很多。we randomly choose 10% training data from CoNLL2003/ACE2005/CADEC：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230911193116491.png"  style="zoom:30%;" />

低资源的情况下，效果提升明显，有$2$%的提升幅度。

在真实的低资源NER数据集CrossNER的表现：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230911193212164.png"   style="zoom:40%;" />

同样提升比较明显。

## ACLM

ACLM: A Selective-Denoising based Generative Data Augmentation Approach for Low-Resource Complex NER

ACL 2023，University of Maryland，[代码](https://github.com/Sreyan88/ACLM)。

> Complex Named Entity Recognition (NER) is the task of detecting linguistically complex named entities in low-context text. In this paper, we present ACLM (Attention-map aware keyword selection for Conditional Language Model fine-tuning), a novel data augmentation approach, based on conditional generation, to address the data scarcity problem in low-resource complex NER. **ACLM alleviates the context-entity mismatch issue, a problem existing NER data augmentation techniques suffer from and often generates incoherent augmentations by placing complex named entities in the wrong context.** ACLM builds on BART and is optimized on a novel text reconstruction or denoising task - we use selective masking (aided by attention maps) to retain the named entities and certain keywords in the input sentence that provide contextually relevant additional knowledge or hints about the named entities. Compared with other data augmentation strategies, ACLM can generate more diverse and coherent augmentations preserving the true word sense of complex entities in the sentence. We demonstrate the effectiveness of ACLM both qualitatively and quantitatively on monolingual, crosslingual, and multilingual complex NER across various low-resource settings. ACLM outperforms all our neural baselines by a significant margin (1%-36%). In addition, we demonstrate the application of ACLM to other domains that suffer from data scarcity (e.g., biomedical). In practice, ACLM generates more effective and factual augmentations for these domains than prior methods.

作者主要希望通过数据增强来解决complex NER任务：

> complex NER benchmarks like MultiCoNER (Malmasi et al., 2022) present several contemporary challenges in NER, including short low-context texts with emerging and semantically ambiguous complex entities (e.g., movie names in online comments) that reduce the performance of SOTA methods previously evaluated only on the existing NER benchmark datasets.

作者认为之前SOTA的数据增强方法效果不好，因为对于complex NER任务来说，特定的entity要依赖于特定的context：

> We first argue that certain types of complex NEs follow specific linguistic patterns and appear only in specific contexts (examples in Appendix 4), and augmentations that do not follow these patterns impede a NER model from learning such patterns effectively.

方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230917235747765.png"   style="zoom:50%;" />

作者分为四步来获取corrupted sentence（paper里叫做template）：

1. Keyword Selection：使用attention map寻找对entity最有意义的context tokens，然后将top-$p$%的context tokens用看做是*keywords*。具体来说，使用XLM-RoBERTa-large进行在训练集上进行训练，然后使用它最后4层所有Transformer attention head的注意力权重作为选择依据。

   - 低资源的情况下，attention map可能是比较noisy的，所有head相加比较robust

   - BERT的低层更加关注其它token，而BERT的高层更加专注某个token

   - 作者处理的entity可能有多个span或者1个span。对于1个span，每个token的attention score相加。对于有多个span的entity，每个span分别计算attention score获取重要tokens

2. Selective Masking：对于非entity和非重要keywords的其它tokens，用$[MASK]$ token进行替换。mask后的句子作为template。

3. Labeled Sequence Linearization：模仿MELM在entity token前后插入`<tag>`。

4. Dynamic Masking：动态的选择一部分keywords的token也进行替换，增加多样性

根据上面获取的corrupted sentence，微调mBart-50-large，让其重建原来的句子。

在进行数据生成的时候，对于每个sentence，创建$R$个corrupt text，生成$R$个augmented training samples（实现中$R=5$）。

为了进一步增加多样性，作者在数据生成阶段，提出了一个mixer方法，根据一定的概率选择另外一个语义相似的句子生成的template进行拼接，然后生成新的句子：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230918000541961.png"  style="zoom:40%;" />

实现中基于multi-lingual Sentence-BERT的embedding计算不同句子之间的余弦相似度。

最后对生成的数据进行后处理，对与和原sentence非常相似的生成sentence等数据，进行移除。

作者在MultiCoNER上的实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230918000749168.png"   style="zoom:40%;" />

在其它NER数据集（CoNLL 2003 (Tjong Kim Sang and De Meulder, 2003) (news), BC2GM (Smith et al., 2008) (bio-medical), NCBI Disease (Do˘gan et al., 2014) (bio-medical) and TDMSci (Hou et al., 2021) (science)）上的结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230918000837774.png"  style="zoom:40%;" />

这个实验解释了一个重要的结论，在CoNLL2003这种entity和明确的数据集上，LwTR（替换相同entity type的其它entity）这种rule-based的方法反而取得了最好的结果。

对于生成数据的定量评估：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230918001054785.png"   style="zoom:40%;" />

其中，Diversity-E指生成sentence中新出现的实体，Diversity-N指新出现的非entity的tokens，Diversity-L指新生成的句子长度与原来句子的比值。ACLM更擅长引入更多新的context  tokens。

## Paraphrase NER

When and how to paraphrase for named entity recognition?

ACL 2023，{% post_link nlp/when-how-paraphrase-NER  [详细博客] %}。

> While paraphrasing is a promising approach for data augmentation in classification tasks, its effect on named entity recognition (NER) is not investigated systematically due to the difficulty of **span-level label preservation**. In this paper, **we utilize simple strategies to annotate entity spans in generations and compare established and novel methods of paraphrasing in NLP such as back translation, specialized encoder-decoder models such as Pegasus, and GPT-3 variants for their effectiveness in improving downstream performance for NER** across different levels of gold annotations and paraphrasing strength on 5 datasets. We thoroughly explore the influence of paraphrasers, dynamics between paraphrasing strength and gold dataset size on the NER performance with visualizations and statistical testing. We find that the choice of the paraphraser greatly impacts NER performance, with one of the **larger GPT-3 variants exceedingly capable of generating high quality paraphrases, yielding statistically significant improvements in NER performance with increasing paraphrasing strength,** while other paraphrasers show more mixed results. Additionally, inline auto annotations generated by larger GPT-3 are strictly better than heuristic based annotations. We also find diminishing benefits of paraphrasing as gold annotations increase for most datasets. Furthermore, while most paraphrasers promote entity memorization in NER, the proposed GPT-3 configuration performs most favorably among the compared paraphrasers when tested on unseen entities, with memorization reducing further with paraphrasing strength. Finally, we explore mention replacement using GPT-3, which provides additional benefits over base paraphrasing for specific datasets.

作者选择了5个不同领域的NER数据集。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230916214229221.png"   style="zoom:40%;" />

作者先对比两个已有的Paraphrasers工具：

- 基于Back-translation（BT）：For our experiments we use pre-trained English-German and German-English models (∼738M parameters) available from Huggingface model hub via Tiedemann and Thottingal (2020) and the model architecture used is BART (Lewis et al., 2019).
- 基于PEGASUS：We use an off-the-shelf version of PEGASUS fine-tuned for paraphrasing released on Huggingface model hub. 3

然后，作者利用两个GPT-3模型：`text-ada-001` (∼350M parameters), and `text-davinci-002` (∼175B parameters)。使用的temperature为0.8。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230916214754741.png"   style="zoom:40%;" />

作者关注数据增强可能带来的一个问题Entity Memorization。即目前基于改写的数据增强方法，没有改变entity mention，生成的data中出现了entity的重复。因此作者想检查模型是不是直接记住了entity和它对应的label，而不是学会从feature推测label。

如果是记忆，那么model意味着模型走了捷径shortcut learning [*Shortcut learning in deep neural networks. Nature 2020*]，那么此时model应该无法准确处理没有见过的entity。

因此，作者又进行了在test set中，不同entity type里，没有在训练集里出现过的entity作为新的测试集unseen entity (UE) test sets。

为了缓解entity memorization问题，作者提出了一种解决方法Mention replacement（MR）。那就是不要重复entity mention，用GPT生成新的entity mention，然后去替换生成句子中的entity mention：

> In particular, for every entity mention in the gold set, we prompt GPT-3 DaVinci model to generate entity mentions that are similar to the gold entity mention, while also providing a phrase level definition of the entity type being replaced.

使用到的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230916223735026.png"  style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230916223753805.png"   style="zoom:50%;" />

作者选择了5个不同领域的NER数据集，微调distilbert-base-cased作为NER model。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230916214229221-20230917171137150.png"   style="zoom:40%;" />

## Data Generation for clinical NER and RE

Does Synthetic Data Generation of LLMs Help Clinical Text Mining?

arXiv 2023-04

> Recent advancements in large language models (LLMs) have led to the development of highly potent models like OpenAI’s ChatGPT. These models have exhibited exceptional performance in a variety of tasks, such as question answering, essay composition, and code generation. However, their effectiveness in the healthcare sector remains uncertain. **In this study, we seek to investigate the potential of LLMs to aid in clinical text mining by examining their ability to extract structured information from unstructured healthcare texts, with a focus on biological named entity recognition and relation extraction.** However, our preliminary results indicate that employing LLMs directly for these tasks resulted in poor performance and raised privacy concerns associated with uploading patients’ information to the LLM API. To overcome these limitations, we propose a new training paradigm that involves generating a vast quantity of high-quality synthetic data with labels utilizing LLMs and fine-tuning a local model for the downstream task. Our method has resulted in significant improvements in the performance of downstream tasks, improving the F1-score from 23.37% to 63.99% for the named entity recognition task and from 75.86% to 83.59% for the relation extraction task. Furthermore, **generating data using LLMs can significantly reduce the time and effort required for data collection and labeling, as well as mitigate data privacy concerns.** In summary, the proposed framework presents a promising solution to enhance the applicability of LLM models to clinical text mining.

作者先是尝试了ChatGPT在clinical NER和RE任务上，zero-shot ICL设置下和目前SOTA的差距：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925170021648-20230925171040987.png"   style="zoom:50%;" />

在clinical NER和RE上，作者发现效果并不好，这当然很正常，ChatGPT并不是专门为clinical domain训练的，而执行这一domain肯定需要大量的domain knowledge；同时直接调用LLM的API存在隐私泄露问题。因此作者尝试利用LLM去生成一系列的训练数据，而不是直接进行任务。用LLM生成数据去训练一个小模型，小模型可以直接本地部署，避免了隐私泄露问题。

作者用prompt engineering创造合适的prompt：

- 询问GPT “Provide five concise prompts or templates that can be used to generate data samples of [Task Descriptions].”
- 用每个prompt生成10个句子，然后人工检查下句子质量，选择效果最好的prompt
- 然后让GPT基于前面选择的最好的prompt，继续提供新的prompt。这一过程持续3遍

作者找到的最合适的prompt（没有demonstrations）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925170508864-20230925171041027.png"   style="zoom:50%;" />

NER任务是根据entity直接生成句子；RE任务是输入头尾实体，判断某个relation是否存在

可视化结果显示，不控制的情况下，GPT自己发挥生成的句子和原来的sentence肯定有分布上的差别：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925170635577.png"   style="zoom:50%;" />

## $\mbox{S}^2$ynRE

S2ynRE: Two-stage Self-training with Synthetic data for Low-resource Relation Extraction

中科大，ACL 2023，[代码](https: //github.com/BenfengXu/S2ynRE)。

> Current relation extraction methods suffer from the inadequacy of large-scale annotated data. While distant supervision alleviates the problem of data quantities, there still exists domain disparity in data qualities due to its reliance on domain-restrained knowledge bases. In this work, **we propose S2ynRE, a framework of two-stage Self-training with Synthetic data for Relation Extraction.** We ﬁrst leverage the capability of large language models to adapt to the target domain and automatically synthesize large quantities of coherent, realistic training data. We then propose an accompanied two-stage self-training algorithm that iteratively and alternately learns from synthetic and golden data together. We conduct comprehensive experiments and detailed ablations on popular relation extraction datasets to demonstrate the effectiveness of the proposed framework. Code is available at https://github.com/BenfengXu/S2ynRE.

对于RE任务来说，高质量有标注的data获取很难，之前一种解决这个问题的思路是远监督distant supervision，尽管远监督获得了效果的提升，但是远监督的数据不能够保证和下游任务的schema、context分布特征等是相符的：

> Although this line of methods have seen certain improvements, they still inevitably raise the concern that the distantly annotated data can vary considerably from downstream tasks both in target schema and in context distributions, thus may not be able to offer optimal transferability.

换句话说，要获得理想的领域特征一致的远监督数据本身也可能是比较难的。

因此，作者顺着最近的一些利用LLM生成text data的工作的思路，考虑使用LM来生成数据。作者的贡献主要有两点：

- 利用GPT-3.5和finetuned GPT-2 Large去适应target domain distribution，然后生成无label的RE data
- 提出了a two-stage self-training训练策略，更好的利用生成的无标注数据和原有标注数据

作者的RE任务是给定头尾实体，预测relation。

利用GPT-2 Large生成数据，首先按照language modeling的loss在训练集上微调；然后在推理阶段，输入`<bos>`开始进行采样生成new data。

利用GPT-3生成数据，采用5-shot ICL，随机找demonstrations的策略：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230926161739020.png"  style="zoom:50%;" />

注意这里prompt对于结果的可控，只是通过一些指令性的表述，如`similar topic, domain and the same sub-obj format`。

然后是如何利用生成的无标注data，一般的策略是self-training，即给无标注data伪标注然后和原有data混合，训练小模型，训练好的小模型再重新标注无标注data。

作者认为这种直接将生成的数据加入到原有的数据方法前提是，要求生成的数据需要和原来的数据有一样的分布。

相反，作者将无标注数据和有标注数据分开，先使用gold data训练多个teacher model，然后标注生成的data，注意是soft label；然后用一个新初始化的student model在带有soft label的生成数据上训练，更新参数；之后继续在gold data上训练，更新后的model重新标注生成的data；这样迭代式的训练：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230926162246951.png"   style="zoom:50%;" />

对于实验结果具体可以参考原paper，这里提供几个值得记录的结果：

作者使用BERT+Linear作为RE model。

直接用GPT不一定能够超过finetuned LM来生成data，下面的结果没有找到是具体哪个dataset上的测试结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230926162817492.png"  style="zoom:50%;" />

作者使用type-token ratio ([*Evaluating story generation systems using automated linguistic analyses. 2017*]; *Data augmentation using pre-trained transformer models. 2020*)来评估diversity。



# Cross-domain IE

## CDA

Data Augmentation for Cross-Domain Named Entity Recognition

简写是Cross-domain Data Augmentation (CDA)方法。

EMNLP 2021，休斯顿大学与Snap，[代码](https://github.com/RiTUAL-UH/style_NER)。

> Current work in named entity recognition (NER) shows that data augmentation techniques can produce more robust models. However, most existing techniques focus on augmenting in-domain data in low-resource scenarios where annotated data is quite limited. In contrast, **we study cross-domain data augmentation for the NER task.** We investigate the possibility of leveraging data from highresource domains by projecting it into the lowresource domains. Specifically, we propose a novel neural architecture to transform the data representation from a high-resource to a **low-resource domain by learning the patterns (e.g. style, noise, abbreviations, etc.)** in the text that differentiate them and a shared feature space where both domains are aligned. We experiment with diverse datasets and show that transforming the data to the low-resource domain representation achieves significant improvements over only using data from high-resource domains.

应该是首个考虑用数据增强策略做跨域NER任务的方法。

之前的数据增强IE方法主要是利用in-domain data进行数据增强。作者发现不同domain有不同的patterns：

> Based on our observations, the text in different domains usually presents unique patterns (e.g. style, noise abbreviations, etc.).

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913185736771.png"   style="zoom:30%;" />

例如上面例子中新闻domain的句子更长，表达也更加正式；而social domain的句子有更多的噪音，句子更短，有更多口语/个性化的表达。

但是，作者认为不同domain的text的语义是可以迁移的，并且是存在领域不变量invariables的。作者研究从high-resource domain到low-resource domain数据增强NER方法。

和之前的数据增强方法一样，作者同样训练了一个LM来生成数据，编码器+解码器。编码器是biLSTM，解码器是另一层LSTM。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913191836826.png"   style="zoom:30%;" />

作者提出的训练模型包括两步：

- Denoising Reconstruction：learn the textual pattern and generate compressed representations of the data from each domain

  - 在输入的text中加入噪音，能够强迫model更加学会保留原始的数据结构信息，所以作者首先通过几种word-level operation来插入噪音：

    <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913190334476.png"  style="zoom:30%;" />

  - 使用相同参数的encoder和decoder去重建两个domain的input sentence。模型的参数可以看做是学习了隐式的领域对齐。loss是解码器输出和input text之间的差异。

  - 这一步还额外训练了一个对抗式判别器discriminator用来判断编码器的输出是来自哪个领域，为下一步model学习domain mapping做准备。

- Detransforming Reconstruction：align the compressed representations of the data from different domains so that the model can project the data from one domain to another

  - 首先，用上一步学习好的encoder+decoder，把source domain的sentence转化为target domain style的sentence；把target domain的sentence转化为source domain style的sentence
  - 然后，利用跨域转化后的句子，经过编码器和解码器，期望能够恢复在原来domain的句子
  - 这一步继续训练对抗式判别器discriminator，如果判别器根据编码器的输出，判断领域变换后的sentence是原来domain的概率越小，则认为domain mapping效果越好

 作者基于Ontonotes 5.0 Dataset（domains：Broadcast Conversation (BC), Broadcast News (BN), Magazine (MZ), Newswire (NW), and Web Data (WB).）和Temporal Twitter Dataset（Social Media (SM) domain）进行实验。基于source domain的data生成target domain的新training data。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913192010655.png"   style="zoom:30%;" />

上面实验结果中能够看出，原来的in-domain的数据增强方法（如DAGA方法）无法很好的处理跨域问题。这说明原来数据增强方法无法直接生成对应domain的数据。

实验用的NER model是BERT+Linear Layer。



## Style Transfer

Style Transfer as Data Augmentation: A Case Study on Named Entity Recognition

与前面CDA是同一作者。EMNLP 2022，[代码](https://github.com/RiTUAL-UH/DA_NER)。

> In this work, we take the named entity recognition task in the English language as a case study and explore style transfer as a data augmentation method to increase the size and diversity of training data in low-resource scenarios. We propose a new method to effectively transform the text from a high-resource domain to a low-resource domain by **changing its style-related attributes to generate synthetic data for training.** Moreover, we design a constrained decoding algorithm along with a set of key ingredients for data selection to guarantee the generation of valid and coherent data. Experiments and analysis on five different domain pairs under different data regimes demonstrate that our approach can significantly improve results compared to current state-of-the-art data augmentation methods. Our approach is a practical solution to data scarcity, and we expect it to be applicable to other NLP tasks.

作者探究使用style transfer来为cross-domain NER任务做数据增强的方法。由于并没有带有NER label的style transfer数据集，因此作者提出可以利用非NER任务的style transfer数据集。（风格转换一定程度上不局限在特定任务，但是作者这种做法有个隐含的前提，就是NER的source domain和target domain中的styles已经包括在了非NER任务的style transfer数据集中）。

作者同样训练一个encoder+decoder的LM进行数据生成。这篇论文中作者使用的是T5-base。

第一步就是在非NER任务的style transfer数据集GYAFC (Rao and Tetreault, 2018)上进行训练。这个数据集包括了formal and informal的句子对。通过输入某个style的句子，让T5学会输出对应其它style的句子，优化重建loss $L_{pg}$。作者follow前人的工作，将style transfer看做是改写生成的问题paraphrase generation problem。和作者之前工作CDA中的domain判别器类似，这一步也额外训练了一个对抗性的style判别器，用来判断编码器输出的embedding是属于哪种style。

第二步是想办法让T5能够学会在NER的句子上进行风格转换。首先要把label注入到sentence中，这里作者把`<START_ENTITY_TYPE>` and `<END_ENTITY_TYPE>`插入到entity span的左右侧。然后就是作者提出的cycle-consistent reconstruction，简单说是输入某个sentence，让T5转化为另一种style的sentence，把这个转换后的sentence再输入到T5中，让T5重新回复原来style的sentence。第二步同样优化对抗性style判别器。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913215520630.png"   style="zoom:40%;" />

第三步是生成。为了保证生成数据是valid的，提出了基于prefix tree的Constrained Decoding策略，是保留top-K或top-p的token候选项，然后约束生成句子的输出范围，比如之前输出的span是属于`<Text>`，那么接下来输出的span就必须是`<EOS>` or `<B_ENT>`：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913220118855.png"   style="zoom:30%;" />

即使经过上一步，也不能保证生成数据是可靠的。为了进一步提升质量，比如过滤掉简单的重复、胡言乱语等生成的text，计算四个方法的4个metric，然后加权求和作为对于生成data质量的评估：

- Consistency: a confidence score from a pretrained style classifier as the extent a generated sentence is in the target style. 基于T5 base，用一个外部的model判断是否符合特定style
- Adequacy: a confidence score from a pretrained NLU model on how much semantics is preserved in the generated sentence. 基于[开源model](https://github.com/ PrithivirajDamodaran/Parrot_Paraphraser)，判断生成句子保留的语义
- Fluency: a confidence score from a pretrained NLU model indicating the fluency of the generated sentence. 基于[开源model](https://github.com/ PrithivirajDamodaran/Parrot_Paraphraser)，判断生成句子的流程程度
- Diversity: the edit distance between original sentences and the generated sentences at the character level. 利用原始sentence和生成sentence的编辑距离来衡量生成句子的多样性。

实验数据集与CDA中的一样，使用OntoNotes 5.0作为source domain和Temporal Twitter Corpus作为target domain。OntoNotes 5.0的domain style是formal的，Temporal Twitter Corpus的domain style是informal的。

作者的NER model是基于BERT base+Linear，与CDA的一致。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230913230108250.png"   style="zoom:30%;" />



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

实验以CONLL2003作为source domain，CrossNER数据集下的多个子集作为target domain。训练的时候应用到了fine-tuning based和prompt-tuning based两种NER微调策略，具体参考论文。作者在BERT和RoBERT两类模型不同size的LM上进行了实验。

只使用source domain的数据来进行训练，然后测试在target domain上的效果，验证模型的领域泛化性。
