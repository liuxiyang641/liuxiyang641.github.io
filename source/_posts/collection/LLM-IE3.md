---
title: LLM-IE3
published: false
date: 2024-03-24 23:20:28
categories:
  - Paper
  - LLM
  - IE
tags:
  - LLM
  - IE
  - Collection
---

# 基于LLM的Information Extraction 3

基于LLM的信息抽取工作总结合集3。

<!--more-->

## DocGNRE

Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models. BIGAI. EMNLP 2023. [代码](https://github.com/bigai-nlco/DocGNRE).

> Document-level Relation Extraction (DocRE), which aims to extract relations from a long context, is a critical challenge in achieving finegrained structural comprehension and generating interpretable document representations. Inspired by recent advances in in-context learning capabilities emergent from large language models (LLMs), such as ChatGPT, we aim to design an automated annotation method for DocRE with minimum human effort. Unfortunately, **vanilla in-context learning is infeasible for document-level Relation Extraction (RE) due to the plenty of predefined fine-grained relation types and the uncontrolled generations of LLMs.** To tackle this issue, we propose a method integrating a Large Language Model (LLM) and a natural language inference (NLI) module to generate relation triples, thereby augmenting document-level relation datasets. We demonstrate the effectiveness of our approach by introducing an enhanced dataset known as **DocGNRE, which excels in re-annotating numerous long-tail relation types.** We are confident that our method holds the potential for broader applications in domain-specific relation type definitions and offers tangible benefits in advancing generalized language semantic comprehension.

**Issue**：对document进行标注代表是很大的，最开始的DocRED数据集存在很多false negative，因此后续出现了Re-DocRED等工作，其面临：

- achieving complete manual annotation is challenging：DocRED每个document平均存在19.5 entities，97种relation，要求考虑37,000种可能候选结果
- the supplementary annotations are derived from the existing data distribution：Re-DocRED使用了RE model来重新标注，这引入了model bias，即标注的结果仍然无法很好的解决长尾relation标注的问题

**Solution**：作者提出利用LLM来增强DocRE相关数据集（即ReDocRED数据集）的completeness，获得了新的数据集DocGNRE，其分布：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324233008294.png"  style="zoom:40%;" />

文档级关系抽取明显比句子级关系抽取更困难，由于更多细粒度的关系类型、需要抽取的候选更多。

作者观察到，LLM常常会生成不满足输入constrains的结果。此外，作者发现，让LLM使用contextual words描述relation，而不是predefined relation types，结果更加准确。因此，作者使用一个pretrained NLI model来进行label映射，LLM是负责进行更加Open的信息抽取：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324233149323.png" style="zoom:50%;" />

整体流程：

1. 输入document和候选实体集合，让LLM输出候选三元组集合（注意没有relation label）
2. 对于候选三元组，直接拼接头尾实体和关系成为自然语言后，作为premise；
3. 迭代的替换relation，作为hypothesis
4. 利用`T5-based NLI model`预测premise和hypothesis的关系（entailment, neutrality, or contradiction），如果entailment的得分最高，就认为映射成功，保留三元组
5. 对于数据集的test set，作者使用Mechanical Turk进行了人工验证，保证测试集的正确性

实验基于`gpt-3.5-turbo`，可以看到在zero-shot document-level RE任务上，直接利用LLM进行预测的效果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324233632483.png"  style="zoom:50%;" />

使用构造的数据集，训练目前SOTA模型的效果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324233703017.png" style="zoom:30%;" />

## GenRDK

Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction. 南京科技大学. WWW 2024. [代码](https://github.com/QiSun123/GenRDK).

> Document-level Relation Triplet Extraction (DocRTE) is a fundamental task in information systems that aims to simultaneously extract entities with semantic relations from a document. Existing methods heavily rely on a substantial amount of fully labeled data. **However, collecting and annotating data for newly emerging relations is time-consuming and labor-intensive.** Recent advanced Large Language Models (LLMs), such as ChatGPT and LLaMA, exhibit impressive long-text generation capabilities, inspiring us to explore an alternative approach for obtaining auto-labeled documents with new relations. In this paper, we propose a Zero-shot Document-level Relation Triplet Extraction (ZeroDocRTE) framework, which Generates labeled data by Retrieval and Denoising Knowledge from LLMs, called GenRDK. Specifically, we propose a chain-of-retrieval prompt to guide ChatGPT to generate labeled long-text data step by step. To improve the quality of synthetic data, **we propose a denoising strategy based on the consistency of cross-document knowledge.** Leveraging our denoised synthetic data, we proceed to fine-tune the LLaMA2-13B-Chat for extracting document-level relation triplets. We perform experiments for both zero-shot document-level relation and triplet extraction on two public datasets. The experimental results illustrate that our GenRDK framework outperforms strong baselines.

**Issue**：对于新出现的relation，要获得对应的数据代价是很高的。

**Solution**: 作者首先是提出了一个zero-shot文档级三元组抽取任务，Document-level Relation Triplet Extraction task。然后，利用LLM为新relation生成训练数据，再训练对应的model（也是一个LLM）进行三元组抽取。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324234132619.png"  style="zoom:50%;" />

首先是，如何让LLM为新relation生成对应的document。找和unseen relation相似的relation作为输入集合，然后，利用多轮对话形式的一系列prompt步骤（即作者声称的chain-of-retrieval），生成新的document数据。这里作者使用ChatGPT进行生成：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324234352718.png"  style="zoom:50%;" />

为了解决生成的数据中存在错误标注以及缺失标注的问题，作者首先训练了一个model来进行伪标注：

- 基于已知的relation数据集，分割为多个独立的relation group，分别独立的作为输入。让其学会预测输入的目标relation
- `LLaMA2-13B-Chat`，`LoRA`进行训练
- 然后对生成的document进行伪标注，输入是unseen relation，输出是pseudo label

作者观察到，相同的relation fact可能在不同的document中出现。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324234704729.png"  style="zoom:50%;" />

因此，作者期望伪标注的label，和包括了unseen relation label的synthetic的label是一致的。在所有构造的document数据中，作者分别构造了两个graph $KG_{p},KG_{s}$，使用entities作为node、pseudo label/synthetic label作为edge，使用三元组出现的频率$F$作为edge weight。

然后把两个graph融合，也就是对应三元组频率相加即可：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324235107132.png"  style="zoom:50%;" />

如果相加后的某个三元组出现频率过低，那么作者怀疑是错误的标注。因此进行裁剪，使用平均值-标准差作为下界阈值，低于阈值的三元组被移除：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240324235611182.png"  style="zoom:40%;" />

实验同时测试了Zero-shot DocRTE（基于`LLaMA2-13B-Chat`进行微调）和DocRE（基于graph-based DocRE model进行微调）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240325000423590.png"  style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240325000442047.png"  style="zoom:50%;" />

## LLMaAA

LLMaAA: Making Large Language Models as Active Annotators. 北大. EMNLP 2023 Findings. [代码](https://github. com/ridiculouz/LLMAAA).

> Prevalent supervised learning methods in natural language processing (NLP) are notoriously data-hungry, which demand large amounts of high-quality annotated data. In practice, acquiring such data is a costly endeavor. Recently, the superior performance of large language models (LLMs) has propelled the development of dataset generation, where the training data are solely synthesized from LLMs. However, **such an approach usually suffers from low-quality issues and requires orders of magnitude more labeled data to achieve satisfactory performance.** To fully exploit the potential of LLMs and make use of massive unlabeled data, **we propose LLMaAA, which takes LLMs as annotators and puts them into an active learning loop to determine what to annotate efficiently.** To learn robustly with pseudo labels, we optimize both the annotation and training processes: (1) we draw k-NN samples from a small demonstration pool as in-context examples, and (2) we adopt the automatic reweighting technique to assign training samples with learnable weights. Compared with previous approaches, LLMaAA features both efficiency and reliability. We conduct experiments and analysis on two classic NLP tasks, named entity recognition and relation extraction. With LLM A AA, task-specific models trained from LLM-generated labels can outperform their teacher LLMs within only hundreds of annotated examples, which is much more cost-effective than other baselines.

**Issue**: 对于将LLM作为data annotator，如何尽可能地标注成本，并且是task model性能尽可能高。

作者声明的，直接将LLM应用到各种下游任务的production application中，可能存在的两个方面的担忧：

- Under the prevalent “Language-Model-as-a-Service” (LMaaS, Sun et al., 2022) setting, users are required to feed their own data, potentially including sensitive or private information, to third-party LLM vendors to access the service, which increases the risk of data leakage (Lyu et al., 2020; Yu et al., 2022; Li et al., 2023). 利用LLM来直接进行下游任务，存在数据泄露风险，data-sensitive
- LLMs usually consume abundant tokens by continuous requests to APIs, where the marginal cost and latency become substantial in large-scale or real-time applications, hindering LLMs’ practical deployment in cost-sensitive scenarios (Goyal et al., 2020; Cao et al., 2023). 在需要大规模或者实时处理的cost-sensitive真实应用上的风险

**Solution**: 作者的方法，主要是采用了简单的主动学习的策略，来迭代的选择适合于让LLM进行标注的unlabeled data：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240331224321626.png"  style="zoom:50%;" />

作者以$100$ 个gold data出发作为种子集合，然后利用k-NN检索demonstrations，让LLM进行标注，形成一个small labeled data。

为了处理这个标注的数据中，可能存在的noise label，作者采用了Robust Learning学习方法，核心思想就是给东不同的标注sample不同的weight。估计weight的具体做法是：同样将上面的$100$ 个gold data作为一个小的validation set，然后选择能够使得在validation上loss最小的sample weight。具体过程参考论文。

最后，利用activate learning，

> Active learning (AL) seeks to reduce labeling efforts by strategically choosing which examples to annotate.

作者尝试了几种不同的简单的主动学习策略：

- Maximum Entropy：选择预测概率熵最大的，就是最不确定的sample

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240331224841930.png"  style="zoom:50%;" />

- Least Confidence：选择最不confidence的sample

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240331224903463.png"  style="zoom:50%;" />

- K-Means：对候选无标注样本进行k-means聚类，然后选择中心的k个样本作为差异度最大的样本

作者的实验采用的LLM，主要是`ChatGPT`，但是利用`GPT-3`和`GPT-4`进行了对比。对于NER数据集，采用了Chinese OntoNotes 4.0、CoNLL03；对于RE，采用了Re-TACRED数据集，但是只实验了和person type相关的relation：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240331225108921.png" style="zoom:30%;" />

无标注数据就是原来的训练集；最后一共标注500个新的数据。基于`bert-base-cased`或者`chinese-bert-base-wwm`作为task model backbone。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240331225213836.png"  style="zoom:50%;" />

最后，作者还简单讨论了一下，为什么理论上，利用teacher model可以训练出来更强的student model。
