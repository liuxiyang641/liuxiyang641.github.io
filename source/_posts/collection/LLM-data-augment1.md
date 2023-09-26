---
title: LLM-data-augment1
published: true
date: 2023-09-22 17:04:13
categories:
- Paper
- LLM
- Data Augment
tags:
- LLM
- Data Augment
- Collection
---

# LLM数据增强

基于LLM的数据增强论文合集1。

<!--more-->

## GPT3Mix

GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation

EMNLP 2021 findings，NAVER AI lab，[代码](https://github.com/naver-ai/hypermix)。

> Large-scale language models such as GPT3 are excellent few-shot learners, allowing them to be controlled via natural text prompts. Recent studies report that prompt-based direct classification eliminates the need for fine-tuning but lacks data and inference scalability. **This paper proposes a novel data augmentation technique that leverages large-scale language models to generate realistic text samples from a mixture of real samples.** We also propose utilizing soft-labels predicted by the language models, effectively distilling knowledge from the large-scale language models and creating textual perturbations simultaneously. We perform data augmentation experiments on diverse classification tasks and show that our method hugely outperforms existing text augmentation methods. We also conduct experiments on our newly proposed benchmark to show that the augmentation effect is not only attributed to memorization. Further ablation studies and a qualitative analysis provide more insights into our approach.

作者声称是首个用LLM来进行数据增强工作的文章。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922204210110.png"   style="zoom:35%;" />

作者认为的需要用LLM来做数据增强，而不是直接执行特定任务的理由：

- First, the number of incontext training examples is hard limited by the maximum prompt length enabled by the inherent language model architecture.
- Second, promptbased approaches require online inference on the expensive large-scale language models. The inference may not be scalable in real-world use cases, because it is slow and incurs huge memory overhead.
- Lastly, the prompt-based approaches do away with conventional machine learning techniques, making it mostly incompatible with existing established fine-tuning methods.

提出的GPT3Mix方法图：

![image-20230922203338068](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922203338068.png)

作者构造的生成训练数据的prompt template包括：

- Text Type：输入的text的类型，比如movie review
- Label Type：label class的类型，比如情感分类中是sentiment
- Label-token Verbalizer：将label转化为text tokens

构造数据的ICL prompt示例（不指定生成数据的label、prompt中一次性提供所有的label信息只适用于label数量较少的情况）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922203610701.png"   style="zoom:40%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922203625168.png"   style="zoom:40%;" />

生成数据的时候，注意两点：

- 同时生成data和label，并且label在data之后出现，这样保证label是依赖于生成的data的
- 不是简单的生成label，还使用LLM对label tokens的conditional probability来作为soft-label。带有soft-label的GPT生成的text加入到原来的数据中，获得效果的提升。

上下文的demonstrations是从训练集中随机采样，默认使用的样例数量是2。

作者在实验部分，还自己构造了一个RT20 binary sentiment classification数据集（关于movie reviews），收集的是GPT3使用的最新data之后出现的新评论。下面是实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922204133079.png"  style="zoom:40%;" />

## Data Augmentation for Intent Classification

Data Augmentation for Intent Classification with Off-the-shelf Large Language Models

NLP4ConvAI 2022，[代码](https://github.com/ElementAI/data-augmentation-with-llms)。

> Data augmentation is a widely employed technique to alleviate the problem of data scarcity. **In this work, we propose a prompting-based approach to generate labelled training data for intent classification with off-the-shelf language models (LMs) such as GPT-3.** An advantage of this method is that no task-specific LM-fine-tuning for data generation is required; hence the method requires no hyper-parameter tuning and is applicable even when the available training data is very scarce. We evaluate the proposed method in a few-shot setting on four diverse intent classification tasks. We find that GPT-generated data significantly boosts the performance of intent classifiers when intents in consideration are sufficiently distinct from each other. In tasks with semantically close intents, we observe that the generated data is less helpful. Our analysis shows that this is because GPT often generates utterances that belong to a closely-related intent instead of the desired one. **We present preliminary evidence that a prompting-based GPT classifier could be helpful in filtering the generated data to enhance its quality.**

作者认为之前的GPT3Mix工作，由于每次prompt会直接提供所有的label，并且不限制生成数据的label类型，只使用与label数量少的任务。而意图分类Intent Classification（IC）任务可能有上百个不同的intent，并且可能存在非常相近的intent，导致GPT3Mix不适用于意图分类task。

> an intent is a type of request that the conversational agent supports; e.g. the user may want to change the language of the conversation, play a song, transfer money between accounts, etc.

作者的方法主要有两点：

- 每次只生成一类label的数据（但同样是随机找的demonstrations）

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922224921205.png"   style="zoom:30%;" />

- 提出可以用LLM分类的能力，过滤掉text和label不对应的生成数据

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922225011759.png"   style="zoom:30%;" />

## InPars

InPars: Unsupervised Dataset Generation for Information Retrieval

SIGIR 2022 Short paper，[代码](https://github.com/zetaalphavector/inpars)。

> The Information Retrieval (IR) community has recently witnessed a revolution due to large pretrained transformer models. Another key ingredient for this revolution was the MS MARCO dataset, whose scale and diversity has enabled zero-shot transfer learning to various tasks. However, not all IR tasks and domains can benefit from one single dataset equally. Extensive research in various NLP tasks has shown that using domain-specific training data, as opposed to a general-purpose one, improves the performance of neural models [45, 56]. **In this work, we harness the few-shot capabilities of large pretrained language models as synthetic data generators for IR tasks.** We show that models fine-tuned solely on our synthetic datasets outperform strong baselines such as BM25 as well as recently proposed self-supervised dense retrieval methods. Code, models, and data are available at https://github.com/zetaalphavector/inpars.

大模型可以直接用来判断query和document之间的相关性，但是由于候选document的数量过多，在实践中，要用LLM来进行检索还是不实际的。

传统的dense retriever通过提前计算document的embedding，利用现有的搜索框架，可以高效的寻找相关documents。但是训练dense retriever需要足够的domain-specific training data。

因此，作者提出使用LLM来生成IR任务的数据 $(query, document)$ pairs：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230922235622982.png"   style="zoom:30%;" />

步骤：

- 通过corpus-level ICL，人工构造3个样例，实现利用LLM为document生成对应的问题

- 从语料库中采样$100,000$个documents，让LLM针对documents生成一个初步相关question，生成question用了两个模板：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923000908755.png"   style="zoom:30%;" />

  第一个模板是原始的训练集中document和对应的question，第二个模板是作者提出的Guided by Bad Questions（GBQ）模板，将原来的question作为bad question，然后人工写一个更加复杂的question作为good question，让LLM生成good question。

- 利用LLM的conditional probability估计document-question之间的相关性：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923001145634.png"   style="zoom:50%;" />

  依据相关性排序，从$100,000$个document-question pairs中，选择$1,000$个topK的数据作为训练数据。

实验效果发现，在无监督的IR方法上效果不错，但是距离有监督的IR方法还有差距：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923001337490.png"   style="zoom:40%;" />

## AugGPT

AugGPT: Leveraging ChatGPT for Text Data Augmentation

arXiv 2023-03

> Text data augmentation is an effective strategy for overcoming the challenge of limited sample sizes in many natural language processing (NLP) tasks. This challenge is especially prominent in the few-shot learning scenario, where the data in the target domain is generally much scarcer and of lowered quality. **A natural and widely-used strategy to mitigate such challenges is to perform data augmentation to better capture the data invariance and increase the sample size.** However, current text data augmentation methods either can’t ensure the correct labeling of the generated data (lacking faithfulness) or can’t ensure sufficient diversity in the generated data (lacking compactness), or both. Inspired by the recent success of large language models, especially the development of ChatGPT, which demonstrated improved language comprehension abilities, in this work, **we propose a text data augmentation approach based on ChatGPT (named AugGPT).** AugGPT rephrases each sentence in the training samples into multiple conceptually similar but semantically different samples. The augmented samples can then be used in downstream model training. Experiment results on few-shot learning text classification tasks show the superior performance of the proposed AugGPT approach over state-of-the-art text data augmentation methods in terms of testing accuracy and distribution of the augmented samples.

基于改写方法的利用LLM生成text classification task的训练数据。作者认为LLM适合用来做数据增强的两点理由：

- LLM通过预训练，在大规模语料中进行预训练，在自己的参数空间中学习编码到了很多隐式的factual knowledge
- LLM经过RLHF等训练阶段，能够produce more informative and impartial responses to input

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923151338440.png"   style="zoom:40%;" />

作者的实验是用clinical NLP tasks来进行评估，因为Data augmentation is particularly in demand in clinical NLP, because the significant burden of expert annotation and stringent privacy regulations make large-scale data labeling infeasible.

改写用的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923151506045.png"   style="zoom:40%;" />

值得一提的是，作者在实验部分用生成句子和原来句子的embedding cosine相似度评估语义的一致性；然后用TransRate指标来评估生成的数据是否足够compactness，以至于能够方便对不同的类进行区分：

> TransRate is a metric that quantifies transferability based on the mutual information between the features extracted by a pre-trained model and their labels, with a single pass through the target data. A higher TransRate could indicate better learnability of the data.

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923151847434.png"   style="zoom:50%;" />

## DISCO

DISCO: Distilling Counterfactuals with Large Language Models

ACL 2023，EPFL，[代码](https://github.com/eric11eca/disco)。

> Models trained with counterfactually augmented data learn representations of the causal structure of tasks, enabling robust generalization. However, high-quality counterfactual data is scarce for most tasks and not easily generated at scale. When crowdsourced, such data is typically limited in scale and diversity; when generated using supervised methods, it is computationally expensive to extend to new counterfactual dimensions. In this work, **we introduce DISCO (DIStilled COunterfactual Data), a new method for automatically generating high-quality counterfactual data at scale.** DISCO engineers prompts to generate phrasal perturbations with a large general language model. Then, a task-specific teacher model filters these generations to distill high-quality counterfactual data. While task-agnostic, we apply our pipeline to the task of natural language inference (NLI) and find that on challenging evaluations such as the NLI stress test, comparatively smaller student models trained with DISCOgenerated counterfactuals are more robust (6% absolute) and generalize better across distributions (2%) compared to models trained without data augmentation. Furthermore, DISCOaugmented models are 10% more consistent between counterfactual pairs on three evaluation sets, demonstrating that DISCO augmentation enables models to more reliably learn causal representations. Our repository are available at: https://github.com/eric11eca/disco

首个利用LLM生成counterfactual data的工作。

Counterfactual data augmentation (CAD) [*Learning the difference that makes a difference with counterfactually-augmented data. 2019*]是一种移除spurious correlation，提升模型robustness的数据增强技术。它修改原来数据中的重要组成部分，让原始的label改变：

> Counterfactual data augmentation (CAD) (Kaushik et al., 2019) is one general approach to improve model robustness by training on edited instances that systematically alter the critical or causally salient parts of dataset instances that contributes to the label assignment.

一个不够鲁棒的model，对于反事实，可能由于在训练过程学习到了伪相关的特征，从而导致反事实的data还是预测的原来的label。

作者针对NLI任务进行实验，NLI任务的输入是premise-hypothesis pair $<P,H>$，标签$l\in \{ Entailment, Contradiction, Neutral \}$。作者的反事实数据增强方法就是修改前提premise的部分描述，新的前提$P^{\prime}$能够让label变更为新$l^{\prime}$。

作者提出方法的大致流程：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923225708595.png"   style="zoom:35%;" />

- 首先，作者只选择了部分训练集中的数据进行训练，利用*dataset cartography*技术[*Dataset cartography: Mapping and diagnosing datasets with training dynamics. 2020*]，选择ambiguous和hard samples作为之后进行增强的对象；

- 然后，作者利用之前的一个neural syntactic parser[*FLAIR: An easy-to-use framework for state-of-the-art NLP. NAACL 2019*]将前提premise句子转化为span的集合；

- 由于不知道修改哪个span能够导致label的改变，迭代地用`[blank]`去替换每个span，然后让LLM填补`[blank]`以至于对应的label变为新的label。作者提出了两种prompt，一种是让LLM补全`[blank]`，是最常见的GPT的completing mode；一种是利用GPT3的*insert mode*，让GPT直接回复中间确实内容：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230923230609837.png"   style="zoom:40%;" />

- 最后，过滤质量比较低的data；首先是用一系列预定义规则进行过滤heuristic-based automatic filter，如是不是和原来的数据有较多的重叠、是不是使用了negation words作为捷径以反转label。然后用一个现有的SOTA NLI模型，给定原始数据和新数据，确定预测的label分布是否发生了大的变化。

实验部分内容较多，具体参考论文。有2个衡量多样性的metric可以注意下：

- Self-BLEU [*Texygen: A benchmarking platform for text generation models. 2018*]：评估lexical and syntactic diversity
- Dataset Distance：计算新data和旧data的信息多样性，使用OTDD (optimal transport dataset distance) [*Geometric dataset distances via optimal transport. 2020*]作为指标

## GenRead

Generate rather than Retrieve: Large Language Models are Strong Context Generators

{% post_link llm/GenRead  [详细博客] %}。

University of Notre Dame和Microsoft，ICLR 2023，[代码](https://github.com/wyu97/GenRead)。

> Knowledge-intensive tasks, such as open-domain question answering (QA), require access to a large amount of world or domain knowledge. A common approach for knowledge-intensive tasks is to employ a retrieve-then-read pipeline that first retrieves a handful of relevant contextual documents from an external corpus such as Wikipedia and then predicts an answer conditioned on the retrieved documents. **In this paper, we present a novel perspective for solving knowledge-intensive tasks by replacing document retrievers with large language model generators.** We call our method generate-then-read (GenRead), which first prompts a large language model to generate contextual documents based on a given question, and then reads the generated documents to produce the final answer. Furthermore, **we propose a novel clustering-based prompting method that selects distinct prompts, in order to generate diverse documents that cover different perspectives, leading to better recall over acceptable answers.** We conduct extensive experiments on three different knowledge-intensive tasks, including open-domain QA, fact checking, and dialogue system. Notably, GenRead achieves 71.6 and 54.4 exact match scores on TriviaQA and WebQ, significantly outperforming the state-of-the-art retrieve-thenread pipeline DPR-FiD by +4.0 and +3.9, without retrieving any documents from any external knowledge source. Lastly, we demonstrate the model performance can be further improved by combining retrieval and generation. Our code and generated documents can be found at https://github.com/wyu97/GenRead.

作者提出了使用LLM生成的question的documents，作为question的background来回答问题，*generate-then-read*。

knowledge-intensive tasks如开放域QA任务等，常常需要大量的word knowledge / domain knowledge。之前的常常通过检索外部知识源Wikipedia等来获得relevant contextual documents。

*retrieve-then-read*来解决knowledge-intensive tasks存在的问题：

- First, candidate documents for retrieval are chunked (e.g., 100 words) and fixed, so the retrieved documents might contain noisy information that is irrelevant to the question. 
- Second, the representations of questions and documents are typically obtained independently in modern two-tower dense retrieval models (Karpukhin et al., 2020), leading to only shallow interactions captured between them (Khattab et al., 2021). 
- Third, document retrieval over a large corpus requires the retriever model to first encode all candidate documents and store representations for each document.

而作者认为，LLM生成的document比传统的检索结果更加和query question更加相关，原因是：LLM的生成结果是通过基于question的token，然后经过attention等机制生成的，而一般的检索只是利用question和document的embedding相似度去检索的。显然LLM的生成结果会和question更加相关。

> We believe this is because large language models generate contextual documents by performing deep token-level cross-attention between all the question and document contents, resulting in generated documents that are more specific to the question than retrieved documents.

在检索方法中，检索的答案越多，能够提供更多的不同角度/方面的knowledge，从而增加最后回答答案的准确率。

但是如果是相同的prompt，LLM会倾向不断输出重复的内容。因此作者提出从不同的聚类中选择上下文样例，从而产生更多样的输出documents。

作者提出clustering-based prompt方法，提取不同的上下文样例，构造不同的prompt，生成的多个结果文档，一起再来辅助回答问题。核心包括3步：

1. 初始化：先用LLM给训练集中的每个question生成一个document。也可以使用检索的方法，为每个question从外部知识源中检索一个相关document；
2. 编码document，基于K-means无监督聚类：作者使用GPT-3这类LLM为每个question-document进行编码，然后进行K-means聚类。聚类的数量K，和要生成的documents数量一致
3. 采样并且生成K个documents：对每一个聚类，采样n个样例作为上下文，然后生成query question的一个document，最终生成的K个documents。这些documents作为background，和query question组合成一个prompt，获得最终的答案。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230915160124104.png"   style="zoom:50%;" />

## Increasing diversity and accuracy

Increasing Diversity While Maintaining Accuracy: Text Data Generation with Large Language Models and Human Interventions. 

密歇根大学与Microsoft，ACL 2023. {% post_link llm/Increasing-Diver-Acc-Data-Gen-LLM  [详细博客] %}。

> Large language models (LLMs) can be used to generate text data for training and evaluating other models. However, creating high-quality datasets with LLMs can be challenging. **In this work, we explore human-AI partnerships to facilitate high diversity and accuracy in LLM-based text data generation.** We first examine two approaches to diversify text generation: 1) logit suppression, which minimizes the generation of languages that have already been frequently generated, and 2) temperature sampling, which flattens the token sampling probability. We found that diversification approaches can increase data diversity but often at the cost of data accuracy (i.e., text and labels being appropriate for the target domain). To address this issue, we examined two human interventions, 1) label replacement (LR), correcting misaligned labels, and 2) out-of-scope filtering (OOSF), removing instances that are out of the user’s domain of interest or to which no considered label applies. With oracle studies, we found that LR increases the absolute accuracy of models trained with diversified datasets by 14.4%. Moreover, we found that some models trained with data generated with LR interventions outperformed LLM-based few-shot classification. In contrast, OOSF was not effective in increasing model accuracy, implying the need for future work in human-in-the-loop text data generation.

作者讨论了2种在解码阶段增加多样性的方法：

- Logit Suppression：decreases the probability of high-frequency tokens。之前生成的tokens，根据频率，降低它在下一次采样中的概率。这里叫做logit的原因应该是，作者通过调用OpenAI的logit bias API来实现这一点。

- High Temperature：增大temperature $T$，更大的temperature意味着最终的概率分布更加平滑flat：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230914230956172.png"   style="zoom:40%;" />

示意图：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230914231021977.png"   style="zoom:50%;" />

作者调用GPT生成数据的时候，考虑的是短文本分类任务，构造的prompt主要考虑text type和label：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230914231203885.png"   style="zoom:40%;" />

## Vote-k

Selective Annotation Makes Language Models Better Few-Shot Learners. 

香港大学，ICLR 2023，[代码](https://github.com/HKUNLP/icl-selective-annotation)。

> Many recent approaches to natural language tasks are built on the remarkable abilities of large language models. Large language models can perform in-context learning, where they learn a new task from a few task demonstrations, without any parameter updates. This work examines the implications of in-context learning for the creation of datasets for new natural language tasks. Departing from recent in-context learning methods, we formulate an annotation-efficient, two-step framework: selective annotation that chooses a pool of examples to annotate from unlabeled data in advance, followed by prompt retrieval that retrieves task examples from the annotated pool at test time. Based on this framework, **we propose an unsupervised, graph-based selective annotation method, vote-k, to select diverse, representative examples to annotate.** Extensive experiments on 10 datasets (covering classification, commonsense reasoning, dialogue, and text/code generation) demonstrate that our selective annotation method improves the task performance by a large margin. On average, vote-k achieves a 12.9%/11.4% relative gain under an annotation budget of 18/100, as compared to randomly selecting examples to annotate. Compared to state-of-the-art supervised finetuning approaches, it yields similar performance with 10-100× less annotation cost across 10 tasks. We further analyze the effectiveness of our framework in various scenarios: language models with varying sizes, alternative selective annotation methods, and cases where there is a test data domain shift. We hope that our studies will serve as a basis for data annotations as large language models are increasingly applied to new tasks.

利用LLM的ICL能力，为每个instance寻找相似的demonstrations能够极大的提升性能。但是这潜在的要求存在一个规模较大的标注数据能够提供上下文，之前很少有工作讨论LLM的ICL需要的demonstrations的标注代价。

作者提出，不需要为很多的data进行标注而作为候选demonstrations，可以选择*representativeness*和*diversity*的少量无标注data进行标注：

- representativeness will help many test instances to find similar demonstrations
- diversity increases the total coverage

作者提出的方法vote-k来选择合适的无标注数据进行标注：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230924223545290.png"   style="zoom:50%;" />

步骤：

- 首先，通过Sentence-BERT将所有的无标注data编码为embedding，然后计算一个data point和其它data point的余弦相似度，选择$k$个最相似的data points（实验中发现$k=150$通常比较合适），创造一条有向边，构造出一个graph $G=(V,E)$；

- 然后，有两个集合$\mathcal{L}$代表需要被标注的data，$\mathcal{U}$代表剩下的无标注data。初始$\mathcal{L}=\empty$，然后迭代的从$\mathcal{U}$中选择degree score最大的一个data point $u$加入到$\mathcal{L}$中：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230924224045081.png"   style="zoom:50%;" />

  上面的公式通过超参$\rho$降低了和已经被标注的data相似的无标注data的权重（实验中发现$\rho=10$通常比较合适），从而鼓励多样性；选最大的score鼓励代表性。迭代一直进行$M/10$次；

- 为了进一步增加多样性，以标注好的$M/10$个data作为上下文样例，使用LLM为剩余的无标注数据进行预测，并且按照预测probability排序的划分为$M$个buckets，选择前$9M/10$个buckets的结果加入到$\mathcal{L}$中，不选择那些可以被最confident预测出来的无标注数据，最终$|\mathcal{L}|=M$；

在实验中，对于这些选择出来的有标注数据，还是使用基于相似度的kNN方法作为demonstrations，提升LLM的ICL能力。

##  Distilling Step-by-Step

Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

华盛顿大学与Google，ACL 2023 Findings，[代码](https://github.com/google-research/distilling-step-by-step)。

> Deploying large language models (LLMs) is challenging because they are memory inefficient and compute-intensive for practical applications. In reaction, researchers train smaller task-specific models by either fine-tuning with human labels or distilling using LLM-generated labels. However, finetuning and distillation require large amounts of training data to achieve comparable performance to LLMs. We introduce Distilling step-by-step, a new mechanism that (a) trains smaller models that outperform LLMs, and (b) achieves so by leveraging less training data needed by finetuning or distillation. **Our method extracts LLM rationales as additional supervision for training small models within a multi-task framework.** We present three findings across 4 NLP benchmarks: First, compared to both finetuning and distillation, our mechanism achieves better performance with much fewer labeled/unlabeled training examples. Second, compared to few-shot prompted LLMs, we achieve better performance using substantially smaller model sizes. Third, we reduce both the model size and the amount of data required to outperform LLMs; our finetuned 770M T5 model outperforms the few-shot prompted 540B PaLM model using only 80% of available data on a benchmark, whereas standard finetuning the same T5 model struggles to match even by using 100% of the dataset.

作者从540B的PaLM中导出rationales，然后让小模型T5学会输出对应的rationales。

部署大模型需要很高的代价：

> Serving a single 175 billion LLM requires at least 350GB GPU memory using specialized infrastructure (Zheng et al., 2022). To make matters worse, today’s state-of-the-art LLMs are composed of over 500B parameters (Chowdhery et al., 2022), requiring significantly more memory and compute.

为了避免这一点，有两种做法，一种是传统的在下游task的现有数据集上进行训练，这要求有人工标注；另一种是用LLM为无标注的data进行标注，蒸馏到小模型上，这种做法需要的大规模无标注数据可能是比较难的requires large amounts of unlabeled data which can be hard to obtain (Tang et al., 2019; Liang et al., 2020).

作者提出的Distilling Step-by-Step方法是通过导出LLM的推理rationales，减少小模型训练所需要的训练数据量：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925110839590.png"   style="zoom:30%;" />

rationales提供了更多的为什么input能够被输出为output的信息，并且是通常很难仅仅从input上找到的：

> Intuitively, rationales provide richer, more detailed information about why an input is mapped to a specific output label, and often contain relevant task knowledge that may be hard to infer solely from the original inputs.

rationale可以用来做什么？这一点已经有比较多的研究工作：
1. rationale可以用来规范model behavior；human rationales can be used to regularize model behavior (Ross et al., 2017)
2. rationale可以用来作为输入的一部分，提高预测性能；it can be used as additional inputs to guide a model’s predictions (Rajani et al., 2019); it can be used to improve overall model performance (Zaidan et al., 2007; Zhang et al., 2016; Camburu et al., 2018;Hancock et al., 2019; Pruthi et al., 2022);
3. rationale可以直接作为目标输出的一部分，提升模型预测的可解释性；human rationales can be used as gold standard labels to make models more interpretable by generating similar rationales (Wiegreffe et al., 2021; Narang et al., 2020; Eisenstein et al., 2022).

作者的方法很简单，第一步就是用人工写的CoT prompting模板从LLM中导出rationales；第二步是将rationales作为预测objective，用多任务学习机制加入一个额外的loss，让小模型学会输出rationales：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925110326813.png"   style="zoom:50%;" />

导出rationales的prompt如下：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925110457736.png"   style="zoom:40%;" />

将rationales看做是一种多任务，而不是和原来的label拼接在一起进行输出是必要的：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925110711281.png"   style="zoom:50%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925110741082.png"   style="zoom:50%;" />

作者在实验中发现，如果是用单任务的方法让模型同时输出label和rationales，有时候会反而减低性能，这一点在前人的工作中也有发现相似的规律。

## Data Generation for clinical NER and RE

Does Synthetic Data Generation of LLMs Help Clinical Text Mining?

arXiv 2023-04

> Recent advancements in large language models (LLMs) have led to the development of highly potent models like OpenAI’s ChatGPT. These models have exhibited exceptional performance in a variety of tasks, such as question answering, essay composition, and code generation. However, their effectiveness in the healthcare sector remains uncertain. **In this study, we seek to investigate the potential of LLMs to aid in clinical text mining by examining their ability to extract structured information from unstructured healthcare texts, with a focus on biological named entity recognition and relation extraction.** However, our preliminary results indicate that employing LLMs directly for these tasks resulted in poor performance and raised privacy concerns associated with uploading patients’ information to the LLM API. To overcome these limitations, we propose a new training paradigm that involves generating a vast quantity of high-quality synthetic data with labels utilizing LLMs and fine-tuning a local model for the downstream task. Our method has resulted in significant improvements in the performance of downstream tasks, improving the F1-score from 23.37% to 63.99% for the named entity recognition task and from 75.86% to 83.59% for the relation extraction task. Furthermore, **generating data using LLMs can significantly reduce the time and effort required for data collection and labeling, as well as mitigate data privacy concerns.** In summary, the proposed framework presents a promising solution to enhance the applicability of LLM models to clinical text mining.

作者先是尝试了ChatGPT在clinical NER和RE任务上，zero-shot ICL设置下和目前SOTA的差距：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925170021648.png"   style="zoom:50%;" />

在clinical NER和RE上，作者发现效果并不好，这当然很正常，ChatGPT并不是专门为clinical domain训练的，而执行这一domain肯定需要大量的domain knowledge；同时直接调用LLM的API存在隐私泄露问题。因此作者尝试利用LLM去生成一系列的训练数据，而不是直接进行任务。用LLM生成数据去训练一个小模型，小模型可以直接本地部署，避免了隐私泄露问题。

作者用prompt engineering创造合适的prompt：

- 询问GPT “Provide five concise prompts or templates that can be used to generate data samples of [Task Descriptions].”
- 用每个prompt生成10个句子，然后人工检查下句子质量，选择效果最好的prompt
- 然后让GPT基于前面选择的最好的prompt，继续提供新的prompt。这一过程持续3遍

作者找到的最合适的prompt（没有demonstrations）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925170508864.png"   style="zoom:50%;" />

NER任务是根据entity直接生成句子；RE任务是输入头尾实体，判断某个relation是否存在

可视化结果显示，不控制的情况下，GPT自己发挥生成的句子和原来的sentence肯定有分布上的差别：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925170635577.png"   style="zoom:50%;" />

## LM-CPPF

LM-CPPF: Paraphrasing-Guided Data Augmentation for Contrastive Prompt-Based Few-Shot Fine-Tuning

德兰黑大学与Google，ACL 2023 Short Paper，[代码](https://github.com/AmirAbaskohi/LM-CPPF)。

> In recent years, there has been significant progress in developing pre-trained language models for NLP. However, these models often struggle when fine-tuned on small datasets. To address this issue, researchers have proposed various adaptation approaches. Promptbased tuning is arguably the most common way, especially for larger models. Previous research shows that adding contrastive learning to prompt-based fine-tuning is effective as it helps the model generate embeddings that are more distinguishable between classes, and it can also be more sample-efficient as the model learns from positive and negative examples simultaneously. One of the most important components of contrastive learning is data augmentation, but unlike computer vision, effective data augmentation for NLP is still challenging. **This paper proposes LM-CPPF, Contrastive Paraphrasing-guided Prompt-based Fine-tuning of Language Models, which leverages promptbased few-shot paraphrasing using generative language models**, especially large language models such as GPT-3 and OPT-175B, for data augmentation. Our experiments on multiple text classification benchmarks show that this augmentation method outperforms other methods, such as easy data augmentation, back translation, and multiple templates.

作者用LLM改写原有的句子，利用有监督对比学习拉近改写前和改写后的小模型的特征距离。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925214427179.png"   style="zoom:50%;" />

- 第一步是对于要改写的target sentence，mask掉label，然后随机采样几个样例作为demonstrations拼接为一个text。作者微调的小模型是Roberta-base，用其计算被mask的label的loss，Masked Language Modeling (MLM) loss；
- 第二步是利用GPT-3或者OPT改写target sentence，作者尝试了几种不同的方法，发现了下面最合适的instruction：`Generate a paraphrase of the following text using different words and sentence structures while still conveying the same meaning`和demonstration format：`<Original Text>, in other words <Paraphrased>`。改写后的句子在图中对应的是Sent_3。改写的句子同样随机采样几个demonstrations，作为增强数据；
- 第三步是增强的数据同样输入到Roberta-base中，利用Supervised Contrastive Learning loss [*Supervised Contrastive Learning. NeurIPS 2020*]去优化；

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925215531981.png"   style="zoom:50%;" />

## CoTAM

Generating Efficient Training Data via LLM-based Attribute Manipulation

University of California，arXiv 2023-07，[代码](github.com/KomeijiForce/CoTAM)。

> In this paper, we propose a novel method, Chain-of-Thoughts Attribute Manipulation (CoTAM), to guide few-shot learning by carefully crafted data from Large Language Models (LLMs). The main idea is to create data with changes only in the attribute targeted by the task. Inspired by facial attribute manipulation, **our approach generates label-switched data by leveraging LLMs to manipulate task-specific attributes and reconstruct new sentences in a controlled manner.** Instead of conventional latent representation controlling, we implement chainof-thoughts decomposition and reconstruction to adapt the procedure to LLMs. Extensive results on text classification and other tasks verify the advantage of CoTAM over other LLMbased text generation methods with the same number of training examples. Analysis visualizes the attribute manipulation effectiveness of CoTAM and presents the potential of LLMguided learning with even less supervision.

作者从LLM部署所需要的巨大资源出发，考虑用数据增强训练一个更好的小模型。作者认为之前的LLM数据增强方法没有考虑可控的文本生成，可能影响了生成数据的信息有效性以及可能包含spurious correlation，variance比较大，让小模型比较难去学习

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925233300574.png"   style="zoom:30%;" />

作者收到了attribute manipulation in computer vision (Shen et al., 2020; Shen and Zhou, 2021)方法的启发，这些方法通过修改latent space中的embedding，重新生成新的image。作者提出，使用LLM可以为text生成不同的attributes，通过修改attributes，让生成的text的label发生改变：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925233503184.png"   style="zoom:50%;" />

主要就是利用LLM进行3步prompt：

第一步是让label作为一个attribute，然后让LLM提出更多的其它attributes：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925233545637.png"  style="zoom:50%;" />

第二步是只改变label attribute，让LLM提出能够包括这些attributes句子的方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925233603612.png" style="zoom:50%;" />

第三步是让LLM根据上面的方法，写一个新句子：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925233627575.png"   style="zoom:50%;" />

作者生成句子的实例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230925233913007.png"   style="zoom:50%;" />

## $\mbox{S}^2$ynRE

S2ynRE: Two-stage Self-training with Synthetic data for Low-resource Relation Extraction

中科大，ACL 2023，[代码](https: //github.com/BenfengXu/S2ynRE)。

> Current relation extraction methods suffer from the inadequacy of large-scale annotated data. While distant supervision alleviates the problem of data quantities, there still exists domain disparity in data qualities due to its reliance on domain-restrained knowledge bases. In this work, **we propose S2ynRE, a framework of two-stage Self-training with Synthetic data for Relation Extraction.** We ﬁrst leverage the capability of large language models to adapt to the target domain and automatically synthesize large quantities of coherent, realistic training data. We then propose an accompanied two-stage self-training algorithm that iteratively and alternately learns from synthetic and golden data together. We conduct comprehensive experiments and detailed ablations on popular relation extraction datasets to demonstrate the effectiveness of the proposed framework. Code is available at https: //github.com/BenfengXu/S2ynRE.

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

## PVI

Selective In-Context Data Augmentation for Intent Detection using Pointwise V-Information

EACL 2023，台湾大学与Amazon

> This work focuses on in-context data augmentation for intent detection. Having found that augmentation via in-context prompting of large pretrained language models (PLMs) alone does not improve performance, **we introduce a novel approach based on PLMs and pointwise Vinformation (PVI), a metric that can measure the usefulness of a datapoint for training a model.** Our method first fine-tunes a PLM on a small seed of training data and then synthesizes new datapoints – utterances that correspond to given intents. It then employs intent-aware filtering, based on PVI, to remove datapoints that are not helpful to the downstream intent classifier. Our method is thus able to leverage the expressive power of large language models to produce diverse training data. Empirical results demonstrate that our method can produce synthetic training data that achieve state-of-the-art performance on three challenging intent detection datasets under few-shot settings (1.28% absolute improvement in 5-shot and 1.18% absolute in 10-shot, on average) and perform on par with the state-of-the-art in full-shot settings (within 0.01% absolute, on average).

作者利用ICL基于OPT-66B生成intent classification task的更多数据：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230926205005892.png" style="zoom:35%;" />

然后，为了选择什么样的生成数据适合用来训练小模型，作者提出了一种选择模型生成data的方法，利用Pointwise V-Information (PVI) [*Understanding dataset difficulty with V-usable information. ICML 2022*]：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230926205115177.png" style="zoom:50%;" />

也就是有没有用data $x$微调前后，预测分布class $y$的概率的变化。如果PVI score越大，代表当前这一data对于class $y$的信息越多，越应该被挑选出来。

作者为每个intent class在validation set下测试出来PVI的平均值，作为PVI threshold，大于阈值的生成数据才会被使用。

作者在附录中，还尝试了使用data cartography技术[*Dataset cartography: Mapping and diagnosing datasets with training dynamics. EMNLP 2020*]和classification uncertainty相关metric去选择生成的数据，但是发现还是直接应用全部的生成数据效果最好：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230926210132984.png" style="zoom:30%;" />
