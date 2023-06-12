---
title: LM-as-KB-Survey
notshow: false
date: 2023-03-08 19:39:10
categories:
- Paper
- NLP
tags:
- NLP
- Language Model
- Survey
---

# A Review on Language Models as Knowledge Bases

arxiv 2022，Meta AI的一篇关于把预训练语言模型看做是知识库KB的综述。

> Recently, there has been a surge of interest in the NLP community on the use of pretrained Language Models (LMs) as Knowledge Bases (KBs). Researchers have shown that LMs trained on a sufﬁciently large (web) corpus will encode a signiﬁcant amount of knowledge implicitly in its parameters. The resulting LM can be probed for different kinds of knowledge and thus acting as a KB. This has a major advantage over traditional KBs in that this method requires no human supervision. In this paper, we present a set of aspects that we deem an LM should have to fully act as a KB, and review the recent literature with respect to those aspects.1

<!--more-->

## 1. Introduction

在大规模语料上训练得到的语言模型LM已经展现出了拥有以下几种知识：

- world knowledge
- relational knowledge
- commonsense knowledge
- linguistic knowledge
- actionable knowledge

但是LM的知识是隐式的，导致它的知识可访问性和可解释性都比较差。和之相对的，知识库knowledge base使用显式的符号知识表达现实世界，它的知识访问、更新和可解释性都更强。因此，研究者开始探究怎么样能够像KB一样控制LM中学习到的知识呢？该问题在论文《Language Models as Knowledge Bases? 》（EMNLP 2019）中首次提出。

这篇survey从LM-as-KBs的6个角度进行了综述：Access, Edit, Consistency, Reasoning, and Explainability and Interpretability。

## 2. Accessing Knowledge

KB可以使用查询语言或者查询工具很简单的找到对应的knowledge，但是对于LM来说要找到对应的knowledge更难。有两种主要的方式访问LM中的知识：finetuning和prompting。

### 2.1 Finetuning

在下游任务上直接微调预训练模型的参数，当然是最直接使用预训练隐式知识的方法。有研究者发现，大多数的知识还是在预训练过程中学习到的，而finetuning仅仅是学习到了一个访问该知识的接口（*Analyzing commonsense emergence in few-shot knowledge models. 21*）。

### 2.2 Prompting

prompting是一种让任务来适应模型的方法，不需要改变LM的模型参数，只需要找到合适的任务提示。作者把prompting分为discrete prompt和soft prompt两种。

discrete prompting是指直接使用自然语言/token进行描述；

soft prompting是指使用词向量进行提示；有研究者发现soft prompts的表达能力比discrete prompting更强（*Learning how to ask: Querying lms with mixtures of soft prompts. 21*）。*之前没有仔细了解过prompting，此结论不确定是否正确*。

## 3. Consistency

一致性是LM作为KB要面临的重要挑战之一。在下面三个情况下都需要考虑一致性：

### 3.1 Paraphrase

改写paraphrase，相同的意思使用不同句子/词去表达。LM对于不同paraphrase prompt的输出结果应该是一致的。

> Bhagat and Hovy (2013) [*What Is a Paraphrase? 13*] deﬁne the term quasi-paraphrases as ‘sentences or phrases that convey approximately the same meaning using different words’.

### 3.2 Commonsense

LM的另一个一致性体现在对于学习到的知识的一致性。研究者发现LM可能对于negation词（如not）是不鲁棒的。比如一个LM能够同时学习到“Birds can fly”和“Birds cannot fly”两个矛盾知识（*Negated and misprimed probes for pretrained language models: Birds can talk, but cannot ﬂy. 20*）。

LM对于蕴含entailment知识也应该是一致的（个人理解，entailment就是指当我们提到了一个知识成立的时候，它内部包括的知识也应该都成立），比如蛇是脊椎动物“A viper is a vertebrate”蕴含了另一个知识蛇有大脑“A viper has a brain”。（*Do Language Models Have Beliefs? Methods for Detecting, Updating, and Visualizing Model Beliefs. 21*）

### 3.3 Multilingual

LM对于不同语言描述的同一个查询，应该给出相同的输出。

## 4. Model Editing

知识库中的知识很简单的就可以被修改和更新，但是LM学习到的知识要更新/编辑就比较困难了。De Cao等人提出一个editing方法应该满足以下三点：

- Generality：editing方法不应该局限在某个具体的LM模型
- Reliability：editing方法应该只影响要修改的知识，不能影响其它的知识。
- Consistency：editing方法修改之后，应该保证对于各种语义相同的输入给出相同的输出，也就是要保证前面说的一致性要求。这就要求editing方法既要修改不正确的隐式知识，和修改的事实关联的所有事实也需要被修改，同时其它的事实保持不变。

现在的editing方法主要有三类：

- finetuning：最粗暴的方法是直接让模型针对新的知识进行从头学习，但这由于LM的训练成本基本上是不现实的。另外一种方法是构造一个支持新知识的evidence collection，让模型进行学习。但是这种持续学习的方法，要特别注意灾难性遗忘的问题，也就是会迅速忘记之前的旧知识（由于所有的参数都要更新，也不知道哪个参数需要修改，修改幅度有多大）。
- hyper networks：另一种方法是通过训练一个外部网络，让它输出要修改知识所需要的weight shift，从而能够编辑知识（*Editing Factual Knowledge in Language Models. 21*）。
- direct editing：Meng等人提出可以直接把Transformer block看做是key-value对，通过追踪相关的weight，直接修改对应的weight即可（*Locating and editing factual knowledge in gpt. 22*）。

## 5. Reasoning

LM模型已经表现出了一定的推理能力，比如常识推理、自然语言推理、数学推理、归纳推理等等。并且如果输入些推理过程的提示，LM模型也可以模仿着给出推理过程。

但是LM到底有没有推理能力，还没有定论（*Chain of thought prompting elicits reasoning in large language models. 22*）。

## 6. Interpretability

作者区分了两个“可解释”：

- interpretability指对模型内部机理的探究；
- explainability指模型的输出是否可解释，比如让模型自己给出输出答案的原因，属于事后解释。

可解释性可能是影响大规模语言模型真正落地到实际应用中最大的问题了。研究者从不同的角度进行了探究，但是个人认为目前的进展还不能充分解释LM内部机理。

- Probing：研究者将LM内部的表示和外部属性进行关联，从而辅助理解LM到底学习到了什么信息。
- Attention：自注意力是Transformer中的重要组成，研究者对attention进行了许多探究，包括不同层attention学习到的模式有什么区别、同一层不同attention head学习到的模式有什么区别、attention在不同情况下会更加关注什么样的token输入等等。
- Mechanistic Interpretability：Elhage等人提出了一个解释Transformer的数学视角（*A mathematical framework for transformer circuits. 21*）。
- Causal tracing：Meng等人尝试追踪模型输出和参数之间的路径关联（*Locating and editing factual knowledge in gpt. 22*）。

## 7.  Explainability

作者提到，有人使用influence function来尝试提高输出的可解释性，对此方法不了解（*Explaining black box predictions and unveiling data artifacts through inﬂuence functions. 20*）。

注意力本身的结果也常常被用来提供可解释输出。有研究者认为attention可以提供必要的explainability，但是有研究者认为attention不能够提供真正的explainability（*Is attention interpretable? 19*，*Attention is not Explanation 19*）。另外有研究者认为这个和具体模型相关（*Attention is not not explanation. 19*）。
所以attention能否被用来作为输出解释的一部分，目前在学术界还有争议。

另外一种方式是直接让LM给出它们做决策的解释，比如可以让它指出输入文本中支持输出的fragment（*Probing across time: What does roberta know and when? 16*）。
