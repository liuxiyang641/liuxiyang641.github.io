---
title: LLM-instruction-tuning
published: true
date: 2023-06-03 16:52:40
categories:
- Paper
- LLM
- instruction-tuning
tags:
- LLM
- instruction-tuning
---

# instruction-tuning相关论文

<!--more-->

## Instruction-induction

Instruction Induction: From Few Examples to Natural Language Task Descriptions

Tel Aviv University和Meta，[代码](https://github.com/orhonovich/instruction-induction)。首次提出从几个demonstrations使用LLM自动生成task instruction的任务。

> Large language models are able to perform a task by conditioning on a few input-output demonstrations – a paradigm known as in-context learning. We show that language models can explicitly infer an underlying task from a few demonstrations by prompting them to generate a natural language instruction that fits the examples. **To explore this ability, we introduce the instruction induction challenge, compile a dataset consisting of 24 tasks, and define a novel evaluation metric based on executing the generated instruction.** We discover that, to a large extent, the ability to generate instructions does indeed emerge when using a model that is both large enough and aligned to follow instructions; InstructGPT achieves 65.7% of human performance in our execution-based metric, while the original GPT-3 model reaches only 9.8% of human performance. This surprising result suggests that instruction induction might be a viable learning paradigm in and of itself, where instead of fitting a set of latent continuous parameters to the data, one searches for the best description in the natural language hypothesis space.

之前的工作已经证明通过给定几个demonstrations，LLM能够模拟demonstrations对给定的query text产生类似的输出（即In-context learning）。作者进一步提出让LLM直接描述demonstrations表达的task是什么。

作者提出的进行Instruction Induction的prompt：

![image-20230603165654412](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603165654412.png)

作者创建了一个新的包括24个task的数据集，下面是示例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603170121068.png"  style="zoom:30%;" />

为了评估LLM进行指令归纳的效果，作者提出使用两个指标：

- BERTScores: 评估LLM生成的指令和人工设计的指令
- execution accuracy: 作者新提出的指标，意思是让LLM zero-shot的执行自身生成的task instruction，计算执行的效果。

在实验部分，是使用5个demonstrations进行归纳。作者发现InstructGPT才表现出了较好的instruction-induction能力，没有经过instruction-tuning的GPT-3没有表现出这种能力：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603170218546.png"   style="zoom:40%;" />

## Self-Instruct

ACL 2023，华盛顿大学，[代码](https://github.com/ yizhongw/self-instruct)。

{% post_link llm/Self-Instruct [个人详细博客] %}

> Large “instruction-tuned” language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce SELF-INSTRUCT, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on SUPER-NATURALINSTRUCTIONS, on par with the performance of InstructGPT 001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with SELF-INSTRUCT outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT 001 . SELF-INSTRUCT provides an almost annotation-free method for aligning pretrained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning.

人工生成instructions一方面代价很大，另一方面人工生成的instructions难以保证quantity, diversity, and creativity。

作者提出使用LLM从已有的task instruction出发，自动生成新的task instruction和对应的input-output，然后过滤掉不符合规则的新task instructions，再加入到已有的task instructions集合中。作者在这个自动构造的instruction data上fine-tuning GPT3，发现效果提升了33%，非常接近InstructGPT001的效果。

作者提出的方法：

![image-20230603150047353](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603150047353-20230603165425791.png)

首先，作者拥有一个task pool，包括175 tasks (1 instruction and 1 instance for each task)。这175个初始的task instructions都是由本文作者自己创建的。

然后，作者从task pool中随机抽取8个task instructions（6 are from the human-written tasks, and 2 are from the model-generated tasks）。下面是产生新task instruction的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603150335100-20230603165425854.png" style="zoom: 25%;" />

之后，作者使用LLM判断新产生的instruction是否是一个classification task（using 12 classification instructions and 19 non-classification instructions）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603150505630-20230603165425906.png"   style="zoom:25%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603150518579-20230603165425957.png" alt="image-20230603150518579" style="zoom:25%;" />

随后，对于新产生的task instruction，用LLM生成新的对应的instance。对于生成任务，作者先生成input，再生成output，作者称为Input-first Approach：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603150903068-20230603165426058.png"   style="zoom:25%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603150941339-20230603165426247.png"  style="zoom:25%;" />

对于分类任务，作者发现如果是先生成input，LLM总是会倾向于生成某一个label的输入。因此作者使用LLM先生成output label，再让LLM生成input，作者称为Output-first Approach：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603151018452-20230603165426327.png"   style="zoom:25%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603151030519-20230603165426380.png"   style="zoom:25%;" />

对于LLM生成的task instruction、input和output，需要通过一些规则过滤，比如：

- 只有当和已有的task instruction相似度全部比较低（$\mbox{ROUGE-L}< 0.7$）的时候，一个新task instruction会被添加到task pool里
- We also exclude instructions that contain some specific keywords (e.g., image, picture, graph) that usually can not be processed by LMs.
- When generating new instances for each instruction, we filter out instances that are exactly the same or those with the same input but different outputs.
- Invalid generations are identified and filtered out based on heuristics (e.g., instruction is too long or too short, instance output is a repetition of the input).

作者从原始的175个task出发，最后构造了5万多的task，并且差异性也比较大：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603151912820.png"   style="zoom:30%;" />

在SuperNI数据集上的实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230603152723289.png"   style="zoom:30%;" />

SuperNI数据集大多是已有的NLP任务，为了进一步评估模型在实际使用场景下的价值，作者人工创建了一个包括252 task的新数据集。
