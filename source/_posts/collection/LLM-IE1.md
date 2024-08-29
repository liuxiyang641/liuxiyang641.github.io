---
title: LLM-IE1
published: false
date: 2023-05-15 23:24:09
categories:
  - Paper
  - LLM
  - IE
tags:
  - LLM
  - IE
  - Collection
---

# 基于LLM的Information Extraction 1

基于LLM的信息抽取工作总结合集1。

<!--more-->

## filter-then-rerank

Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!. EMNLP 2023 Findings. 南洋理工. [代码](https://github.com/mayubo2333/LLM-IE). 

> Large Language Models (LLMs) have made remarkable strides in various tasks. However, whether they are competitive few-shot solvers for information extraction (IE) tasks and surpass fine-tuned small Pre-trained Language Models (SLMs) remains an open problem. This paper aims to provide a thorough answer to this problem, and moreover, to explore an approach towards effective and economical IE systems that combine the strengths of LLMs and SLMs. Through extensive experiments on eight datasets across three IE tasks, **we show that LLMs are not effective few-shot information extractors in general, given their unsatisfactory performance in most settings and the high latency and budget requirements.** However, we demonstrate that LLMs can well complement SLMs and effectively solve hard samples that SLMs struggle with. Building on these findings, **we propose an adaptive filter-then-rerank paradigm, in which SLMs act as filters and LLMs act as rerankers.** By utilizing LLMs to rerank a small portion of difficult samples identified by SLMs, our preliminary system consistently achieves promising improvements (2.1% F1-gain on average) on various IE tasks, with acceptable cost of time and money.

作者评估了以Codex（`code-davinci-002`，2023/03/03之前）为基准的LLM+in-context learning方法在信息抽取任务上的性能，对比了基于RoBERTa和T5小型语言模型的现有IE SOTA方法。下面是一个例子：

![image-20230515233558514](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230515233558514.png)

作者发现当执行one-shot任务时，LLM性能还可以；当训练样本数逐渐增加时，基于LLM的方法受限于输入长度限制以及预训练过程等因素，没有办法达到SOTA的IE性能。不过作者的demonstration是为每个label随机采样固定数量的示例，而不是给每个测试样例都有不同的demonstrations。

下面是作者的实验结果：

![image-20230515233212836](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230515233212836.png)

上图有两个发现：

- 当输入的label类型少的时候，如在CONLL03数据集只有4种label表现效果不错；而在更多的label数据集，比如MAVEN有168中event type，实际上LLM模型不能够很好的理解不同label的内在含义[*Large language models still can’t plan (a benchmark for llms on planning and reasoning about change. 2022*]。并且越多的label意味着需要越多越复杂的输入demos。
- 三种任务比较起来，在NER任务上表现还可以。

基于LLM的IE方法还有另一个重要问题是接口返回值很慢，特别是输入特别大的情况下需要的处理时间就更长了；而小的模型推理速度很快，具体可以参考论文中的Table 1。

作者提出，LLM模型可以用来解决更加hard的样本，去解决那些小的基于监督训练的模型无法很好预测的样本，这些hard sample可能需要external knowledge或者更复杂的reasoning能力，这些正好是LLM模型的长处。因此作者提出了使用小的模型Small Language Model（SLM）先进行训练后预测，对于比较简单的样本，直接使用SLM的输出结果；对于比较难预测的样本，输出几个预测得分在top-n的label，让LLM进行rerank，最后进行输出。

作者判断一个样本是否难以被基于SLM的模型进行训练的依据就是不同label score中最大score越小，表示越难判断这个样本。

下面是模型图，作者在实现自己的模型使用了InstructGPT（text-davinci-003）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230515234734532.png"   style="zoom:50%;" />

方法看起来比较简单，有一点可以注意下，作者把IE任务下可能要求的格式化的输出（比如三元组）转换为句子的形式，让LLM行去做multi-choice question，这样LLM模型可能可以更好的理解demos中的实例。

![image-20230515235129692](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230515235129692.png)

在few-shot的IE任务下平均F1提升了2.1%。

## CheatGPT IE Evaluation

Evaluating ChatGPT’s Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness

北大，arXiv 2023.04

> The capability of Large Language Models (LLMs) like ChatGPT to comprehend user intent and provide reasonable responses has made them extremely popular lately. In this paper, we focus on assessing the overall ability of ChatGPT using 7 fine-grained information extraction (IE) tasks. Specially, we present the systematically analysis by measuring ChatGPT’s performance, explainability, calibration, and faithfulness, and resulting in 15 keys from either the ChatGPT or domain experts. **Our findings reveal that ChatGPT’s performance in Standard-IE setting is poor, but it surprisingly exhibits excellent performance in the OpenIE setting, as evidenced by human evaluation.** In addition, our research indicates that ChatGPT provides high-quality and trustworthy explanations for its decisions. However, there is an issue of ChatGPT being overconfident in its predictions, which resulting in low calibration. Furthermore, ChatGPT demonstrates a high level of faithfulness to the original text in the majority of cases. We manually annotate and release the test sets of 7 finegrained IE tasks contains 14 datasets to further promote the research. The datasets and code are available at this url.

这篇论文同样是讨论基于LLM的IE，只不过作者是基于ChatGPT，也没有使用更多的技术，比如上面论文的in-context learning。作者从Performance，Explainability，Calibration（模型对于输出结果的自信程度）和Faithfulness（输出结果是否与输入内容一致）四个大的方面，用15个指标（人工+自动）进行了评估：

![](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230515235818750.png)

作者使用了两种场景下的IE：

- Standard IE：给定label set
- Open IE：不给定label set，让ChatGPT自己回答，人工判断答案是否正确

为了避免历史回答记录的影响，每次回答都会清空上一次回答的记录，下面是作者进行事件检测任务时输入的样例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516000754165.png"   style="zoom:25%;" />

最终实验结果如下：

![image-20230516000110009](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516000110009.png)

在entity typing任务下表现不错。注意一下，里面的relation extraction任务实际上是实体-关系联合抽取任务，relation classification任务是只预测relation的任务。总体上LLM和SOTA还有较大差距，但是作者进一步发现如果是计算top-k指标的话，效果还不错：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516000316332.png"   style="zoom:40%;" />

而如果在open IE场景下，ChatGPT效果会更好：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516000408562.png"   style="zoom:40%;" />

对于可解释性，作者发现ChatGPT能够给出不错的解释（具体结果参考论文的Table 4）。

对于Calibration，发现ChatGPT不论是否分类正确，总是对自己的结果很自信，给出很高的得分。

最后，作者发现ChatGPT输出结果基本上和输入是相符的。

## ChatIE

Zero-Shot Information Extraction via Chatting with ChatGPT

北交，arXiv 2023.02，[代码](https://github.com/cocacola-lab/ChatIE)。

> Zero-shot information extraction (IE) aims to build IE systems from the unannotated text. It is challenging due to involving little human intervention. Challenging but worthwhile, zero-shot IE reduces the time and effort that data labeling takes. Recent efforts on large language models (LLMs, e.g., GPT3, ChatGPT) show promising performance on zero-shot settings, thus inspiring us to explore prompt-based methods. In this work, we ask whether strong IE models can be constructed by directly prompting LLMs. Specifically, we transform the zero-shot IE task into a multi-turn question-answering problem with a two-stage framework (ChatIE). With the power of ChatGPT, we extensively evaluate our framework on three IE tasks: entity-relation triple extract, named entity recognition, and event extraction. Empirical results on six datasets across two languages show that ChatIE achieves impressive performance and even surpasses some full-shot models on several datasets (e.g., NYT11-HRL). We believe that our work could shed light on building IE models with limited resources.

作者发现直接让ChatGPT问答IE任务效果不好，因此提出了两步的多轮问答方式的方法chatIE，目标是zero-shot IE任务，下面是模型图：

![image-20230516000929739](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516000929739.png)

第一步提问输入文本中有哪些类的可能，比如有哪些类实体；

第二步进一步提问每一类下的具体结果，这一步可能有多轮问答。

下面是NER任务的实例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516001252515.png"  style="zoom:25%;" />

实验结果：

![image-20230516001208421](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516001208421.png)

## CodeIE

CodeIE: Large Code Generation Models are Better Few-Shot Information Extractors. 复旦，ACL 2023，[代码](https://github.com/dasepli/CodeIE)。

> Large language models (LLMs) pre-trained on massive corpora have demonstrated impressive few-shot learning ability on many NLP tasks. A common practice is to recast the task into a text-to-text format such that generative LLMs of natural language (NL-LLMs) like GPT-3 can be prompted to solve it. However, it is non-trivial to perform information extraction (IE) tasks with NL-LLMs since the output of the IE task is usually structured and therefore is hard to be converted into plain text. In this paper, we propose to recast the structured output in the form of code instead of natural language and utilize generative LLMs of code (Code-LLMs) such as Codex to perform IE tasks, in particular, named entity recognition and relation extraction. In contrast to NL-LLMs, **we show that Code-LLMs can be well-aligned with these IE tasks by designing code-style prompts and formulating these IE tasks as code generation tasks.** Experiment results on seven benchmarks show that our method consistently outperforms fine-tuning moderate-size pre-trained models specially designed for IE tasks (e.g., UIE) and prompting NL-LLMs under few-shot settings. We further conduct a series of in-depth analyses to demonstrate the merits of leveraging Code-LLMs for IE tasks.

作者提出，基于LLM模型去做IE任务时，把输入和输出都转化为代码的形式更好，因为一般IE任务的输出是格式化的，而预训练模型很多是在自然语言上进行训练的；另外作者发现使用主要分析代码的LLM例如Codex效果比一般的LLM模型更好（作者实验中使用的还是code-davinci-002和text-davinci-002，不清楚上述结论对于003版本以及GPT-4是否成立）。

motivation：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516223705834.png"   style="zoom:30%;" />

作者提出的方法：

![image-20230516223400913](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516223400913.png)

主要是针对few-shot IE任务，加入了几个demonstration。固定的为每个entity/relation类型随机找k个样例作为demonstrations。定义的prompt是python的function格式，让Codex去补全剩下的代码。作者也试验了其它几个比如使用class init函数等，发现这样子效果最好。

- 使用python function表示IE任务
- 使用function docstring说明任务目标
- 待抽取的文本用string类型的变量表示
- 抽取出来的entity/relation使用不断append到list类型的变量表示

作者的实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230516223639496.png"   style="zoom:30%;" />

text prompt和code prompt效果对比：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230621155157156.png"   style="zoom:40%;" />

## Code4Struct

Code4Struct: Code Generation for Few-Shot Event Structure Prediction

ACL 2023, [代码](https://github.com/xingyaoww/code4struct)。

> Large Language Model (LLM) trained on a mixture of text and code has demonstrated impressive capability in translating natural language (NL) into structured code. We observe that semantic structures can be conveniently translated into code and propose CODE4STRUCT to leverage such text-tostructure translation capability to tackle structured prediction tasks. As a case study, **we formulate Event Argument Extraction (EAE) as converting text into event-argument structures that can be represented as a class object using code.** This alignment between structures and code enables us to take advantage of Programming Language (PL) features such as inheritance 1 and type annotation 2 to introduce external knowledge or add constraints. We show that, with sufficient in-context examples, formulating EAE as a code generation problem is advantageous over using variants of text-based prompts. Despite only using 20 training event instances for each event type, Code4Struct is comparable to supervised models trained on 4,202 instances and outperforms current state-of-the-art (SOTA) trained on 20-shot data by 29.5% absolute F1. By leveraging the inheritance feature of PL, Code4Struct can use 10-shot training data from a sibling event type to predict arguments for zero-resource event types and outperforms the zero-shot baseline by 12% absolute F1.

作者提出把EE抽取任务转化为代码的形式，一方面代码语言天然的能够描述复杂的事件本体，一方面在zero-shot设置下，代码生成的结果能够更加严格的遵循prompt的规定。

EAE抽取和python代码对应的表格：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627155654622.png"   style="zoom:50%;" />

方法图：

![image-20230627155624222](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627155624222.png)

固定的为每个事件类型找出现频率最多的k个event作为demonstrations。

code-prompt也是可以用自然语言的形式进行描述的，只不过更加的繁琐，并且难以在zero-shot设置下保证模型输出是满足要求的：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627155855556.png"   style="zoom:30%;" />

实验结果：

![image-20230627155927744](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627155927744.png)

在text-davinci-003下，拥有了足够多的样例，code-prompt和text-prompt差别不大。

另外，作者在实验的时候发现，使用同层级具有相同parent event的事件作为demonstrations，效果更好。这就提供了一种可能，也就是用样例比较多的event type，去作为样例比较少的event type的demonstrations。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627160039905.png"   style="zoom:50%;" />



## ChatGPT for KG

Enhancing Knowledge Graph Construction Using Large Language Models.

arXiv 2023

> The growing trend of Large Language Models (LLM) development has attracted significant attention, with models for various applications emerging consistently. However, the combined application of Large Language Models with semantic technologies for reasoning and inference is still a challenging task. This paper analyzes how the current advances in foundational LLM, like ChatGPT, can be compared with the specialized pretrained models, like REBEL, for joint entity and relation extraction. To evaluate this approach, we conducted several experiments using sustainability-related text as our use case. We created pipelines for the automatic creation of Knowledge Graphs from raw texts, and our findings indicate that using advanced LLM models can improve the accuracy of the process of creating these graphs from unstructured text. Furthermore, we explored the potential of automatic ontology creation using foundation LLM models, which resulted in even more relevant and accurate knowledge graphs.

作者使用ChatGPT去完整的构造一个与可持续相关的知识图谱，包括实体关系抽取与本体创建等，作者发现让ChatGPT去关联抽取出的实体关系与本体，效果不好。让ChatGPT直接生成本体可能是更合适的方式。

论文中没有提供具体的prompt设计方法，参考价值不大。

## VicunaNER

arXiv 2023.05，新加坡国立大学（截止05/18日还只能看到不太完整的论文，没有实验结果）

> Large Language Models (LLMs, e.g., ChatGPT) have shown impressive zero- and fewshot capabilities in Named Entity Recognition (NER). However, these models can only be accessed via online APIs, which may cause data leak and non-reproducible problems. In this paper, we propose VicunaNER, a zero/fewshot NER framework based on the newly released open-source LLM – Vicuna. VicunaNER is a two-phase framework, where each phase leverages multi-turn dialogues with Vicuna to recognize entities from texts. We name the second phase as Re-Recognition, which recognizes those entities not recognized in the first phase (a.k.a. Recongition). Moreover, we set entity correctness check dialogues in each phase to filter out wrong entities. We evaluate VicunaNER’s zero-shot capacity on 10 datasets crossing 5 domains and few-shot capacity on Few-NERD. Experimental results demonstrate that VicunaNER achieves superior performance in both shot settings. Additionally, we conduct comprehensive investigations on Vicuna from multiple perspectives.

作者基于羊驼模型进行NER任务，作者选择open LLM的理由如下：

- 数据集泄露问题，比如三星的敏感数据被泄露到了ChatGPT
- 不可复现问题，在线闭源的LLM都在持续更新，很难重复前人的研究结果

方法主要是基于多轮问答的NER，具体而言是有四步：

1. 让Vicuna抽取entity
2. 询问Vicuna抽取出的entity是否正确，过滤掉不正确的实体（第一阶段结束）
3. 给定上一阶段抽取到的实体，让Vicuna继续识别未识别出的实体
4. 询问Vicuna抽取出的entity是否正确，过滤掉不正确的实体（第二阶段结束）

再加入更多轮的问答作者发现并没有明显提升性能。

下面是方法图：

![image-20230518111407753](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230518111407753.png)

从这里可以体会下LLM的优点，作者的两个阶段相当于是输入不同的小任务，对于LLM模型来说没有区别，它可以直接执行这两个任务。相比起来，传统的模型更加specific，很难达到这样的泛化性，通过集成几个不同的小任务提升大任务的效果，而且不需要分别训练模型。（上面的工作思想核心应该说出是boost？）。

## CodeKGC

CodeKGC: Code Language Model for Generative Knowledge Graph Construction. 浙大zjunlp，ACM Transactions on Asian and Low-Resource Language Information Processing 2024，[代码](https://github.com/zjunlp/DeepKE/tree/main/example/llm)。

> Current generative knowledge graph construction approaches usually fail to capture structural knowledge by simply flattening natural language into serialized texts or a specification language. However, large generative language model trained on structured data such as code has demonstrated impressive capability in understanding natural language for structural prediction and reasoning tasks. Intuitively, we address the task of generative knowledge graph construction with code language model: given a code-format natural language input, the target is to generate triples which can be represented as code completion tasks. Specifically, **we develop schema-aware prompts that effectively utilize the semantic structure within the knowledge graph.** As code inherently possesses structure, such as class and function definitions, it serves as a useful model for prior semantic structural knowledge. Furthermore, we employ a rationale-enhanced generation method to boost the performance. Rationales provide intermediate steps, thereby improving knowledge extraction abilities. Experimental results indicate that the proposed approach can obtain better performance on benchmark datasets compared with baselines.

motivation：

作者认为对于知识图谱构建这样的任务来说，由于三元组之间存在依赖，互相关联，让语言模型直接生成结构化的输出比较难，因此作者将知识图谱信息抽取任务看做是代码生成任务，使用编程语言来描述输入的文本和输出，而不是使用自然语言。

method：

![image-20230519154317958](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230519154317958.png)

作者利用python语言描述prompt。输入的文本是作为python中的`Docstrings`。

schema的定义是通过python的`class`，作者定义了基础的类`Entity`，`Rel`和`Triple`。其它的实体和关系类会继承`Entity`和`Rel`。每一个三元组被定义为对应的`Triple`类，比如`(𝐿𝑜𝑛𝑑𝑜𝑛,𝑙𝑜𝑐𝑎𝑡𝑒𝑑𝑖𝑛,𝑈𝐾)`对应`Triple(LOC("London"), Rel("located in"), LOC("London"))`。

作者还另外提出了一个可选的Rationale-enhanced生成方法，也就是先抽取出关系，再抽取实体，最后抽取三元组。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230519155533998.png"   style="zoom:40%;" />

在实验部分作者实际也使用了`code-davinci-002`，但是作者提到由于Codex使用范围有限（OpenAI在3月23日停止了对Codex API的持续支持），因此作者仅仅在消融实验部分使用了Codex。

## InstructUIE

InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction

复旦，arXiv 2023.04，[代码](https://github.com/BeyonderXX/InstructUIE)。

>  Large language models have unlocked strong multi-task capabilities from reading instructive prompts. However, recent studies have shown that existing large models still have difficulty with information extraction tasks. For example, gpt-3.5-turbo achieved an F1 score of 18.22 on the Ontonotes dataset, which is significantly lower than the state-of-the-art performance. **In this paper, we propose InstructUIE, a unified information extraction framework based on instruction tuning, which can uniformly model various information extraction tasks and capture the inter-task dependency.** To validate the proposed method, we introduce IE INSTRUCTIONS, a benchmark of 32 diverse information extraction datasets in a unified text-to-text format with expert-written instructions. Experimental results demonstrate that our method achieves comparable performance to Bert in supervised settings and significantly outperforms the state-of-the-art and gpt3.5 in zero-shot settings.

在之前的一些研究中发现LLM模型在IE任务上表现并不好，因此作者希望能够实现一个基于LLM的unified information extraction (UIE) model。作者集合了现有的NER，RE和EE数据集，构造了一个加入instruction的benchmark——IE INSTRUCTION，用其来instruction-tuning LLM用于IE任务。

作者把IE任务看做是自然语言生成任务，一个text-to-text的任务。输入是带有instruction的prompt，输出是文本。下面是方法图：

![image-20230519170242805](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230519170242805.png)

方法比较简单，输入的prompt包括下面几部分：

- Task Instruction：type of information to be extracted, the format of the output structure, and any additional constraints or rules that need to be followed during the extraction process.
- Options：the set of possible outputs that can be generated by the model for a given input.
- Text：input sentence.

作者构建了一个基于信息抽取公开数据集的benchmark——IE INSTRUCTIONS，包括32个小的数据集，数据分布如下：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230519170702469.png"  style="zoom:30%;" />

作者对收集的数据集进行了以下处理：

- 统一不同数据集的label描述
- 把一些缩写或简写的标签转化为自然语言，比如`place_of_birth`转化为`place of birth`。
- 把所有数据集都转化为text-to-text的形式

同时，作者把每个IE任务进一步细分成为了不同的小任务：

- NER: span extraction and entity typing
- RE: entity pair extraction and relation classification
- EE: trigger extraction and argument extraction

作者的实验基于11B FlanT5，作者进行了有监督的在IE INSTRUCTIONS上微调LLM和无监督的zero-shot两种实验:

- Supervised Settings: 10,000 examples for each dataset
- Zero-shot Settings:
  - Train: 18 NER datasets and 6 RE datasets
  - Test: 7 NER datasets and 2 RE datasets

具体实验结果参看论文。

## InstructIE

InstructIE: A Chinese Instruction-based Information Extraction Dataset. 

浙大NLP，arXiv 2023，[代码](§https://github.com/zjunlp/KnowLM)。

> We introduce a new Information Extraction (IE) task dubbed Instructionbased IE, which aims to ask the system to follow specific instructions or guidelines to extract information. To facilitate research in this area, **we construct a dataset called InstructIE, consisting of 270,000 weakly supervised data from Chinese Wikipedia and 1,000 high-quality crowdsourced annotated instances.** We further evaluate the performance of various baseline models on the InstructIE dataset. The results reveal that although current models exhibit promising performance, there is still room for improvement. Furthermore, we conduct a comprehensive case study analysis, underlining the challenges inherent in the Instruction-based IE task.

作者构建InstructIE数据集的流程：

![image-20230627160605062](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627160605062.png)

统计情况：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627160631496.png"   style="zoom:40%;" />

进行指令微调后的结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627160708941.png"  style="zoom:50%;" />

表格中的CaMA就是浙大最近提出的智析Zhixi信息抽取大模型（LLaMA 13B based）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627160753724.png"   style="zoom:50%;" />

图中的KG2Instruction就是InstructIE数据集。

## GPT-3 for Clinical IE

Large Language Models are Few-Shot Clinical Information Extractors

EMNLP 2022，MIT，[代码](https://huggingface.co/datasets/mitclinicalml/clinical-ie)。

> A long-running goal of the clinical NLP community is the extraction of important variables trapped in clinical notes. However, roadblocks have included dataset shift from the general domain and a lack of public clinical corpora and annotations. **In this work, we show that large language models, such as InstructGPT (Ouyang et al., 2022), perform well at zero- and few-shot information extraction from clinical text despite not being trained specifically for the clinical domain.** Whereas text classification and generation performance have already been studied extensively in such models, here we additionally demonstrate how to leverage them to tackle a diverse set of NLP tasks which require more structured outputs, including span identification, token-level sequence classification, and relation extraction. Further, due to the dearth of available data to evaluate these systems, we introduce new datasets for benchmarking fewshot clinical information extraction based on a manual re-annotation of the CASI dataset (Moon et al., 2014) for new tasks 1 . On the clinical extraction tasks we studied, the GPT-3 systems significantly outperform existing zero- and few-shot baselines.

临床医学可能是一个能够体现LLM在IE任务中特有价值的具体场景。临床信息抽取任务一直面临下面的问题：

1. 文本中包括很多的专业术语和模糊的描述
2. 大多临床数据集不公开，即使公开了也有严格的使用限制，无法用于在线的OpenAI的LLM API

上面的问题在很多数据敏感的专业领域应该都是存在的。使用LLM的好处之一就是它可以不经过训练，在需要外部知识或者复杂推理能力的场景下达到还不错的效果。

作者探究了利用GPT-3进行临床数据的信息抽取的效果，同时重新标注了三个数据集以评估少次抽取性能。下面是方法，基本就是简单的ICL：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230520154610827.png"   style="zoom:30%;" />

使用`text-davinci-edit-001`作为实验对象。

## GPT-3 for Biomedical IE

Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again

EMNLP 2022 Findings，俄亥俄州立大学，[代码](https://github.com/dki-lab/few-shot-bioIE)。

> Large pre-trained language models (PLMs) such as GPT-3 have shown strong in-context learning capabilities, which are highly appealing for domains such as biomedicine that feature high and diverse demands of language technologies but also high data annotation costs. In this paper, **we present the first systematic and comprehensive study to compare the few-shot performance of GPT-3 in-context learning with fine-tuning smaller (i.e., BERT-sized) PLMs on two representative biomedical information extraction (IE) tasks: named entity recognition and relation extraction.** We follow the true few-shot setting (Perez et al., 2021) to avoid overestimating models’ few-shot performance by model selection over a large validation set. We also optimize GPT-3’s performance with known techniques such as contextual calibration and dynamic in-context example retrieval. However, **our results show that GPT-3 still significantly underperforms compared to simply fine-tuning a smaller PLM. In addition, GPT-3 in-context learning also yields smaller gains in accuracy when more training data becomes available.** More in-depth analyses further reveal issues of in-context learning that may be detrimental to IE tasks in general. Given the high cost of experimenting with GPT-3, we hope our study provides helpful guidance for biomedical researchers and practitioners towards more practical solutions such as fine-tuning small PLMs before better in-context learning is available for biomedical IE.

作者使用GPT-3进行生物医学领域的IE任务，主要使用ICL技术，发现GPT-3还不能够超越目前基于SLM的SOTA方法，同时往ICL中加入更多的demos并没有能够持续提升效果。

方法：

![image-20230520164636198](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230520164636198.png)

作者的方法主要是使用ICL技术，为了能够选择和当前样例相近的demos，作者基于RoBERTa-large作为编码器，使用kNN方法从100个固定的训练集样例集合中动态选择。NER最多选择10个样例，RE最多选择5个样例。

额外的两个实现细节：

- Logit Biases：调用OpenAI API的*logit bias*参数，给所有出现在原始句子中的tokens、作者选择的chosen separator和newline token增大概率。选择的logit bias值是$10$。
- Contextual Calibration：作者在前期实验里发现，选定了一组demonstrations后，不管test query是什么，这组demonstrations都有对于某个特定label的偏好bias。为了解决这一点，作者对于RE任务，采用了Contextual calibrating技术[*Calibrate Before Use: Improving Few-shot Performance of Language Models. ICML 2021*]。也就是用特殊的NULL token `N/A` 替换句子和头尾实体，得到null prompt输入LLM，获得其对于label的原始偏好分布，用其来更新RE抽取的分布。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230520164836464.png"   style="zoom:40%;" />

可以看到，在作者的实验环境下，使用GPT-3和当前的SOTA方法还是有差距。

## ChatGPT for ED

Exploring the Feasibility of ChatGPT for Event Extraction

arXiv 2023.03，哈工大-深圳。

> Event extraction is a fundamental task in natural language processing that involves identifying and extracting information about events mentioned in text. However, it is a challenging task due to the lack of annotated data, which is expensive and time-consuming to obtain. The emergence of large language models (LLMs) such as ChatGPT provides an opportunity to solve language tasks with simple prompts without the need for task-specific datasets and fine-tuning. While ChatGPT has demonstrated impressive results in tasks like machine translation, text summarization, and question answering, it presents challenges when used for complex tasks like event extraction. **Unlike other tasks, event extraction requires the model to be provided with a complex set of instructions defining all event types and their schemas.** To explore the feasibility of ChatGPT for event extraction and the challenges it poses, we conducted a series of experiments. **Our results show that ChatGPT has, on average, only 51.04% of the performance of a task-specific model such as EEQA in long-tail and complex scenarios.** Our usability testing experiments indicate that ChatGPT is not robust enough, and continuous refinement of the prompt does not lead to stable performance improvements, which can result in a poor user experience. Besides, ChatGPT is highly sensitive to different prompt styles.

属于对LLM的capacity evaluation，作者使用ChatGPT进行zero-shot的event detection任务。主要用的方法是ICL，下面是示例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230521231825422.png"   style="zoom:30%;" />

总体效果和SOTA还是有差距，下面是在ACE 2005数据集上的实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230521231945056.png"   style="zoom:30%;" />

不过作者在论文中提到了仅仅是使用了测试集中的20个sample进行了测试，这个结果可能不够准确。

不过一个有意思的是作者找了四NLP领域的研究生，让他们去5次改变prompt来获得更好的效果机会，测试样例一共有10个，下面是实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230521232252014.png"   style="zoom:30%;" />

同样，个人认为这个结果也不够鲁棒，比较测试样例只有10个，错1个样例就是10%的差距了。

## Wadhwa et al.

Revisiting Relation Extraction in the era of Large Language Models. Northeastern University，ACL 2023.

> Relation extraction (RE) is the core NLP task of inferring semantic relationships between entities from text. Standard supervised RE techniques entail training modules to tag tokens comprising entity spans and then predict the relationship between them. Recent work has instead treated the problem as a sequence-to-sequence task, linearizing relations between entities as target strings to be generated conditioned on the input. **Here we push the limits of this approach, using larger language models (GPT-3 and Flan-T5 large) than considered in prior work and evaluating their performance on standard RE tasks under varying levels of supervision.** We address issues inherent to evaluating generative approaches to RE by doing human evaluations, in lieu of relying on exact matching. Under this refined evaluation, we find that: (1) Few-shot prompting with GPT-3 achieves near SOTA performance, i.e., roughly equivalent to existing fully supervised models; (2) Flan-T5 is not as capable in the few-shot setting, but supervising and fine-tuning it with Chain-of-Thought (CoT) style explanations (generated via GPT3) yields SOTA results. We release this model as a new baseline for RE tasks.

作者这篇论文主要做了两个工作：

1. 测试并评估GPT-3对于RE任务的性能。由于作者发现GPT-3常常会产生和输入要求不一致的关系，因此作者还重新人工评估了效果GPT-3对RE任务的性能。作者发现在CONLL04和ADE数据集上可以达到接近SOTA的结果。
2. 作者通过使用GPT-3自动生成的explanations作为输入，通过微调FlanT5-large达到了新的SOTA。

作者测试GPT-3的输入如图：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524194415967.png"    style="zoom:40%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524194429292.png"   style="zoom:40%;" />

在测试的时候会随机采样12个examples作为固定的demonstrations。然后作者发现GPT-3会产生和输入不一致的输出relation，但是这些relation让人工去评估的话又会感觉在语义上是一致的。因此作者又人工重新评估了所有的输出结果（通过在Amazon Mechanical Turk平台上众包）。数据集的分布如下所示，ADE这个数据集是用10-fold交叉验证来进行评估。除NYT外，其它两个数据集测试量挺小的。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524194748501.png"   style="zoom:40%;" />

下面是作者的GPT-3实验结果（记住这里的GPT-3评估结果是由人工重新评估之后的，同时其它模型的输出并没有进行人工的评估，个人认为是不准确的）：

![image-20230524195229694](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524195229694.png)

上面的结果显示，GPT-3在NYT数据集上表现效果不好，这是因为NYT的关系类型太多，导致无法准确的描述NYT中不同关系类型。

作者进一步提出，可以使用GPT-3自动生成的解释作为CoT来进一步引导模型微调。作者先让GPT-3生成解释，然后用这些生成的解释输入到Flan-T5-large（760M），随后进行微调进一步可以提升Flan-T5-large的性能。下面是方法图：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524200126966.png"   style="zoom:40%;" />

作者在论文里把Flan-T5-large也叫做是LLM，个人认为不合适。

## QA4RE

Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors. 俄亥俄州立大学，ACL 2023 Findings。[代码](https://github.com/OSU-NLP-Group/QA4RE)。

> Recent work has shown that fine-tuning large language models (LLMs) on large-scale instruction-following datasets substantially improves their performance on a wide range of NLP tasks, especially in the zero-shot setting. However, even advanced instruction-tuned LLMs still fail to outperform small LMs on relation extraction (RE), a fundamental information extraction task. We hypothesize that instruction-tuning has been unable to elicit strong RE capabilities in LLMs due to RE’s low incidence in instruction-tuning datasets, making up less than 1% of all tasks (Wang et al., 2022). To address this limitation, **we propose QA4RE, a framework that aligns RE with question answering (QA), a predominant task in instruction-tuning datasets.** Comprehensive zero-shot RE experiments over four datasets with two series of instruction-tuned LLMs (six LLMs in total) demonstrate that our QA4RE framework consistently improves LLM performance, strongly verifying our hypothesis and enabling LLMs to outperform strong zero-shot baselines by a large margin. Additionally, we provide thorough experiments and discussions to show the robustness, few-shot effectiveness, and strong transferability of our QA4RE framework. This work illustrates a promising way of adapting LLMs to challenging and underrepresented tasks by aligning these tasks with more common instruction-tuning tasks like QA.

作者这篇工作的思想很简单，就是把relation选择转化为multi-choice options选择的QA问题。类似的做法在filter-then-rerank里有实现。

作者这么做的出发点是之前的研究发现LLM对于RE的效果不好，作者自己使用GPT-3.5和FlanT5进行了尝试发现同样效果不好。作者认为这样的原因是LLM模型在进行instruction tuning过程中，只有极少的样本可能涉及了RE任务。下面是一个统计结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230528105537993.png"   style="zoom:50%;" />

因此作者将RE任务的形式和在instruction tuning数据集中更常出现的QA任务形式对齐。下面是方法图：

![image-20230524234509972](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524234509972.png)

作者实现的部分细节：

- 使用SuRE方法[*Summarization as indirect supervision for relation extraction*]中提出的relation template来构造模板

- 使用`text-davinci-003`和`FLAN-T5-XXLarge`作为基座LLM

- 对于prompt engineering，作者使用text-davinci-002在TACRED的dev set上选择250个样例进行评估。然后对所有的测试数据集使用相同的prompt格式。以关系$org:top\_members/employees$为例，作者进行了四种模板的尝试（这四种模板也是前人的工作）：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235128751.png"   style="zoom:35%;" />

作者的zero-shot RE实验结果：

![image-20230524235216668](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235216668.png)

为了限制OpenAI的花费，作者从测试集采样了1000个例子评估基于text-davinci-003的效果。通过作者提出的简单prompt改动，就获得了平均8%左右的提升…

不过作者进一步在附录里提供了对于Flan-T5在整个测试集下的测试结果：

![image-20230528105952811](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230528105952811.png)

同样能够看到简单的改动带来了非常明显的提示。

下面是作者做的更多的探究实验，个人认为有一定参考价值。

对于4种prompt格式的实验：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235402066.png"   style="zoom:35%;" />

可以看到prompt的设计还是很关键的，怎么样找到合适的prompt引起了10%左右的偏差（即使在人类看来不同的relation option模板都是正确的）。

作者还额外做了few-shot实验，few-shot上表现的效果不好，特别是受到输入长度的限制不能持续的输入shot样例。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235554154.png"  style="zoom:35%;" />

另一个有参考意义的实验是作者在task instruction中加入了对于label的描述，而保持option还是缩略的relation：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235720898.png"  style="zoom:35%;" />

实验结果发现仅仅通过对label进行抽象的解释，不能够很好的提升LLM的回答。反而如本文提出的QA4RE一样把不同label的输出直接转化为具体的句子，LLM更容易理解。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235816398.png"  style="zoom:35%;" />

作者还比较了基于LLM的方法随着model size变化的性能变化：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230524235859346.png"   style="zoom:35%;" />

可以看到随着模型size的增大，效果越来越好。但是提升的幅度和size大小并不是成比例的

（整篇论文方法部分只讨论了1页左右，实验部分讨论了4页多）

## GPT-RE

GPT-RE: In-context Learning for Relation Extraction using Large Language Models. 京都大学. EMLNLP 2023. [代码](https://github.com/YukinoWan/GPT-RE).

> In spite of the potential for ground-breaking achievements offered by large language models (LLMs) (e.g., GPT-3), they still lag significantly behind fully-supervised baselines (e.g., fine-tuned BERT) in relation extraction (RE). This is due to the two major shortcomings of LLMs in RE: (1) **low relevance regarding entity and relation in retrieved demonstrations for in-context learning;** and (2) **the strong inclination to wrongly classify NULL examples into other pre-defined labels**.
>
> In this paper, we propose GPT-RE to bridge the gap between LLMs and fully-supervised baselines. GPT-RE successfully addresses the aforementioned issues by (1) incorporating task-specific entity representations in demonstration retrieval; and (2) enriching the demonstrations with gold label-induced reasoning logic. We evaluate GPT-RE on four widely-used RE datasets, and observe that GPT-RE achieves improvements over not only existing GPT-3 baselines, but also fully-supervised baselines. Specifically, GPT-RE achieves SOTA performances on the Semeval and SciERC datasets, and competitive performances on the TACRED and ACE05 datasets.

作者这篇工作是对于[*Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again*]工作的改进，主要是针对其两个问题进行改进：

- 在ICL中的demonstrations的选择只是从sentence-level进行比较，忽略了实体和关系的语义
- LLM很难准确地分类NULL关系

下面是示例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230527155614789.png"   style="zoom:35%;" />

作者提了两点改进：

- 通过修改prompt格式，让prompt更加强调句子中的实体信息；或者直接使用一个在RE任务上fine-tuned好的BERT模型来获取头尾实体的embedding；之后再进行基于kNN的demonstrations检索
- 加入了CoT，也就是让GPT自己生成一个解释，加入到ICL的demonstrations中。

方法图：

![image-20230527160412848](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230527160412848.png)

demonstrations检索方法的改进：

- 方案1：修改context格式，例如’He has a sister Lisa‘修改为’The relation between ‘He’ and ‘Lisa’ in the context: He has a sister Lisa.‘
- 方案2：更加直接，使用fine-tuned头尾实体表征来进行kNN检索。作者自己微调了一个针对RE任务的BERT来获取头尾实体表征。（一个简单粗暴，泛化性弱的方法，但是效果最好）

CoT中的explanations是使用GPT自动生成的，下面是一个示例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230528111844081.png"   style="zoom:40%;" />

作者使用SimCSE方法来衡量相似度。下面是实验用的数据集和实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230527214328082.png"   style="zoom:40%;" />

![image-20230527214353436](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230527214353436.png)

实验结果中（基于text-davinci-003），

- Random：随机找few-shot的demonstrations
- Sent：使用SimCSE
- RE_SimCSE: 方案1
- RE_FT：方案2

可以观察得到结论：

- 虽然总体表现出了不错的效果，但是可以看到需要样例数比较大（最少也在15个样例以上）才能达到SOTA，而且调用GPT的代价比较大（花钱，以至于作者在TACRED和ACE05这两个比较大test数据集下只选择了10%的样例）。
- 完全无训练无梯度更新的方案一，仍然没有达到SOTA。只有使用训练后的表征来检索，才能达到SOTA，并且提升幅度很大。
- 没有比较推理速度，个人认为推理速度不会快（检索+大模型）
- 加入CoT的效果提升幅度不是特别大，并且要求了额外的GPT请求，更加花钱（作者在实验部分对CoT的对比仅仅通过15个样例，而不是最好的样例数量，这是因为输入长度的限制）

下面是作者对于随着shot数量变化，模型性能的变化，可以看到提升还是很大的（最多有十几个点的提升）。同时从消融掉CoT（即reasoning）的效果来看，加入CoT（解释）的情况下，对于更少demonstrations的情况提升更明显（图中shot=30的时候没有对应的消融实验，由于输入长度限制）。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230527231519715.png"  style="zoom:35%;" />

## AutoKG

LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities

arXiv 2023.05，浙大ZJUNLP，[代码](https://github .com/zjunlp/AutoKG)。

> This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We employ eight distinct datasets that encompass aspects including entity, relation and event extraction, link prediction, and question answering. Empirically, our findings suggest that GPT-4 outperforms ChatGPT in the majority of tasks and even surpasses fine-tuned models in certain reasoning and question-answering datasets. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, which culminates in the presentation of the Virtual Knowledge Extraction task and the development of the VINE dataset. Drawing on these empirical findings, we further propose AutoKG, a multiagent-based approach employing LLMs for KG construction and reasoning, which aims to chart the future of this field and offer exciting opportunities for advancement. We anticipate that our research can provide invaluable 1 insights for future undertakings of KG.

调研时看到的首个使用GPT-4进行知识图谱相关任务的paper，可惜受限于GPT-4的访问代价，作者仅仅是对每个任务都进行了20个左右的测试样例的评估。发现GPT-4对于IE任务效果比ChatGPT要好，但是仍然和SOTA有差距，同时GPT-4更加擅长KG reasoning（linking prediction）和QA任务。

然后作者自己从RE-TACRED数据集中选择句子，使用随机创建的新词替换其中的实体和关系，构造了一个GPT-4在训练过程中没有见过的虚假数据集VINE，发现GPT-4确实是能够快速理解instruction去进行信息抽取。最后是作者借助CAMEL方法中提出的role-playing方法，提出了一个AutoKG的概念。

![image-20230529000001137](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230529000001137.png)

## structured prompting

Prompting Language Models for Linguistic Structure

ACL 2023，华盛顿大学

> Although pretrained language models (PLMs) can be prompted to perform a wide range of language tasks, **it remains an open question how much this ability comes from generalizable linguistic understanding versus surface-level lexical patterns.** To test this, we present a structured prompting approach for linguistic structured prediction tasks, allowing us to perform zero- and few-shot sequence tagging with autoregressive PLMs. We evaluate this approach on part-of-speech tagging, named entity recognition, and sentence chunking, demonstrating strong few-shot performance in all cases. We also find that while PLMs contain significant prior knowledge of task labels due to task leakage into the pre-training corpus, structured prompting can also retrieve linguistic structure with arbitrary labels. These findings indicate that the in-context learning ability and linguistic knowledge of PLMs generalizes beyond memorization of their training data.

作者提出了一种简单的序列标注prompt方法，就是在输出的每个word token之后加入要标注的label。作者提到了，在输出的时候不是直接输出所有的tag序列，而是同时要输出原有的word+tag。如果不重复输出word的话，效果甚至会下降70%-80%。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230531113232237.png"   style="zoom:40%;" />

基于GPT-NeoX、GPT-Curie、GPT-Davinci进行了实验。使用上下文学习，从数据集中随机找$k$个样例，只要保证这些样例能够覆盖所有的label即可。

有一点实验启发的是，作者发现在NER任务下，LLM也常常会错误的分类`O` label，和其它的研究发现RE任务常常错误分类`None`一样。这说明了这些比较模糊、或者内部语义分布比较多样的label，让LLM直接去做很可能准确度不高：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230531153548171.png"   style="zoom:40%;" />

作者还基于GPT-Neo的预训练数据Pile中，去查找有没有label数据，结果发现是有的：

![image-20230531153914033](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230531153914033.png)

## GPT-NER

GPT-NER: Named Entity Recognition via Large Language Models

北大，arXiv，[代码](https://github.com/ShuheWang1998/GPT-NER)。

> Despite the fact that large-scale Language Models (LLM) have achieved SOTA performances on a variety of NLP tasks, its performance on NER is still significantly below supervised baselines. This is due to the gap between the two tasks the NER and LLMs: the former is a sequence labeling task in nature while the latter is a text-generation model.
>
> In this paper, we propose GPT-NER to resolve this issue. GPT-NER bridges the gap by transforming the sequence labeling task to a generation task that can be easily adapted by LLMs e.g., the task of finding location entities in the input text Columbus is a city is transformed to generate the text sequence @@Columbus## is a city, where special tokens @@## marks the entity to extract. To efficiently address the hallucination issue of LLMs, where LLMs have a strong inclination to over-confidently label NULL inputs as entities, we propose a self-verification strategy by prompting LLMs to ask itself whether the extracted entities belong to a labeled entity tag.
>
> We conduct experiments on five widely adopted NER datasets, and GPT-NER achieves comparable performances to fully supervised baselines, which is the first time as far as we are concerned. More importantly, we find that GPT-NER exhibits a greater ability in the low-resource and few-shot setups, when the amount of training data is extremely scarce, GPT-NER performs significantly better than supervised models. This demonstrates the capabilities of GPT-NER in real-world NER applications where the number of labeled examples is limited.

方法图：

![image-20230627153804979](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627153804979.png)

作者认为NER focuses more on local evidence rather than a sentence-level task。比如要查询句子“John is a soldier”，从sentence-level相似性来看，“he is a soldier”是相似的。但是“he is a soldier”中都没事实体，可能寻找其它属于PERSON的实体应该合理。

![image-20230627154141754](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627154141754.png)

作者还额外进行了结果的验证：

![image-20230627154210391](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627154210391.png)

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627154237264.png"   style="zoom:40%;" />

从实验结果来看，不同检索方案之间的差异很大。而自我验证答案正确与否的实验结果，发现也就是提升了大概1个点左右。

## BertNet

BertNet: Harvesting Knowledge Graphs with Arbitrary Relations from Pretrained Language Models

ACL 2023 Findings, [代码](https://github.com/tanyuqian/knowledge-harvest-from-lms)。

> It is crucial to automatically construct knowledge graphs (KGs) of diverse new relations to support knowledge discovery and broad applications. Previous KG construction methods, based on either crowdsourcing or text mining, are often limited to a small predefined set of relations due to manual cost or restrictions in text corpus. Recent research proposed to use pretrained language models (LMs) as implicit knowledge bases that accept knowledge queries with prompts. Yet, the implicit knowledge lacks many desirable properties of a full-scale symbolic KG, such as easy access, navigation, editing, and quality assurance. In this paper, we propose a new approach of harvesting massive KGs of arbitrary relations from pretrained LMs. With minimal input of a relation definition (a prompt and a few shot of example entity pairs), the approach efficiently searches in the vast entity pair space to extract diverse accurate knowledge of the desired relation. We develop an effective search-and-rescore mechanism for improved efficiency and accuracy. We deploy the approach to harvest KGs of over 400 new relations from different LMs. Extensive human and automatic evaluations show our approach manages to extract diverse accurate knowledge, including tuples of complex relations (e.g., "A is capable of but not good at B"). The resulting KGs as a symbolic interpretation of the source LMs also reveal new insights into the LMs’ knowledge capacities.

直接从LLM抽取信息，方法图：

![image-20230627154802433](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627154802433.png)

评估不同prompt的公式：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627154830846.png"   style="zoom:40%;" />

$p$代码prompt，$h,t$代表头尾实体对。根据不同prompt生成的新的头尾实体，和每一个prompt匹配后，计算上面的公式获得打分，再进行排序。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627155033043.png"   style="zoom:50%;" />

表格中的accuracy是人工判断生成的信息三元组是否正确。relation set是指作者初始阶段使用的relation set集合来源。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230627155208099.png"   style="zoom:35%;" />

模型越大，抽取出来的信息正确程度越高。

## CoT-MIE

Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction

阿里Ant group，2023-08 arXiv

> Multimodal Named Entity Recognition (MNER) and Multimodal Relation Extraction (MRE) necessitate the fundamental reasoning capacity for intricate linguistic and multimodal comprehension. In this study, we explore distilling the reasoning ability of large language models (LLMs) into a more compact student model by generating a chain of thought (CoT) – a sequence of intermediate reasoning steps. Specifically, we commence by exemplifying the elicitation of such reasoning ability from LLMs through CoT prompts covering multi-grain (noun, sentence, multimodality) and data-augmentation (style, entity, image) dimensions. Subsequently, we present a novel conditional prompt distillation method to assimilate the commonsense reasoning ability from LLMs, thereby enhancing the utility of the student model in addressing text-only inputs without the requisite addition of image and CoT knowledge. Extensive experiments reveal that our approach attains state-of-the-art accuracy and manifests a plethora of advantages concerning interpretability, data efficiency, and cross-domain generalization on MNER and MRE datasets.

作者声称是希望能够将LLM的推理能力交给小模型，但是个人阅读下来感觉小模型也没有学会推理能力。并且这里一直在强调CoT，事实上这篇论文个人更愿意看做是一种数据增强/知识检索的方法，毕竟LLM本身没有针对信息抽取给出中间的推理步骤。

作者的做法出发点是：

- 之前的基于检索的模型，难以保证检索到的结果和查询的句子是匹配的，比如下图：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824231237542.png"   style="zoom:40%;" />

- 大模型的推理成本比较高，但是它的推理能力比较好。希望能够用个小模型学会大模型的推理能力，并且有较低的推理成本。

作者的方法：

![image-20230824231405165](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824231405165.png)

首先，作者用BLIP2把多模态信息抽取中的图片转化为文本caption。

然后利用LLM生成下面几种额外的知识：

- Noun：对于句子中的potential entities, slang, and terminology等名词进行查询，对应的prompt是：`Help me explain the meaning of special words for understanding. + x`
- Sentence：对于整个句子进行理解，It can explain the sentiment, cause, and subject of users. 对应的prompt是`Explain the sentence to me with necessary background. + x`
- Multimodality：让LLM解释潜在的image和text之间的关系，这一步可以用来去噪、潜在的对齐visual object和textual entity，对应的prompt是：`What is the relation between the text and the attached image? + x + I`

作者还利用LLM进行了数据增强：

- Style：利用LLM转换输入句子的风格，让文本的描述保持一致的风格，对应的prompt是`Transform the sentence in Twitter style without changing the meaning. + x`
- Entity：用同类型的entity替换候选的entity，然后用LLM判断替换后的伪样本是否成立，判断的prompt是`Whether the sentence is possible in fact, answer yes or no. + x`
- Image：让LLM猜测能够和文本描述对应的image长什么样子，对应的prompt是`What is a possible image with the text in a tweet? + x`

数据增强后的样本被看做是新的样本。

然后问题的关键是怎么样能够让小模型学会LLM的推理，作者声称提出了Conditional Prompt Distillation的方法。具体做法是首先作者把原始的text $x$、图像的caption $I$以及LLM生成的知识$c$拼接到一起，经过text encoder获得输出分布$H_k$；然后，作者定义了可学习的soft prompt来作为conditional prompt聚合text：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824232713196.png"   style="zoom:40%;" />

这里生成的$p$和原始的text $x$拼接在一起，经过text encoder获得输出分布$H_t$；最后，作者期望这两种分布是相近的：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824232832068.png"   style="zoom:40%;" />

个人对于这个公式有点疑惑，这里的分布到底是信息抽取的classification distribution还是token distribution？

更疑惑的是，最后预测结果仍然是加入了LLM生成知识$c$的结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824233139004.png"   style="zoom:40%;" />

难道是在测试阶段仅仅用小模型，不需要LLM提前处理？

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824233204415.png"   style="zoom:30%;" />

和目前的SOTA相比，MNRE数据集上还有10%的差距；而Twitter15和17数据集可以认为是达到了SOTA。

另外从消融的结果来看，对于名词的解释，可能作用相对比较大：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824233303269.png"   style="zoom:30%;" />

论文的case study可以看下，感觉这些LLM生成的knowledge还是比较有意义的，问题在于没有CoT..也不确定小模型是否学习到了推理能力：

![image-20230824233447694](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230824233447694.png)

## PGIM

Prompt ChatGPT In MNER: Improved multimodal named entity recognition method based on auxiliary refining knowledge from ChatGPT

天津大学，2023-05，arXiv

> Multimodal Named Entity Recognition (MNER) on social media aims to enhance textual entity prediction by incorporating image-based clues. Existing research in this domain has primarily focused on maximizing the utilization of potentially relevant information in images or incorporating external knowledge from explicit knowledge bases (KBs). However, **these methods either neglect the necessity of providing the model with relevant external knowledge, or the retrieved external knowledge suffers from high redundancy.** To address these problems, **we propose a conceptually simple two-stage framework called Prompt ChatGPT In MNER (PGIM) in this paper.** We leverage ChatGPT as an implicit knowledge engine to acquire auxiliary refined knowledge, thereby bolstering the model’s performance in MNER tasks. Specifically, we first utilize a Multimodal Similar Example Awareness module to select suitable examples from a small number of manually annotated samples. These examples are then integrated into a formatted prompt template tailored to the MNER task, guiding ChatGPT to generate auxiliary refined knowledge. Finally, the acquired knowledge is integrated with the raw text and inputted into the downstream model for further processing. Extensive experiments show that our PGIM significantly outperforms all existing state-of-the-art methods on two classic MNER datasets.

作者是期望利用LLM来解决：

- 一般的text+image的多模态小模型，可能需要外部的知识来进行识别

- 而基于外部knowledge的信息抽取方法检索到的外部知识可能相关性程较低，或者是冗余

作者同样把LLM看做是一个可以提供high-quality auxiliary knowledge的base。

方法：

![image-20230825160004278](https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825160004278.png)

首先，作者在这里使用LLM导出的外部knowledge包括了LLM抽取出的实体，以及推理的原因。

那么怎么样让LLM能够生成这样的knowledge呢？

作者随机从数据集中选择了一小部分样例，然后人工写了推理原因，这一小部分样例会作为待抽取的句子的上下文来获取LLM的knowledge。

作者使用cosine相似度，从这小部分人工标注的样例中选择合适的样例作为上下文（实现中选择$5$个样例做上下文）：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825160618895.png"   style="zoom:40%;" />

公式里的$H$代表着multimodal representations，作者使用UMT方法导出multimodal representations来计算样例相似度。（但不清楚这里的$H$具体是指序列中哪个embedding？）

拿到上下文之后，作者用来查询的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825161714124.png"   style="zoom:40%;" />

注意一下，只是使用了纯文本的ChatGPT，因此作者是使用BLIP2把image转化为text caption去查询的。并且在prompt里，作者提示LLM可以选择是否采用来自image的信息。

在拿到了LLM输出的auxiliary knowledge $z$之后，与原有的text拼接，经过一个Transformer encoder（实验中是XLM-RoBERTa-large），最后过CRF获取实体的BIO预测标注。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825161339967.png"  style="zoom:40%;" />

Twitter2015数据集相比较MoRe方法提升不太明显。（image在这两个Twitter数据集上到底有多大作用，个人现在很怀疑，并且标注也不够好，有很多的噪音…）

case study：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825161447842.png"   style="zoom:40%;" />

能够看出来，作者倾向于在LLM的输出推理过程中，直接对span进行解释，因此蓝色的句子里会很明显的线索来知道最后识别实体。

## CollabKG

CollabKG: A Learnable Human-Machine-Cooperative Information Extraction Toolkit for (Event) Knowledge Graph Construction

北交，arXiv 2023-07，ChatIE作者，[仓库](https://github.com/cocacola-lab/CollabKG)。

> In order to construct or extend entity-centric and event-centric knowledge graphs (KG and EKG), the information extraction (IE) annotation toolkit is essential. However, existing IE toolkits have several non-trivial problems, such as not supporting multi-tasks, not supporting automatic updates. In this work, **we present CollabKG, a learnable human-machine-cooperative IE toolkit for KG and EKG construction.** Specifically, for the multitask issue, CollabKG unifies different IE subtasks, including named entity recognition (NER), entity-relation triple extraction (RE), and event extraction (EE), and supports both KG and EKG. Then, combining advanced prompting-based IE technology, the human-machine-cooperation mechanism with LLMs as the assistant machine is presented which can provide a lower cost as well as a higher performance. Lastly, owing to the two-way interaction between the human and machine, CollabKG with learning ability allows self-renewal. Besides, CollabKG has several appealing features (e.g., customization, trainingfree, propagation, etc.) that make the system powerful, easy-to-use, and high-productivity. We holistically compare our toolkit with other existing tools on these features. Human evaluation quantitatively illustrates that CollabKG significantly improves annotation quality, efficiency, and stability simultaneously.

作者在ChatIE的的基础上，构造了一个可以人机协同的IE抽取工具。最大的特点个人认为是，它会不断把人工标注的结果，有选择的存入到一个数据库当中，这个数据库会被用来构造prompt，方便ChatIE进行信息抽取。这样就使得CollabKG一方面可以学习到新的概念/知识，同时还不需要额外的训练。支持中英文，支持实体抽取、三元组抽取、事件触发词检测以及事件论元抽取等。

与现有的其它IE toolkit的对比：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825221723968.png"   style="zoom:25%;" />

作者定义的工作流程：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825221819267.png"   style="zoom:35%;" />

首先，用户可以选择自动基于LLM进行信息抽取，会首先基于待抽取的text去一个知识库中进行检索（具体怎么样进行检索，论文中没有说明），然后选择到找到相似的已有标注样例，构造prompt，调用ChatGPT进行自动抽取（具体构造出来的prompt是什么样子没有在论文中给出）。

然后，自动抽取的结果会返回给用户，用户可以使用工具里定义的各种标注来修改自动抽取的结果，可以选择是否接受accept、删除delete或者待定suggested状态。

最后，人工修订后的结果会返回，CollabKG会将标注进行转化，变为统一的格式，选择最近的/高频的标注放入到知识库当中。

另外，作者定义统一的信息抽取样式/格式：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825222520746.png"   style="zoom:30%;" />

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825222733141.png"   style="zoom:25%;" />

具体的标注界面：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230825222926358.png"   style="zoom:25%;" />

## UniversalNER

UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition

2023-08，ICLR 2024，南加州大学，[项目](universal-ner.github.io)

> Large language models (LLMs) have demonstrated remarkable generalizability, such as understanding arbitrary entities and relations. Instruction tuning has proven effective for distilling LLMs into more cost-efficient models such as Alpaca and Vicuna. Yet such student models still trail the original LLMs by large margins in downstream applications. **In this paper, we explore targeted distillation with mission-focused instruction tuning to train student models that can excel in a broad application class such as open information extraction.** Using named entity recognition (NER) for case study, we show how ChatGPT can be distilled into much smaller UniversalNER models for open NER. For evaluation, we assemble the largest NER benchmark to date, comprising 43 datasets across 9 diverse domains such as biomedicine, programming, social media, law, finance. Without using any direct supervision, UniversalNER attains remarkable NER accuracy across tens of thousands of entity types, outperforming general instruction-tuned models such as Alpaca and Vicuna by over 30 absolute F1 points in average. With a tiny fraction of parameters, UniversalNER not only acquires ChatGPT’s capability in recognizing arbitrary entity types, but also outperforms its NER accuracy by 7-9 absolute F1 points in average. Remarkably, UniversalNER even outperforms by a large margin state-of-the-art multi-task instruction-tuned systems such as InstructUIE, which uses supervised NER examples. We also conduct thorough ablation studies to assess the impact of various components in our distillation approach. We will release the distillation recipe, data, and UniversalNER models to facilitate future research on targeted distillation.

一篇和InstructIE和InstructUIE相似思想的工作，都是训练IE LLM。这篇论文同时结合了现有的NER数据集和利用ChatGPT从raw text中进行open NER标注后的新构造的数据集进行训练。foundation model是LLaMA 7B/13B。

作者在论文强调的观点是，很多现有的instruction-tuning工作是利用了ChatGPT等更大size的LLM来构造指令，这可以看做是一种蒸馏技术。但是很多instruction-tuning工作是关注让student LLM学会在不同任务上遵循指令，这种做法是不可能超越teacher LLM如ChatGPT的。

因此作者觉得应该让LLM更加关注某一类任务，作者选择了NER任务作为探究任务。作者的指令重点不在于为不同的task构造不同的描述，而是想办法能够描述清楚不同数据集、不同领域的NER label的含义。

作者利用ChatGPT从Pile corpus中进行sentence-level open NER标注，不限制entity类型。只要是GPT认为是entity的mention都被导出。下面是进行标注的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829155757746.png"   style="zoom:30%;" />

经过清洗后，作者获得了240,725实体，13,020实体类型。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829155933160.png"   style="zoom:30%;" />

然后是如何构造instruction，以及如何训练。

- 作者根据ChatGPT的标注，直接询问某一类entity在text中的mention，相当于ChatIE方法的第二步。下面是instruction：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829160050553.png"   style="zoom:30%;" />

- Negative sampling，作者发现需要让LLM学会回答什么entity type没有在text中出现能够极大的提高模型学习效果（实验中有20%以上的效果提升）。进行依据entity type frequency的采样，构造负样本。

- 除去了利用ChatGPT标注的新数据外，作者也用到了现有的各类NER datasets。为了解决不同数据集之间label definitions的差异问题（如PERSON entity在ACL数据集中包括了she, he这些人称代词，而在multiNERD就没有包括人称代词），因此label需要和dataset相关联，作者额外的在prompt左侧加入dataset的name来进行辨别：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829160538488.png"   style="zoom:30%;" />

作者收集了现有的43个NER数据集，涉及9个domain，包括general, biomedical, clinical, STEM, programming, social media, law, finance, and transportation domains。

训练遵循和Vicuna一样的训练规划。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829160752996.png"  style="zoom:25%;" /><img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829160826877.png"   style="zoom:25%;" />

下面是对负采样的消融实验，可以看到让LLM学会回答自己不知道什么/什么东西不存在是很关键的：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230829161000484.png"   style="zoom:30%;" />



## SUMASK

Revisiting Large Language Models as Zero-shot Relation Extractors. 东南大学. EMNLP 2023

> Relation extraction (RE) consistently involves a certain degree of labeled or unlabeled data even if under zero-shot setting. Recent studies have shown that large language models (LLMs) transfer well to new tasks out-of-the-box simply given a natural language prompt, which provides the possibility of extracting relations from text without any data and parameter tuning. This work focuses on the study of exploring LLMs, such as ChatGPT, as zero-shot relation extractors. On the one hand, **we analyze the drawbacks of existing RE prompts and attempt to incorporate recent prompt techniques such as chain-of-thought (CoT) to improve zeroshot RE. We propose the summarize-and-ask (SUMASK) prompting, a simple prompt recursively using LLMs to transform RE inputs to the effective question answering (QA) format.** On the other hand, we conduct comprehensive experiments on various benchmarks and settings to investigate the capabilities of LLMs on zero-shot RE. Specifically, we have the following findings: (i) SUMASK consistently and significantly improves LLMs performance on different model sizes, benchmarks and settings; (ii) Zero-shot prompting with ChatGPT achieves competitive or superior results compared with zero-shot and fully supervised methods; (iii) LLMs deliver promising performance in extracting overlapping relations; (iv) The performance varies greatly regarding different relations. Different from small language models, LLMs are effective in handling challenge none-of-the-above (NoTA) relation.

设计更好的prompt，多轮提问LLM来实现更好的zero-shot relation extraction（给定头尾entity）。在zero-shot场景下，效果是目前的sota；在全监督设置下，使用zero-shot prompting的方式，在TACRED、TACREV和Re-TACRED三个数据集下靠近了sota或者达到了新sota。

首先，作者发现和以前的结果一致，如果是采用最vanilla的prompt，效果距离以前的sota方法效果很远：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016145531310.png"  style="zoom:40%;" />

作者推测原因是上面的prompt要求以下步骤在一步内完成：

- 理解句子中头尾实体的relation semantics；
- 理解relation label semantic；
- 然后能够把两种semantic进行匹配；

这些步骤在一步完成，让LLM对于RE任务表现不好。因此，作者借助CoT的思想，将RE任务拆分为不同的推理步骤：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016145721912.png"  style="zoom:50%;" />

具体步骤：

1. 给定sentence，让LLM推测句子中头尾实体可能的语义联系，作者称为输出summarization，重复$k$次；
2. 输入候选relation，构造候选三元组，让LLM根据三元组创建出对应的回答yes/no的问题question，重复$k$次；
3. 对于前面的$k$个summarization和question，让LLM回答是否能够根据summarization回答question，通过投票选择最终LLM认为头尾实体间的候选关系是否成立；

对于每种候选关系$r$，都要进行$k^3$的提问（实验中$k=5$），资源消耗和时间成本非常大。下面是几个例子：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016151659972.png"  style="zoom:50%;" />

经过上面3步，会获得多个最终LLM认为成立的候选relation，因此需要设计某种机制进行选择，最直接的想法是选择3个步骤输出答案概率最大的对应的relation：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016150313425.png" style="zoom:40%;" />

但是，不是所有LLM都能够输出概率的，因此，使用不确定性进行估计，选择不确定性最小的回答：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016150449686.png"  style="zoom:40%;" />

不确定性估计的方法借鉴了前人的方法[*Active prompting with chain-ofthought for large language models. 2023*]，将$k$个回答输入到Sentence-BERT中，获得编码，然后计算每个回答表征和平均表征的距离作为dispersion degree：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016150606944.png"  style="zoom:40%;" />

最后选择依据：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016150705709.png" style="zoom:40%;" />

需要注意的是，作者采用了Entity-Relation Mapping机制[*Fastre: Towards fast relation extraction with convolutional encoder and improved cascade binary tagging framework. IJCAI 2022*]，提前排除了很多relation，如果已知了entity type，就提前排除掉一些不可能成立的relation。

作者实验采用了`GPT-J-6B`、`BLOOM-7.1B`、`T0pp-11B`和`gpt-3.5-turbo-0301`，最终是gpt-3.5-turbo-0301效果最好，下面实验里的SUMASK就是对应的采用gpt-3.5的结果。

Zero-shot relation classification实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016151026213.png" style="zoom:30%;" />

和全监督的RE方法对比：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016151114659.png" style="zoom:30%;" />

和全监督的方法相比，zero-shot的LLM有潜力靠近sota，但是个人仍然有以下的疑问：
1. 资源消耗？
2. 时间？
3. 这些数据集表现比较好，是否是因为LLM在预训练阶段已经理解了对应的label，如果是新的dataset，LLM是否仍然能够表现好？

作者论文中Table4还进行了其它实验，发现不同relation之间的性能差距非常大，最好的有90%以上准确率，最差的10%-20%准确率。

对于Overlapping RE任务的实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016151345868.png"  style="zoom:30%;" />

## Guideline Learning

Guideline Learning for In-Context Information Extraction. EMNLP 2023. 中科院

> Large language models (LLMs) can perform a new task by merely conditioning on task instructions and a few input-output examples, without optimizing any parameters. This is called In-Context Learning (ICL). In-context Information Extraction has recently garnered attention in the research community. However, current experiment results are generally suboptimal. **We attribute this primarily to the fact that the complex task settings and a variety of edge cases are hard to be fully expressed in the length-limited context.** In this paper, **we propose a Guideline Learning (GL) framework for In-context IE which learns to generate and follow guidelines.** During the learning phrase, GL automatically synthesizes a set of guidelines from a few annotations, and during inference, helpful guidelines are retrieved for better ICL. Experiments on event extraction and relation extraction show that guideline learning can improve the performance of in-context IE.

**Issue**：作者认为IE任务是一个复杂的任务，为了准确全面的定义好task的*target concept*需要很多的examples和rules进行定义。例如在ACM关系抽取中的guidelines超过33页内容。

传统的方法需要提前有很多的训练样本+大量的训练参数；而LLM+ICL是一种能够无梯度更新的范式，然而这种范式在以前的论文里效果还不够好。

作者认为原因是，LLM对于IE任务理解的*comprehended concept*和*target concept*之间存在*conceptual bias*：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016191744066.png" style="zoom:35%;" />

**Solution**: 因此，作者希望能够让LLM自己学习guidelines，这种guidelines，实际上，就是对于label semantic的描述。作者提出方法的图：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016192009243.png" style="zoom:30%;" />

核心思想是，从预测错误的样例中学习反馈，思想follow了前人的工作[*Memory-assisted prompt editing to improve GPT-3 after deployment. EMNLP 2022*]。

伪代码：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016192126083.png"  style="zoom:35%;" />

作者定义的guidelines就是一系列的自然语言描述的rules集合：

> Suppose we have collected the Guidelines $\mathcal{G} = \{ r_i \}|_{i=1}^{|\mathcal{G}|}$ which is a set of rules that supports read, write, and retrieve operations. Each rule, expressed as a natural language sentence, defines an aspect of the task. The guidelines illustrate how to perform the task.

首先是伪代码中的`Retrieve`：从已有的guidelines集合里，寻找和当前样例最相近的guidelines。具体做法是利用LLM去抽象出sentence $x$的general form $\tilde{x}$，对于RE任务，作者用了两种方式，一种是简化具体的描述，只保留text中必要的tokens，一种是让LLM猜测实体的type。两种形式的general forms拼接在一起作为RE任务对于sentence semantic的抽象$\tilde{x}$，输入到OpenAI的`text-embedding-ada-002` API，获得embedding，然后基于余弦相似度选择最相似的rules。同时随机从训练集中找样例作为demonstrations，在整个数据集下是固定的。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016192458557.png" style="zoom:50%;" />

然后`Reason`：输入task instruction、query instance、few-shot examples和rules，让LLM输出抽取结果和LLM认为有用的rules。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016193053838.png"  style="zoom:50%;" />

根据抽取结果，更新rules的scores，来判断某个rule的有用程度(某个rule被使用的次数中，能够帮助预测正确的次数越多越好)：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016193218204.png"  style="zoom:40%;" />

如果抽取relation是错误的，那么要进行反馈`Reflect`：使用general form和true label直接拼接作为新的rule。从作者的说明来看：

> In this paper, we simply concatenate the general form $\tilde{x}$ of the instance $i$ and the golden label to generate a rule. Figure 3 presents an example of this process in EE.

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240408155256384.png"  style="zoom:50%;" />

作者另外的贡献是，提出了一种Active Instance Selection的策略，选择适合作为上面的guidelines learning的方法。如果是随机选择下一次适合标注的样本的话，效率可能比较低，因为可能会选出LLM已经能够正确预测的样本。因此作者选择那些LLM最不confidence的data。

具体的选择方法是，利用self-consistency CoT，让LLM生成多个推理路径和对应的答案。然后根据答案的分布，统计不同类型relation的分布，然后使用熵的负值作为confidence。最后选择负熵最小的data，也就是LLM预测概率分布最平滑的data。

最后实现使用`gpt-3.5-turbo`，实验了Event Extraction和relation extraction（给定头尾实体）两类任务。事件抽取使用了ChFinAnn数据集；RE任务使用了SemEval 2010 task 8数据集（有9种relation），测试集随机采样1000个。从RE任务结果来看，和SOTA还有差距：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016194434785.png"  style="zoom:40%;" />

上面作者的baseline：

- RE-ICL：直接让LLM基于ICL输出relation label
- RE-GL-b：作者的方法，without guidelines
- RE-GL-r：从训练集中，随机选择500个instances作为guidelines learning（平均每个relation 50个instances）
- RE-GL-a：从训练集中，主动学习策略，从随机的1000个instances中，选出500个instances进行guidelines learning

case study：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20240408155338816.png"  style="zoom:40%;" />

## RationalCL

Rationale-Enhanced Language Models are Better Continual Relation Learners. EMNLP 2023. 北大. [代码](https://github.com/WeiminXiong/RationaleCL)。

> Continual relation extraction (CRE) aims to solve the problem of catastrophic forgetting when learning a sequence of newly emerging relations. Recent CRE studies have found that catastrophic forgetting arises from the model’s lack of robustness against future analogous relations. To address the issue, **we introduce rationale, i.e., the explanations of relation classification results generated by large language models (LLM), into CRE task.** Specifically, we design the multi-task rationale tuning strategy to help the model learn current relations robustly. We also conduct contrastive rationale replay to further distinguish analogous relations. Experimental results on two standard benchmarks demonstrate that our method outperforms the state-of-the-art CRE models. Our code is available at https://github.com/WeiminXiong/RationaleCL

作者期望解决的任务是Continual Relation Extraction，下面是任务定义：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016222421718.png"  style="zoom:30%;" />

这一任务之前的方法是专注于利用知识蒸馏等技术缓解灾难性遗忘问题。最近有工作发现，模型在遭遇和已经见过的relation相似的新relation情况下，不能够很好的搞清楚相似relation之间的关系。

作者认为，可以通过引入rationales来缓解这一问题。这是因为一方面This is inspired by the intuition that, training models with explicit rationale supervision can provide greater robustness (Chen et al., 2022).。另外，让小模型学会输出rationales，能够让缓解小模型可能的shortcut learning的问题。

但是目前数据集里并没有rationales。因此作者期望让LLM来输出rationales。虽然GPT对于直接进行RE任务效果还不够好，但是如果给定relation，让GPT给出原因来，效果还是可以的。即

> According to their studies, ChatGPT is limited by the output format requirements in accomplishing fine-grained relation extraction tasks, and it is difficult to directly generate the target relation label within the defined range. However, ChatGPT’s semantic understanding is sufficient, and when we provide chatgpt with correct relation labels, ChatGPT can understand the meaning of the relation according to the context and give reasonable rationales. According to the human-check results (Li et al., 2023), domain experts highly approve of the reasons given by ChatGPT.

作者利用`gpt-3.5-turbo`输出的rationales，finetune一个T5-base，使用多任务的架构：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016221257901.png" style="zoom:50%;" />

第一阶段训练三个任务的loss：

- 主任务，输入文本$X$，输出relation $Y$
- 辅助任务1，输入文本$X$，让小模型输出relation $Y$和rationale $R$
- 辅助任务2，输入文本$X$和rationale $R$，让小模型输出relation $Y$

第一阶段结束后，为了缓解灾难性遗忘问题，作者follow前人的工作[*Continual relation learning via episodic memory activation and reconsolidation. ACL 2020*]，维护一个episodic memory module，里面保存不同relation下代表性的data（实现中每个relation保留10个data）。这个memory里面的data，会在之后继续被进行学习。

第二阶段是为了解决新relation和已有的相似relation容易混淆的问题。作者提出了contrastive rationales的概念：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016222112205.png"  style="zoom:35%;" />

通过embedding计算两个relation之间的余弦相似度，找到相似relation；然后让LLM去给出输出一个relation，而不输出另一个relation的rationales。用新的contrastive rationales替换memory module里面的已有rationales，然后训练。

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231016222501679.png" style="zoom:30%;" />

## Text2KGBench

Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text. ISWC 2023. IBM. [代码](https://github.com/cenguix/Text2KGBench).

> The recent advances in large language models (LLM) and foundation models with emergent capabilities have been shown to improve the performance of many NLP tasks. LLMs and Knowledge Graphs (KG) can complement each other such that LLMs can be used for KG construction or completion while existing KGs can be used for different tasks such as making LLM outputs explainable or fact-checking in Neuro-Symbolic manner. **In this paper, we present Text2KGBench, a benchmark to evaluate the capabilities of language models to generate KGs from natural language text guided by an ontology.** Given an input ontology and a set of sentences, the task is to extract facts from the text while complying with the given ontology (concepts, relations, domain/range constraints) and being faithful to the input sentences. We provide two datasets (i) Wikidata-TekGen with 10 ontologies and 13,474 sentences and (ii) DBpedia-WebNLG with 19 ontologies and 4,860 sentences. We define seven evaluation metrics to measure fact extraction performance, ontology conformance, and hallucinations by LLMs. Furthermore, **we provide results for two baseline models, Vicuna-13B and Alpaca-LoRA-13B using automatic prompt generation from test cases.** The baseline results show that there is room for improvement using both Semantic Web and Natural Language Processing techniques.

KG的构建可以通过RDB2RDF[*A survey of current approaches for mapping of relational databases to rdf.*]这种方法从relational data中构建；可以用RML这种方法[*Rml: A generic language for integrated rdf mappings of heterogeneous data.*]，从半结构化的数据中构建；还可以考虑用众包的形式构建，例如Wikidata。

但是仍然存在很多非结构化的文本，并且受限于规模和隐私考虑等因素，无法使用众包。也因此，一种构建KG的思路就是利用NLP技术，比如NER、RE、entity linking等方法，从非结构化文本中抽取结构化信息。目前有两个相关的workshop：

> There is a growing interest in the Semantic Web community to explore such approaches as seen from the workshops such as Text2KG [Tiwari et al., 2022, 2023] and NLP4KGC [Vakaj et al., 2023].

作者构建了Text2KGBench来评估使用LLM，在给定ontology的情况下，能够生成KG的能力。作者的benchmark里KG不是用RDF/OWL的形式描述的，而是使用relation三元组的形式。

包括了两个数据集：

- Wikidata-TekGen：包括10种从Wikidata中导出的ontologies，然后使用TekGen语料库找到对应的三元组和句子
- DBpedia-WebNLG：包括19种本体

下面是Wikidata-TekGen数据集music本体示例：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231017152255979.png" style="zoom:40%;" />

基于作者的Text2KGBench，作者尝试了Vicuna-13B和Alpaca-LoRA-13B在给定本体的情况下，基于kNN ICL进行信息抽取的效果。下面是使用的prompt：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231017152527701.png"  style="zoom:30%;" />

作者提出，要使用三类指标来评估：

- Fact Extraction Accuracy: Precision (P), Recall (R), and F1
- Ontology Conformance: Ontology Conformance (OC) metric, a triple is considered to be conforming to the ontology if the relation is one of the canonical relations listed in the ontology.
- Hallucinations: subject hallucination (SH), relation hallucination (RH), and object hallucination (OH). For each triple, SH and OH check if the subject and object are present in either the sentence or the ontology concepts, and RH checks if the relation is present in the ontology relations.

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20231017153036430.png"  style="zoom:50%;" />

可以看到，总体效果还有很大的进步空间，同时是存在一定程度的幻觉的。

