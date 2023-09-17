---
title: LLM-reason
published: true
date: 2023-09-06 16:22:15
categories:
- Paper
- LLM
- Reason
tags:
- LLM
- Reason
---

# 使用LLM的推理方法

使用LLM进行推理的相关论文总结

<!--more-->

## CoT

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

Google Research，NeurIPS 2022

> We explore how generating **a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.** In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called **chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting.**
>
> Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

首次提出了chain-of-thought思路的论文。简单说就是在上下文学习中，对于每个输出输出样例examplar，加入中间的推理步骤，格式为`<input, chain of thought, output>`：

> A chain of thought is a series of intermediate natural language reasoning steps that lead to the final output, and we refer to this approach as chain-of-thought prompting.

CoT灵感的来源是两个：

- First, techniques for arithmetic reasoning can benefit from generating natural language rationales that lead to the final answer. 之前的rationale-augmented training的方法发现通过利用自然语言描述的中间步骤rationales可以帮助解决数学推理问题；但是这种方法需要获得大量的rationales，实际上是很难获得的。
- Second, large language models offer the exciting prospect of in-context few-shot learning via prompting. LLM已经通过上下文学习模拟样例，可以解决很多问题，无需训练。但是在需要推理能力的任务上表现很差

作者CoT的思想就是融合了这两种思路，如果能够获得中间推理的步骤，可以帮助解决推理问题；而上下文学习又能够让LLM模拟少数几个样例，无训练的快速学习；那么如果让LLM学会模拟几个样例中的推理步骤，是不是就可以让LLM也迅速学会模拟推理？

方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906164647259.png"   style="zoom:40%;" />

思考和推理的过程像是一个chain，因此叫做chain of thought。即使出现的prompt样例中的中间step看起来只是一段话，但它本质上仍然表达了思维链的过程。

实验结果显示，通过无训练的CoT prompting方法，可以让PaLM 540B在数学问题上达到sota：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906164755714.png"   style="zoom:25%;" />

CoT有个潜在的假设/前提，某个任务的输入到输出过程，是可以用自然语言的形式进行描述的。

CoT的几个好处：

- First, chain of thought, in principle, allows models to decompose multi-step problems into intermediate steps, which means that additional computation can be allocated to problems that require more reasoning steps.
- Second, a chain of thought provides an interpretable window into the behavior of the model, **suggesting how it might have arrived at a particular answer and providing opportunities to debug where the reasoning path went wrong** (although fully characterizing a model’s computations that support an answer remains an open question).
- Third, chain-of-thought reasoning can be used for tasks such as math word problems, commonsense reasoning, and symbolic manipulation, and is potentially **applicable (at least in principle) to any task that humans can solve via language**.
- Finally, chain-of-thought reasoning can be readily elicited in sufficiently large off-the-shelf language models simply by including examples of chain of thought sequences into the exemplars of few-shot prompting.

## Self-Verification

Large Language Models are Better Reasoners with Self-Verification

arXiv 2013

> Recently, with the chain of thought (CoT) prompting, large language models (LLMs), e.g., GPT-3, have shown strong reasoning ability in several natural language processing tasks such as arithmetic, commonsense, and logical reasoning. However, LLMs with CoT require multi-step prompting and multi-token prediction, which is highly sensitive to individual mistakes and vulnerable to error accumulation. The above issues make the LLMs need the ability to verify the answers. In fact, after inferring conclusions in some thinking decision tasks, **people often check them by re-verifying steps to avoid some mistakes. In this paper, we propose and prove that LLMs also have similar self-verification abilities.** We take the conclusion obtained by CoT as one of the conditions for solving the original problem. By taking turns masking the original conditions and predicting their results, **we calculate an explainable answer verification score based on whether the re-predicted conditions are correct.** Experimental results demonstrate that the proposed method can improve the reasoning performance on various arithmetic, commonsense, and logical reasoning datasets. Our code is publicly available at: https://github.com/WENGSYX/Self-Verification.

使用CoT能够提升LLM的推理能力，但是CoT prompting方法对于个别的小的mistake很敏感从而会出现错误的结果。另外考虑到多次输入相同的prompt，LLM也可能出现不同的推理结果。因此如何从LLM的推理结果中找出正确的答案是很重要的。

之前修正多步推理过程中出现的错误的方法，是training一个额外的验证器verifier。这种做法一方面要求有对应的标注数据；另一方面训练出来的验证器本身又是难以解释的，可靠性难以评估。
因此作者提出了自我验证Self-Verification，无需训练的思路。即让LLM自己验证自己的推理结果。这种做法实际人类也经常做，humans often perform self-verification of inferred conclusions to mitigate mistakes。

作者的方法有两步：

- Forward Reasoning：通过CoT prompting让LLM生成candidate answers，即从conditions推导出conclusions，$X\rightarrow Y$
- Backward Verification：验证上一步的多个候选答案，mask原来的condition，根据LLM推理出的答案，推导被mask掉的condition。准确预测出来的condition越多，验证得分越高。最后选择验证得分最大的答案。$Y\rightarrow X$

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906202533743.png"   style="zoom:40%;" />

实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906203043968.png"  style="zoom:30%;" />

## Self-Consistency CoT

Self-Consistency Improves Chain of Thought Reasoning in Language Models

ICLR 2023，Google research

> Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, **self-consistency**, to replace the naive greedy decoding used in chain-of-thought prompting. It ﬁrst samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths. **Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer.** Our extensive empirical evaluation shows that self-consistency boosts the performance of chain-of-thought prompting with a striking margin on a range of popular arithmetic and commonsense reasoning benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).

原始的CoT prompting方法是直接生成唯一的推理路径和对应的答案，这种策略可以叫做是greedy decoding strategy，即取推理路径中概率最大的策略。

这篇论文提出一种新的decoding策略 Self-Consistency，它的直觉是一个复杂的推理任务，可能有多种推理路径能够得到最终的答案。越复杂的问题，能够获得最终答案的推理路径可能多样性越大。正确的推理路径不是唯一的。

Self-Consistency的方法很简单，就是让LLM生成多个可能的推理路径和答案，然后选择其中最consistent的答案。作者采用了majority vote的策略，因此最终选择候选答案中出现次数最多的那个答案：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906213813563.png"   style="zoom:40%;" />

方法很简单，不过论文中有一些发现值得思考：

- 这是一种类似于self-ensemble的方法，集成一个model的多个预测结果

- 推理路径的出现概率，可以通过每个位置上token的出现概率之和进行估计：

  <img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906214012916.png"   style="zoom:50%;" />

  作者发现这样子计算出来的各个推理路径的概率值互相之间很接近，没有太大差距。这证明了LLM实际上没有能力评估出不同solution之间有什么差异。因此，使用这种推理路径的概率去选择最后的答案，和直接认为每个推理路径的权重是1的效果差别不大

- 直观的说，self-consistency能够适应的推理任务是答案唯一的情况。不过也可以考虑拓展，比如比较生成的文本之间是否矛盾或者冲突，然后选择互相之间最不冲突的文本

部分实验结果：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230906214258541.png"   style="zoom:30%;" />

## Maieutic Prompting

Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations

华盛顿大学，EMNLP 2022，[代码](https://github.com/jaehunjung1/Maieutic-Prompting)。

> Pre-trained language models (LMs) struggle with consistent reasoning; recently, prompting LMs to generate explanations that self-guide the inference has emerged as a promising direction to amend this. However, these approaches are fundamentally bounded by the correctness of explanations, which themselves are often noisy and inconsistent. In this work, **we develop Maieutic Prompting, which aims to infer a correct answer to a question even from the unreliable generations of LM.** Maieutic Prompting induces a tree of explanations abductively (e.g. X is true, because . . .) and recursively, then frames the inference as a satisﬁability problem over these explanations and their logical relations. We test Maieutic Prompting for true/false QA on three challenging benchmarks that require complex commonsense reasoning. Maieutic Prompting achieves up to 20% better accuracy than state-of-the-art prompting methods, and as a fully unsupervised approach, performs competitively with supervised models. We also show that M AIEUTIC PROMPTING improves robustness in inference while providing interpretable rationales.

虽然之前的paper常常假设LLM可以模拟人类的推理能力。但是实际上目前还做不到。LLM的推理是常常自相矛盾的，不可依赖的：

- the explanation does not logically lead to the inferred answer 解释和答案不一致
- the model infers the same label for a statement and its negation 相反的解释有相同的答案
- falsiﬁes its own generated explanation 篡改/伪造自己的解释

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907163907919.png"   style="zoom:25%;" />

作者是受到了maieutic method的启发，逐步的确认和消除有矛盾的假设：

> Maieutic method brings out deﬁnitions implicit in the interlocutor’s beliefs, ... is a method of hypothesis elimination, steadily identifying and eliminating those that lead to contradictions (Vlastos, 1991).

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907164128950.png"   style="zoom:50%;" />

一般的CoT prompting的方法对于某个答案是True/False的问题，通常让LLM先给出解释，再给出答案/预测标签（这称为ad-hoc approach，事前解释）。这种方法生成的explanation是a discriminative explanation that helps in choosing one label over the other。

相反，作者让LLM对True/False给出事后解释。也就是先给定True/False，让LLM给出对应的explanation。这种Abductive generation的方式能够让LLM分别考虑不同的答案的可能性，而不是直接做选择。另外一个好处是，给定的True/False的label information，能够诱导LLM给出更加具体specific的解释。

作者提出的一个重要观点是验证LLM生成的propositions的逻辑完整性/完备性logically integral：

> We formalize this idea as logical integrity: a proposition $Q$ is logically integral when the LM consistently infers the truth value of $Q$ and $\neg Q$ (i.e. $Q$ as True and $\neg Q$ as False, or vice versa)

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907165024823.png"   style="zoom:40%;" />

也就是说对于某个LLM生成的proposition $Q$，LLM不能认为$Q$和$\neg Q$同时成立。

满足logically integral的解释，作者认为是更加可靠的解释。只有满足logically integral的解释会被保留，然后用于最终的逻辑推理。不满足logically integral的解释，会被继续溯因，迭代的让LLM解释，直至获得logically integral的解释。

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907165419259.png"   style="zoom:50%;" />

对于最后生成的逻辑树，作者评估LLM对于叶子节点解释成立的belief：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907165857365.png"   style="zoom:50%;" />

以及不同解释之间的关系，作者使用了额外的一个NLI model（RoBERTa fine-tuned on MNLI dataset）去判断任意两个不同explanation之间的关系，权重$w$固定为1：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907165944830.png"   style="zoom:50%;" />

最后使用MAX-SAT去最大程度的满足各种带有权重的条件。

## PAL

PAL: Program-aided Language Models

ICML 2023，CMU，[代码](https://github.com/reasoning-machines/pal)。

> Large language models (LLMs) have demonstrated an impressive ability to perform arithmetic and symbolic reasoning tasks, when provided with a few examples at test time (“few-shot prompting”). Much of this success can be attributed to prompting methods such as “chainof-thought”, which employ LLMs for both understanding the problem description by decomposing it into steps, as well as solving each step of the problem. **While LLMs seem to be adept at this sort of step-by-step decomposition, LLMs often make logical and arithmetic mistakes in the solution part, even when the problem is decomposed correctly.** In this paper, we present ProgramAided Language models (PA L): a novel approach that uses the LLM to read natural language problems and **generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter.** With PAL, decomposing the natural language problem into runnable steps remains the only learning task for the LLM, while solving is delegated to the interpreter. We demonstrate this synergy between a neural LLM and a symbolic interpreter across 13 mathematical, symbolic, and algorithmic reasoning tasks from BIG-Bench Hard and others. In all these natural language reasoning tasks, generating code using an LLM and reasoning using a Python interpreter leads to more accurate results than much larger models. For example, PAL using CODEX achieves state-of-the-art few-shot accuracy on GSM8K, surpassing PaLM-540B which uses chain-of-thought by absolute 15% top-1.

这篇文章主要是解决虽然LLM能够理解问题，并且把问题的推理过程分别为不同步骤，但是在最后根据推理过程计算答案的时候出现错误的问题。有时候即便是推理过程是正确的，LLM最后的计算答案是错误的。

因此作者提出对问题的理解，以及推理步骤的拆分和规划让LLM完成。计算答案、根据推理过程推导答案由外部工具完成。这样就能够缓解推理链正确，最后对应的推理结果错误的问题。让LLM做自己擅长的部分，不擅长的部分让其它工具完成。

让LLM生成推理答案的时候同时生成编程语言，最后使用外部的工具如python interpreter计算答案：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230907215210863.png"  style="zoom:40%;" />

## Least-to-Most

Least-to-Most Prompting Enables Complex Reasoning in Large Language Models

ICLR 2023, Google research

> Chain-of-thought prompting has demonstrated remarkable performance on various natural language reasoning tasks. **However, it tends to perform poorly on tasks which requires solving problems harder than the exemplars shown in the prompts.** To overcome this challenge of easy-to-hard generalization, we propose a novel prompting strategy, **least-to-most prompting**. **The key idea in this strategy is to break down a complex problem into a series of simpler subproblems and then solve them in sequence.** Solving each subproblem is facilitated by the answers to previously solved subproblems. Our experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difﬁcult problems than those seen in the prompts. A notable ﬁnding is that when the GPT-3 code-davinci-002 model is used with least-to-most prompting, it can solve the compositional generalization benchmark SCAN in any split (including length split) with an accuracy of at least 99% using just 14 exemplars, compared to only 16% accuracy with chain-of-thought prompting. This is particularly noteworthy because neural-symbolic models in the literature that specialize in solving SCAN are trained on the entire training set containing over 15,000 examples. We have included prompts for all the tasks in the Appendix.

前面的CoT prompting方法会面临的问题是，如果要解决的问题比CoT中样例的问题更难的话，效果就很差：

> However, chain-of-thought prompting has a key limitation—it often performs poorly on tasks that require generalization of solving problems harder than the demonstration examples, such as compositional generalization (Lake & Baroni, 2018; Keysers et al., 2020).

因此，作者提出了一种从简单到复杂，先把问题分解为几个子问题，然后逐步的解决子问题的方法：

<img src="https://lxy-blog-pics.oss-cn-beijing.aliyuncs.com/asssets/image-20230908105201421.png"   style="zoom:40%;" />

第一步是通过给定几个样例，让LLM把问题分解成几个子问题Decomposition；

第二步是逐步的解决子问题，子问题的答案又附加到prompt上，作为解决下一个子问题的上下文。

- **Decomposition.** The prompt in this stage contains constant examples that demonstrate the decomposition, followed by the speciﬁc question to be decomposed.
- **Subproblem solving.** The prompt in this stage consists of three parts: (1) constant examples demonstrating how subproblems are solved; (2) a potentially empty list of previously answered subquestions and generated solutions, and (3) the question to be answered next.
