# Transformers and Self-Attention

序列化的模型类似于RNN，存在几个问题：

- Sequential computation的计算限制了并行计算
- 没有对于short和long dependencies的显式建模
- 我们希望能够建模层级

对于迁移不变性的解释。

![image-20210329210505679](image-20210329210505679.png)

## 