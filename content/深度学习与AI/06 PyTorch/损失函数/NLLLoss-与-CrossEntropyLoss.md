---
title: NLLLoss 与 CrossEntropyLoss
url: https://zhuanlan.zhihu.com/p/570118948
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:48:26+00:00'
polished_at: '2026-06-27T19:21:52+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# NLLLoss 与 CrossEntropyLoss

## 简述

[pytorch](https://zhida.zhihu.com/search?content_id=214826102&content_type=Article&match_order=1&q=pytorch&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTUyOTQsInEiOiJweXRvcmNoIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjE0ODI2MTAyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.vwkosit9XQWNrPGYfJa4pnYq0aN6bm83a3QnxKxPCEg&zhida_source=entity)里面常涉及的两个损失函数：NLLLoss()和CrossEntropyLoss()，本质而言都是交叉熵损失函数，只是使用上略有不同。相对而言，CrossEntropyLoss()使用的更普遍。其差别在于，CrossEntropyLoss()不单是做了交叉熵，而且在里面还加入了log和softmax，也就是：
CrossEntropyLoss=NLLLoss + softmax + log
**NLLLoss**：**N**egative **L**og **L**ikelihood **L**oss，负对数似然损失函数。
**CrossEntropyLoss：**交叉熵损失函数。

## 实例

用例子来解释下CrossEntropyLoss和NLLLoss的区别，以三个样本为一个batch，任务为三分类为例。
**1.**

**input_:**输入，模型的预测结果；
**target：**真实结果, groudTruth，预测三个样本分类为类别0, 2, 1；
![Image 1](https://picx.zhimg.com/v2-ae824a5ca79a337cfb29fade8304979f_1440w.jpg)

<p class="kb-image-caption">图例</p>

结果和第4步的结果一样，所以可以发现，**NLLLoss就是做了第4步的工作**（对经过log和softmax后的结果进行处理），取target对应位置的值，取负数后相加求平均。
**6.**来对比下**CrossEntropyLoss**，将第1的input_和target输入到CrossEntropyLoss，得到如下结果
![Image 6](https://pic1.zhimg.com/v2-913910a9d3e5411d1d5ea37ccd548d1a_1440w.png)

<p class="kb-image-caption">图例</p>

CrossEntropyLoss=NLLLoss + softmax + log在实际训练中，如果做的是分类任务，且使用CrossEntropyLoss作为损失函数的话，神经网络的部分就没必要加入nn.Softmax或者nn.LogSoftmax等之类的，因为在CrossEntropyLoss已经内置了该功能。

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/损失函数/BCELoss-二元交叉熵|BCELoss 二元交叉熵]]
[[content/深度学习与AI/06 PyTorch/损失函数/BCEWithLogitsLoss|BCEWithLogitsLoss]]
