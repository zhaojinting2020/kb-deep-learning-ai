---
title: Huggingface Transformers 教程
url: https://zhuanlan.zhihu.com/p/448852278
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:55:31+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
math_repaired_at: '2026-06-27T19:29:26+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
tables_repaired_at: '2026-06-28T18:55:36+00:00'
---

# Huggingface Transformers 教程

​目录本系列文章介绍**[Huggingface Transformers](https://zhida.zhihu.com/search?content_id=187874774&content_type=Article&match_order=1&q=Huggingface+Transformers&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3MTUsInEiOiJIdWdnaW5nZmFjZSBUcmFuc2Zvcm1lcnMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxODc4NzQ3NzQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.PAjjdPMbLkaKn_eRKTDz50DXFvpRWd_jmzIdMFDWH3E&zhida_source=entity)**的用法。Huggingface是一家在[NLP](https://zhida.zhihu.com/search?content_id=187874774&content_type=Article&match_order=1&q=NLP&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3MTUsInEiOiJOTFAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxODc4NzQ3NzQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.scngk17_XILN5rGZn-n0k7cgsGiBmS_JGkOtuv2vvnM&zhida_source=entity)社区做出杰出贡献的纽约创业公司，其所提供的大量预训练模型和代码等资源被广泛的应用于学术研究当中。
**Transformers**提供了数以千计针对于各种任务的预训练模型模型，开发者可以根据自身的需要，选择模型进行训练或微调，也可阅读api文档和源码， 快速开发新模型。本文基于 **Huggingface 推出的NLP 课程**，内容涵盖如何全面系统地使用 HuggingFace 的各类库（即 Transformers, [Datasets](https://zhida.zhihu.com/search?content_id=187874774&content_type=Article&match_order=1&q=Datasets&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3MTUsInEiOiJEYXRhc2V0cyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjE4Nzg3NDc3NCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.uPc-sJNWKfdoz7togBKef4sOBpwiVGeV1rVoek5OAvs&zhida_source=entity), Tokenizers 和 Accelerate），以及 [Hugging Face Hub](https://zhida.zhihu.com/search?content_id=187874774&content_type=Article&match_order=1&q=Hugging+Face+Hub&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3MTUsInEiOiJIdWdnaW5nIEZhY2UgSHViIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTg3ODc0Nzc0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.ZtnuEokJbjWQvFO2wNFPqmu-zORCkJdONOFnNgGlaII&zhida_source=entity) 中的各种模型。本篇是该系列第**上**篇，如果觉得有帮助到你，可以**点个赞**鼓励下，让我继续更新~

### 课程主页 ：[https://huggingface.co/course/c](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter0%3Ffw%3Dpt)

## 零, [Setup](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter0/1%3Ffw%3Dpt)
1）安装一个非常轻量级的 Transformers `!pip install transformers`然后`import transformers` 2）建议安装开发版本，几乎带有所有用例需要的依赖项`!pip install transformers[sentencepiece]`

## 模型简介 [Transformer models](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter1/1%3Ffw%3Dpt)
### pipelines 简单的小例子

> Transformers 库中最基本的对象是`pipeline()`函数。它将模型与其必要的预处理和后处理步骤连接起来，使我们能够**直接输入任何文本并获得答案**：

当第一次运行的时候，它会下载预训练模型和分词器(tokenizer)并且缓存下来。
from transformers import pipeline classifier = pipeline("sentiment-analysis")  # 情感分析classifier("I've been waiting for a HuggingFace course my whole life.")

# 输出
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
也可以传几句话：
classifier(     ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"] )

# 输出
'''

[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}] '''
目前[可用的](https://link.zhihu.com/?target=https%3A//huggingface.co/transformers/main_classes/pipelines.html)一些pipeline 有：
> `feature-extraction`特征提取：把一段文字用一个向量来表示
>

> `fill-mask`填词：把一段文字的某些部分mask住，然后让模型填空
>

> `ner`命名实体识别：识别文字中出现的人名地名的命名实体
>

> `question-answering`问答：给定一段文本以及针对它的一个问题，从文本中抽取答案
>

> `sentiment-analysis`情感分析：一段文本是正面还是负面的情感倾向
>

> `summarization`摘要：根据一段长文本中生成简短的摘要
>

> `text-generation`文本生成：给定一段文本，让模型补充后面的内容
>

> `translation`翻译：把一种语言的文字翻译成另一种语言
>

> `zero-shot-classification`

这些pipeline的具体例子可见：[Transformer models - Hugging Face Course](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter1/3%3Ffw%3Dpt)

### 各种任务的代表模型
| Model | Examples | Tasks |
| --- | --- | --- |
| Encoder 编码器模型 | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa |
| Decoder 解码器模型 | CTRL, GPT, GPT-2, Transformer XL | Text generation 解码器模型的预训练通常围绕预测句子中的下一个单词。这些模型最适合涉及文本生成的任务 |
| Encoder-decoder 序列到序列模型 | BART, T5, Marian, mBART | Summarization, translation, generative question answering 序列到序列模型最适合围绕根据给定输入生成新句子的任务，例如摘要, 翻译或生成式问答。 |本节测试：[Transformer models - Hugging Face Course](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter1/10%3Ffw%3Dpt) |

Sentence classification, named entity recognition, extractive question answering 适合需要理解完整句子的任务，例如句子分类, 命名实体识别（以及更一般的单词分类）和提取式问答

## 使用 [Using Transformers](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/1%3Ffw%3Dpt)
### Pipeline 背后的流程
![Image 1](https://pic4.zhimg.com/v2-6d3d6845c4f69071ac140ed96d799ad1_1440w.jpg)

<p class="kb-image-caption">图例</p>

与其他神经网络一样，Transformer 模型不能直接处理原始文本，故使用分词器进行预处理。使用`AutoTokenizer`类及其`from_pretrained()`方法。
from transformers import AutoTokenizer checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)若要指定我们想要返回的张量类型（PyTorch, TensorFlow 或普通 NumPy），我们使用`return_tensors`参数raw_inputs = [     "I've been waiting for a HuggingFace course my whole life.", "I hate this so much!", ] inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") print(inputs) PyTorch 张量的结果：
输出本身是一个包含两个键的字典，`input_ids`和`attention_mask`。
{     'input_ids': tensor([         [  101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], [  101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0]     ]), 'attention_mask': tensor([         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]     ])

### 2）Model

Transformers 提供了一个`AutoModel`类，它也有一个`from_pretrained()`方法：
from transformers import AutoModel checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)如果我们将预处理过的输入提供给我们的模型，我们可以看到：
outputs = model(**inputs) print(outputs.last_hidden_state.shape)

# 输出
# torch.Size([2, 16, 768])
![Image 2](https://pica.zhimg.com/v2-f51f9dae359ec191229b35028d0897ca_1440w.jpg)

<p class="kb-image-caption">图例</p>

模型最后一层输出的原始**非标准化分数**。要转换为概率，它们需要经过一个[SoftMax](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Softmax_function)层（所有 Transformers 模型都输出 logits，因为用于训练的损耗函数一般会将最后的激活函数(如SoftMax)与实际损耗函数(如交叉熵)融合 。
import torch predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) print(predictions)

### [Models](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/3%3Ffw%3Dpt)
### 1）创建 Transformer

from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

### 2）不同的加载方式

from transformers import BertModel model = BertModel.from_pretrained("bert-base-cased")

### 3）保存模型

`model.save_pretrained("directory_on_my_computer")`

### 4）使用Transformer model
sequences = ["Hello!", "Cool.", "Nice!"] encoded_sequences = [     [101, 7592, 999, 102], [101, 4658, 1012, 102], [101, 3835, 999, 102], ] import torch model_inputs = torch.tensor(encoded_sequences)

### [Tokenizers](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/4%3Ffw%3Dpt)
### 1）Loading and saving

from transformers import BertTokenizer tokenizer = BertTokenizer.from_pretrained("bert-base-cased") tokenizer("Using a Transformer network is simple")

# 输出
'''

{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]} '''

# 保存

tokenizer.save_pretrained("directory_on_my_computer")

### 2）Tokenization

from transformers import AutoTokenizer tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence) print(tokens) # 输出 : ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

# 从token 到输入 ID

ids = tokenizer.convert_tokens_to_ids(tokens) print(ids) # 输出：[7993, 170, 11303, 1200, 2443, 1110, 3014]

### 3） Decoding

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014]) print(decoded_string) # 输出：'Using a Transformer network is simple'

### [处理多个序列](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/5%3Ffw%3Dpt)[Handling multiple sequences](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/5%3Ffw%3Dpt)
### 1) 模型需要一批输入 Models expect a batch of inputs
将数字列表转换为张量并将其发送到模型：
import torch from transformers import AutoTokenizer, AutoModelForSequenceClassification checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) model = AutoModelForSequenceClassification.from_pretrained(checkpoint) sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence) ids = tokenizer.convert_tokens_to_ids(tokens) input_ids = torch.tensor([ids]) print("Input IDs:", input_ids) output = model(input_ids) print("Logits:", output.logits)

# 输出
'''

Input IDs: `[[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]]` Logits: `[[-2.7276, 2.8789]]` '''

### 2) 填充输入 Padding the inputs
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) sequence1_ids = `[[200, 200, 200]]` sequence2_ids = `[[200, 200]]` batched_ids = [     [200, 200, 200], [200, 200, tokenizer.pad_token_id], ] print(model(torch.tensor(sequence1_ids)).logits) print(model(torch.tensor(sequence2_ids)).logits) print(model(torch.tensor(batched_ids)).logits)

# 输出
'''

tensor(`[[1.5694, -1.3895]]`, grad_fn=<AddmmBackward>) tensor(`[[0.5803, -0.4125]]`, grad_fn=<AddmmBackward>) tensor([[ 1.5694, -1.3895], [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>) '''

### 总结 [Putting it all together](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/6%3Ffw%3Dpt)
我们已经探索了分词器的工作原理，并研究了分词 tokenizers, 转换为输入 ID conversion to input IDs, 填充 padding, 截断 truncation和注意力掩码 attention masks.Transformers API 可以通过高级函数为我们处理所有这些。
tokenizer = AutoTokenizer.from_pretrained(checkpoint) sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)

# 可以标记单个序列

sequence = "I've been waiting for a HuggingFace course my whole life."

# 还可以一次处理多个序列

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"] model_inputs = tokenizer(sequences)

# 可以根据几个目标进行填充
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

# 还可以截断序列

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

# 可以处理到特定框架张量的转换，然后可以将其直接发送到模型。
# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays

model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

### Special tokens

> 分词器在开头添加特殊词[CLS]，在结尾添加特殊词[SEP]。

model_inputs = tokenizer(sequence) print(model_inputs["input_ids"]) tokens = tokenizer.tokenize(sequence) ids = tokenizer.convert_tokens_to_ids(tokens) print(ids)

# 输出
'''

[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102] [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012] '''
print(tokenizer.decode(model_inputs["input_ids"])) print(tokenizer.decode(ids))

# 输出
'''

"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
"i've been waiting for a huggingface course my whole life."
'''

# 总结：从分词器到模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint) model = AutoModelForSequenceClassification.from_pretrained(checkpoint) sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"] tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt") output = model(**tokens)本章测验：[Using Transformers - Hugging Face Course](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter2/8%3Ffw%3Dpt)如果觉得有帮助到你，可以**点个赞**鼓励下，让我继续更新~

## 相关笔记

[深度学习与AI（主题索引）](MOC-dl-ai.md)
[[Conv1d-与-Conv2d-区别|Conv1d 与 Conv2d 区别]]
[[GRU-与-LSTM|GRU 与 LSTM]]
[[RNN-到-Transformer|RNN 到 Transformer]]
[[Self-Attention-图解|Self-Attention 图解]]
[[Word-Embedding|Word Embedding]]
[[nn.Conv1d-详解|nn.Conv1d 详解]]
[[十分钟理解-Transformer|十分钟理解 Transformer]]
