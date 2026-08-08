---
title: Word Embedding
url: https://zhuanlan.zhihu.com/p/384452959
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:56:13+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Word Embedding

## 前言

2013年，Word2Vec横空出世，自然语言处理领域各项任务效果均得到极大提升。自从Word2Vec这个神奇的算法出世以后，导致了一波嵌入（Embedding）热，基于句子, 文档表达的word2vec, doc2vec算法，基于物品序列的item2vec算法，基于图模型的图嵌入技术相继诞生。图嵌入（Graph Embedding，也叫Network Embedding）是一种将图数据（通常为高维稠密的矩阵）映射为低微稠密向量的过程，能够很好地解决图数据难以高效输入机器学习算法的问题。当前比较知名的图嵌入方法有[DeepWalk](https://zhida.zhihu.com/search?content_id=173564634&content_type=Article&match_order=1&q=DeepWalk&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3NjAsInEiOiJEZWVwV2FsayIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjE3MzU2NDYzNCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.Rze9Bl3wRDFnz6TLVM4u4-aD1ZUrn08nujFZshY1PPY&zhida_source=entity), Line和Node2vec，这些都是基于顶点对相似度的图表示学习，仅仅保留了一部分的图的特性。现有的机器学习方法往往无法直接处理文本数据，因此需要找到合适的方法，将文本数据转换为数值型数据，由此引出了Word Embedding（词嵌入）的概念。词嵌入是自然语言处理（NLP）中语言模型与表征学习技术的统称，它是NLP里的早期预训练技术。它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量，这也是分布式表示：向量的每一维度都没有实际意义，而整体代表一个具体概念。分布式表示相较于传统的独热编码（one-hot）表示具备更强的表示能力，而独热编码存在维度灾难和语义鸿沟（不能进行相似度计算）等问题。传统的分布式表示方法，如矩阵分解（SVD/LSA）, [LDA](https://zhida.zhihu.com/search?content_id=173564634&content_type=Article&match_order=1&q=LDA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3NjAsInEiOiJMREEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzM1NjQ2MzQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.RK-arXQcc5cGva_BHfxID8E-06MGFDGXUwu2sVRE-DE&zhida_source=entity)等均是根据全局语料进行训练，是机器学习时代的产物。

Word Embedding的输入是原始文本中的一组不重叠的词汇，假设有句子：apple on a apple tree. 那么为了便于处理，我们可以将这些词汇放置到一个dictionary里，例如：[“apple”, “on”, “a”, “tree”]，这个dictionary就可以看作是Word Embedding的一个输入。

Word Embedding的输出就是每个word的向量表示。对于上文中的原始输入，假设使用最简单的one hot编码方式，那么每个word都对应了一种数值表示。例如，apple对应的vector就是[1, 0, 0, 0]，a对应的vector就是[0, 0, 1, 0]，各种机器学习应用可以基于这种word的数值表示来构建各自的模型。当然，这是一种最简单的映射方法，但却足以阐述Word Embedding的意义。文本表示的类型：
*   基于one-hot, tf-idf, textrank等的bag-of-words；
*   主题模型：LSA（SVD）, pLSA, LDA；
*   基于词向量的固定表征：word2vec, fastText, glove
*   基于词向量的动态表征：[ELMO](https://zhida.zhihu.com/search?content_id=173564634&content_type=Article&match_order=1&q=ELMO&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU3NjAsInEiOiJFTE1PIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTczNTY0NjM0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.4VxwyxiNdc7DUQzlPzrhCz0d9gOl0yKdLz0XQyl1qbc&zhida_source=entity), GPT, bert

上面给出的4个类型也是nlp领域最为常用的文本表示了，文本是由每个单词构成的，而谈起词向量，one-hot是可认为是最为简单的词向量，但存在维度灾难和语义鸿沟等问题；通过构建共现矩阵并利用SVD求解构建词向量，则计算复杂度高；而早期词向量的研究通常来源于语言模型，比如NNLM和RNNLM，其主要目的是语言模型，而词向量只是一个副产物。

## 使Embedding空前流行的Word2Vec
### Word2Vec必读paper

**1.**[[Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BWord2Vec%255D%2520Efficient%2520Estimation%2520of%2520Word%2520Representations%2520in%2520Vector%2520Space%2520%2528Google%25202013%2529.pdf)
Google的Tomas Mikolov提出word2vec的两篇文章之一，这篇文章更具有综述性质，列举了NNLM, RNNLM等诸多词向量模型，但最重要的还是提出了CBOW和Skip-gram两种word2vec的模型结构。虽然词向量的研究早已有之，但不得不说还是Google的word2vec的提出让词向量重归主流，拉开了整个embedding技术发展的序幕。
**2**. [[Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BWord2Vec%255D%2520Distributed%2520Representations%2520of%2520Words%2520and%2520Phrases%2520and%2520their%2520Compositionality%2520%2528Google%25202013%2529.pdf)
Tomas Mikolov的另一篇word2vec奠基性的文章。相比上一篇的综述，本文更详细的阐述了Skip-gram模型的细节，包括模型的具体形式和 Hierarchical Softmax和 Negative Sampling两种可行的训练方法。
**3**. [[Word2Vec] Word2vec Parameter Learning Explained (UMich 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BWord2Vec%255D%2520Word2vec%2520Parameter%2520Learning%2520Explained%2520%2528UMich%25202016%2529.pdf)
虽然Mikolov的两篇代表作标志的word2vec的诞生，但其中忽略了大量技术细节，如果希望完全读懂word2vec的原理和实现方法，比如词向量具体如何抽取，具体的训练过程等，强烈建议大家阅读UMich Xin Rong博士的这篇针对word2vec的解释性文章。

Word2Vec算法原理：
> skip-gram: 用一个词语作为输入，来预测它周围的上下文
>

> cbow: 拿一个词语的上下文作为输入，来预测这个词语本身

![Image 1](https://pic3.zhimg.com/v2-df9e2da063fea6ee9431571007c7dee8_1440w.jpg)

<p class="kb-image-caption">图例</p>

隐层的激活函数是线性的，相当于没做任何处理（这也是 Word2vec 简化之前语言模型的独到之处）。当模型训练完后，最后得到的其实是神经网络的权重【输入层到隐层, 隐层到输出层】，这就是生成的词向量，词向量的维度等于隐层的节点数。注意，由输入层到隐层的网络权重（输入向量）以及隐层到输出层的网络权重（输出向量）均可以作为词向量，一般我们用“输入向量”。需要提到一点的是，这个词向量的维度（与隐含层节点数一致）一般情况下要远远小于词语总数 V 的大小，所以 Word2vec 本质上是一种降维操作—把词语从 one-hot encoder 形式的表示降维到 Word2vec 形式的表示。

### Skip-gram更一般的情形

当 y 有多个词时，网络结构如下：
![Image 3](https://pica.zhimg.com/v2-8527de73c55b2066429fa894c6102392_1440w.jpg)

<p class="kb-image-caption">图例</p>

Word2vec 本质上是一个语言模型，它的输出节点数是 V 个，对应了 V 个词语，本质上是一个多分类问题，但实际当中，词语的个数非常非常多，会给计算造成很大困难，所以需要用技巧来加速训练。为了更新输出向量的参数，我们需要先计算误差，然后通过反向传播更新参数。在计算误差是我们需要遍历词向量的所有维度，这相当于遍历了一遍单词表，碰到大型语料库时计算代价非常昂贵。要解决这个问题，有三种方式：
Hierarchical Softmax：通过 Hierarchical Softmax 将复杂度从 O(n) 降为 O(log n)；Sub-Sampling Frequent Words：通过采样函数一定概率过滤高频单词；Negative Sampling：直接通过采样的方式减少负样本。

### Application

Word2vec 主要原理是根据上下文来预测单词，一个词的意义往往可以从其前后的句子中抽取出来。

Word2vec 已经应用于多个领域，并取得了巨大成功：
*   Airbnb 将用户的浏览行为组成 List，通过 Word2Vec 方法学习 item 的向量，其点击率提升了 21%，且带动了 99% 的预定转化率；
*   Yahoo 邮箱从发送到用户的购物凭证中抽取商品并组成 List，通过 Word2Vec 学习并为用户推荐潜在的商品；
### Conclusion

简单总结一下： Word2Vec 是一个词向量开源工具，包括 Skip-Gram 和 CBOW 的两种算法，加速训练的方法有：Hierarchical Softmax, Sub-Sampling 和 Negative Sampling.
*   Skip-Gram：利用中心词预测上下文；
*   CBOW：利用上下文预测中心词，速度比 Skip-Gram 快；
*   Hierarchical Softmax：引入 Hierarchical 加速 Softmax 的计算过程，对词频低的友好；
*   Sub-Sampling：依据词频进行采样，对词频低的友好；
*   Negative Sampling：通过负采样避免更新全部参数，对词频高的友好；
## word2vec vs glove

GloVe与word2vec，两个模型都可以根据词汇的“共现co-occurrence”信息，将词汇编码成一个向量（所谓共现，即语料中词汇一块出现的频率）。两者最直观的区别在于，word2vec是“predictive”的模型，而GloVe是“count-based”的模型。具体是什么意思呢？

Predictive的模型，如Word2vec，根据context预测中间的词汇，要么根据中间的词汇预测context，分别对应了word2vec的两种训练方式cbow和skip-gram. 对于word2vec，采用三层神经网络就能训练，最后一层的输出要用一个Huffuman树进行词的预测。

Count-based模型，如GloVe，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context. 共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss.(这里再插一句，降维或者重构的本质是什么？我们选择留下某个维度和丢掉某个维度的标准是什么？Find the lower-dimensional representations which can explain most of the variance in the high-dimensional data，这其实也是PCA的原理). 两个模型在并行化上有一些不同，即GloVe更容易并行化，所以对于较大的训练数据，GloVe更快。
*   word2vec是局部语料库训练的，其特征提取是基于滑窗的；而glove的滑窗是为了构建co-occurance matrix，是基于全局语料的，可见glove需要事先统计共现概率；因此，word2vec可以进行在线学习，glove则需要统计固定语料信息。
*   word2vec是无监督学习，同样由于不需要人工标注；glove通常被认为是无监督学习，但实际上glove还是有label的，即共现次数 log(X_{ij}).
*   word2vec损失函数实质上是带权重的交叉熵，权重固定；glove的损失函数是最小平方损失函数，权重可以做映射变换。
*   总体来看，**glove可以被看作是更换了目标函数和权重函数的全局word2vec**。

![Image 5](https://pica.zhimg.com/v2-004df09bcc2f085c72cc0938c08b1910_1440w.jpg)

<p class="kb-image-caption">图例</p>

如上图所示，比如多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

## 演进和发展

word embedding得到的词向量是固定表征的，无法解决一词多义等问题，因此引入基于语言模型的动态表征方法：ELMO, GPT, bert，以ELMO为例：
针对多义词问题，ELMO提供了一种简洁优雅的解决方案，ELMO是“Embedding from Language Models”的简称（论文：Deep contextualized word representation）。ELMO的本质思想是：事先用语言模型学好一个单词的Word Embedding，此时多义词无法区分，但在实际使用Word Embedding的时候，单词已经具备了特定的上下文了，这个时候可以根据上下文单词的语义去调整单词的Word Embedding表示，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以ELMO本身是个根据当前上下文对Word Embedding动态调整的思路。
![Image 7](https://pic1.zhimg.com/v2-542e2524b412705c37b10f32ba6258ae_1440w.jpg)

<p class="kb-image-caption">图例</p>

基于word2vec的一系列embedding方法主要是基于序列进行embedding，在当前商品, 行为, 用户等实体之间的关系越来越复杂化, 网络化的趋势下，原有sequence embedding方法的表达能力受限，因此Graph Embedding方法的研究和应用成为了当前的趋势。

DeeoWalk (SBU 2014)普遍被认为是Graph Embedding的baseline方法，用极小的代价完成从word2vec到graph embedding的转换和工程尝试；此后提出的LINE(MSRA 2015)，相比DeepWalk纯粹随机游走的序列生成方式，LINE可以应用于有向图, 无向图以及边有权重的网络，并通过将一阶, 二阶的邻近关系引入目标函数，能够使最终学出的node embedding的分布更为均衡平滑，避免DeepWalk容易使node embedding聚集的情况发生。node2vec (Stanford 2016)对DeepWalk随机游走方式的改进。为了使最终的embedding结果能够表达网络局部周边结构和整体结构，其游走方式结合了深度优先搜索和广度优先搜索。相比于node2vec对游走方式的改进，SDNE模型 (THU 2016)主要从目标函数的设计上解决embedding网络的局部结构和全局结构的问题。而相比LINE分开学习局部结构和全局结构的做法，SDNE一次性的进行了整体的优化，更有利于获取整体最优的embedding.
> 这里是本人 [深度学习笔记](https://www.zhihu.com/column/c_1392516988472721409)的第一篇文章，也是尝试和探索在深度学习领域知识输出的开始，感谢前人的输入和信息整合。水平有限，欢迎大家交流吐槽。参考资料：
[从Word Embedding到Bert模型-自然语言处理中的预训练技术发展史](https://link.zhihu.com/?target=https%3A//blog.csdn.net/malefactor/article/details/83961886) [Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184) [万物皆Embedding，从经典的word2vec到深度学习基本操作item2vec](https://zhuanlan.zhihu.com/p/53194407) [Word Embedding 知识总结](https://link.zhihu.com/?target=https%3A//blog.csdn.net/savinger/article/details/89308831) [NLP必读 | 十分钟读懂谷歌BERT模型](https://link.zhihu.com/?target=https%3A//www.jianshu.com/p/4dbdb5ab959b) [NLP预训练模型的全面总结](https://zhuanlan.zhihu.com/p/115014536) [【Graph Embedding】Word2Vec：词嵌入详解](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_27075943/article/details/104315836)

## 相关笔记

[深度学习与AI（主题索引）](MOC-dl-ai.md)
[[Conv1d-与-Conv2d-区别|Conv1d 与 Conv2d 区别]]
[[GRU-与-LSTM|GRU 与 LSTM]]
[[Huggingface-Transformers-教程|Huggingface Transformers 教程]]
[[RNN-到-Transformer|RNN 到 Transformer]]
[[Self-Attention-图解|Self-Attention 图解]]
[[nn.Conv1d-详解|nn.Conv1d 详解]]
[[十分钟理解-Transformer|十分钟理解 Transformer]]
