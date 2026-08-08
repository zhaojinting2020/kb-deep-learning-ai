---
title: 不平衡数据：测试集勿平衡
url: https://stackoverflow.com/questions/55921286/should-i-balance-the-test-set-when-i-have-highly-unbalanced-data
polished_at: '2026-06-27T18:51:39+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# 不平衡数据：测试集勿平衡

I am using Sklearn `GridSearchCv` to find the best parameters for a random forest when applied to remote sensing data with 4 classes (buildings, vegetation, water and roads), the problem is I have a lot more "vegetation" classes than the rest (by a lot I mean a difference from thousands to several millions). Should I balance my testing dataset to obtain the metrics?

I already balance the whole set before i split into training and testing, this means that both datasets have the same distribution of classes in a equal manner. I am afraid this does not represent the algorithm's performance on real data, but it gives me a insight of the performance per class. If i use unbalanced data, the "vegetation" class might end up messing with the other averages.

Here's the example of the balance i do, as you can see I do it on the X and y directly. Which are the full data and labels.
if balance:
    smt = RandomUnderSampler(sampling_strategy='auto')     X, y = smt.fit_sample(X, y)     print("Features array shape after balance: " + str(X.shape)) I want to have the best understanding of the model's performance on the real data, but I have not found conclusive answers for this!

## 1 Answer 1
The thumb rule of dealing with imbalenced data is "Never ever balance the test data". the pipeline of dealing with imbalance data:
1.   Do preprocess
2.   Apply train test split(Stratified).
3.   Balance the training data (Generally SMOTE works better)
4.   Train model/models
5.   Test on imbalance test data(Obviously use metrics like [f-score](https://en.wikipedia.org/wiki/F1_score), [Precision, Recall](https://en.wikipedia.org/wiki/Precision_and_recall))

So that you will get the actual performance.

The question arises here is why not to balance data before train test split?

You can't expect the real world data to be balanced when you are deploying in the real world right...

A better way is to use K-fold at step 2 and do the 3,4,5 steps for each fold Refer to [this](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/) article for more info.

## 相关笔记

[深度学习与AI（主题索引）](../../../index/MOC-dl-ai.md)
[[Pairs-Plot-Seaborn-可视化|Seaborn Pairs Plot 可视化]]
[[如何保存fit后的标准化工具函数StandardScaler|如何保存fit后的标准化工具函数StandardScaler]]
