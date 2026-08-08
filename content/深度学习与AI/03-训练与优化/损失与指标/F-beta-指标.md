---
title: F-beta 指标
url: https://zhuanlan.zhihu.com/p/447023863#:~:text=%E4%B8%BA%E4%BA%86%E8%AE%A9%E5%8A%A0%E6%B7%B1%E6%88%91%E4%BB%AC%E5%8D%B0%E8%B1%A1%EF%BC%8C%E8%BF%99%E9%87%8C%E6%88%91%E4%BB%AC%E6%8A%8A%E7%9B%B4%E6%8E%A5%E6%94%BE%E4%B8%8A%E5%85%B3%E4%BA%8E%E7%B2%BE%E7%A1%AE%E7%8E%87%E5%92%8C%E5%8F%AC%E5%9B%9E%E7%8E%87%E7%9A%84%E8%A7%A3%E9%87%8A%EF%BC%8C%E5%A6%82%E6%9E%9C%E5%A4%A7%E5%AE%B6%E5%BF%98%E8%AE%B0%E7%9A%84%E8%AF%9D%EF%BC%8C%E4%BB%A5%E5%90%8E%E4%B8%8D%E5%A6%A8%E6%9D%A5%E5%A4%9A%E7%9C%8B%E7%9C%8B
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:51:38+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# F-beta 指标

> 在江西VTE风险预测和山东案件自动分发比赛中，笔者见到了F2-Score评估指标，此类指标与以往F1-Score不同，出题方选择使用不同的beta权重来更加侧重Precision或者Recall某一指标，所以在实际中常常需要根据具体情况做出取舍，例如一般的搜索情况，在保证召回率的条件下，尽量提升精确率。而像癌症检测, 地震检测, 金融欺诈等，则在保证精确率的条件下，尽量提升召回率。为了让加深我们印象，这里我们把直接放上关于精确率和召回率的解释，如果大家忘记的话，以后不妨来多看看
*   **精确率是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。**
*   **召回率是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了。**

[Fbeta-measure](https://zhida.zhihu.com/search?content_id=187468302&content_type=Article&match_order=1&q=Fbeta-measure&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU0ODMsInEiOiJGYmV0YS1tZWFzdXJlIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTg3NDY4MzAyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.MBAV8Rqcqs94q_AgXlMAUUwj_s6FJGbd_cPUZWlV2QA&zhida_source=entity) 是一种可配置的单分指标，用于根据对正类的预测来评估二元分类模型。

Fbeta-measure 是使用精度和召回率计算的。精度是计算正类的正确预测百分比的指标。Recall计算所有可能做出的正面预测中正面类别的正确预测的百分比。最大化精度将最小化假阳性错误，而最大化召回将最小化假阴性错误。的F值被计算为的精确度和召回的调和平均，每一种有相同的加权。它允许使用单个分数同时考虑精度和召回来评估模型，这在描述模型的性能和比较模型时很有帮助。所述Fbeta是F值增加了β的配置参数的概括。默认的 beta 值为 1.0，这与 F-measure 相同, 也就是我们常见的F1-Score. 较小的 Beta 值，例如 0.5，在计算分数时赋予精度更高的权重而较少召回率，而较大的 Beta 值（例如 2.0）赋予精度较低的权重和较高的召回率权重。

F-Score=(1+\beta ^2)\cdot \frac {Precision\cdot Recall} {\beta^2\cdot Precision + Recall} \\当准确率和召回率都很重要，但需要侧重其中一个时，例如当假阴性比假阳性更重要时，或者相反时，Fbtea将会是一个很有用的指标。

## 精确率和召回率
在我们深入研究 Fbeta指标之前，我们还是要回顾用于评估分类模型所做预测的精确率和召回率度量的基础知识。

### [混淆矩阵](https://zhida.zhihu.com/search?content_id=187468302&content_type=Article&match_order=1&q=%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTU0ODMsInEiOiLmt7fmt4bnn6npmLUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxODc0NjgzMDIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.QGNcG-y3iikSwDroCDr9UITB3cvUm7nGTobMO4QX94w&zhida_source=entity)
【混淆矩阵】总结了通过为每个类的模型进行的预测，和到这些预测实际上属于的类的数量，它有助于了解模型产生的预测错误的类型。最简单的混淆矩阵是针对二类分类问题，具有负（0 类）和正（1 类）类。在这种类型的混淆矩阵中，表格中的每个单元格都有一个特定且易于理解的名称，总结如下：
| Positive Prediction | Negative Prediction Positive Class | True Positive (TP)  | False Negative (FN) Negative Class | False Positive (FP) | True Negative (TN)精确率和召回率指标是根据混淆矩阵中的单元格定义的，特别是像真阳性和假阴性这样的术语。

### 精确率
精确率是一种量化正确预测数量的指标。它的计算方法是正确预测的正例的个数除以预测的正例总数。
precision= TruePositives / (TruePositives + FalsePositives) \\结果是一个介于 0.0（无精度）和 1.0（完全或完美精度）之间的值。精确率的直觉是它不关心假阴性，它最大限度地减少了假阳性。我们可以用下面的一个小例子来证明这一点。

# 导入
from sklearn.metrics import precision_score

# no precision
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] score = precision_score(y_true, y_pred) print('No Precision: %.3f' % score)

# 一些假阳性
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1] score = precision_score(y_true, y_pred) print('Some False Positives: %.3f' % score)

# 一些假阴性
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] score = precision_score(y_true, y_pred) print('Some False Negatives: %.3f' % score)

# 完美精确率
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] score = precision_score(y_true, y_pred) print('Perfect Precision: %.3f' % score)运行示例演示了计算所有不正确和所有正确预测类标签的精度，分别显示无精度(精确率为0)和完美精度（精确率为1）。预测某些错误正样本的示例显示精确率会显著下降，突出表明该指标与最小化false positives有关。预测一些假阴性的示例显示出100%的精确率，突出表明该度量与假阴性无关No Precision: 0.000 Some False Positives: 0.714 Some False Negatives: 1.000 Perfect Precision: 1.000

### 召回率
Recall 是一个度量从可能做出的所有正面预测中做出的正确正样本预测的数量。它的计算方法是正确预测的正例的比率除以可预测的正例总数。

Recall= TruePositives / (TruePositives + FalseNegatives) \\结果是一个介于 0.0（无召回）和 1.0（完全或完美召回）之间的值。召回的直觉是它不关心 false positives，它最大限度地减少了false negatives. 我们可以用下面的一个小例子来证明这一点。

# intuition for recall
from sklearn.metrics import recall_score

# 没有召回
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] score = recall_score(y_true, y_pred) print('No Recall: %.3f' % score)

# 一些假阳性
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1] score = recall_score(y_true, y_pred) print('Some False Positives: %.3f' % score)

# 一些假阴性
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] score = recall_score(y_true, y_pred) print('Some False Negatives: %.3f' % score)

# 完美召回
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] score = recall_score(y_true, y_pred) print('Perfect Recall: %.3f' % score)运行该示例演示了计算所有不正确和所有正确预测类别标签的召回率，分别显示无召回率和完美召回率。预测某些误报的正样本示例显示了完美的召回率，突出表明该度量与假阳性无关。预测一些假阴性的示例显示召回率下降，突出表明该措施与最小化假阴性有关。

No Recall: 0.000 Some False Positives: 1.000 Some False Negatives: 0.600 Perfect Recall: 1.000现在我们熟悉了精确率和召回率，让我们回顾一下 F-measure.

## F-measure
准确率和召回率衡量了正类可能出现的两种错误类型。最大限度地提高精确率可以最大限度地减少假阳性，最大限度地提高召回率可以最大限度地减少假阴性。 F-Measure 或 F-Score 提供了一种将精度和召回率结合到一个能够同时捕获这两个属性的度量中的方法。

F-Measure = (2 * Precision * Recall) / (Precision + Recall) \\这是两个精确率和召回率的**[调和平均值](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Harmonic_mean)**。结果是一个介于最差 F 测量的 0.0 和完美 F 测量的 1.0 之间的值。

F-measure 的直觉是这两个度量在重要性上是平衡的，只有良好的精度和良好的召回率共同导致良好的 F-measure.

### 最差情况
首先，如果所有样本都被刚好错误预测，我们的精度和召回率将为零，从而导致 F-measure 为零；例如：

# worst case f-measure
from sklearn.metrics import f1_score from sklearn.metrics import precision_score from sklearn.metrics import recall_score

# no precision or recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] p = precision_score(y_true, y_pred) r = recall_score(y_true, y_pred) f = f1_score(y_true, y_pred) print('No Precision or Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))运行该示例，我们可以看到在最坏情况下的 F 度量中没有任何精度或召回率。

No Precision or Recall: p=0.000, r=0.000, f=0.000

### 最好情况
相反，完美的预测将导致完美的精确度和召回率，进而获得完美的 F 度量，例如：

# best case f-measure
# perfect precision and recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] p = precision_score(y_true, y_pred) r = recall_score(y_true, y_pred) f = f1_score(y_true, y_pred) print('Perfect Precision and Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))运行这个例子，我们可以看到完美的精度和召回导致完美的 F-measure.

Perfect Precision and Recall: p=1.000, r=1.000, f=1.000

### 50% 准确率，100%召回
不可能有完美的精确度而没有召回，或者没有精确度和完美的召回。准确率和召回率都需要预测真阳性。考虑我们为所有情况预测正类的情况。这将为我们提供 50% 的准确率，因为一半的预测是误报。它会给我们完美的回忆，因为我们不会出现假阴性。对于我们在示例中使用的平衡数据集，一半的预测是真阳性，一半是假阳性；因此，精度比将为 0.5% 或 50%。将 50 感知精度与完美召回相结合将导致惩罚 F 度量，特别是介于 50% 和 100% 之间的调和平均值。下面的示例演示了这一点。

# perfect precision f-measure
# 50% precision， 100% recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] p = precision_score(y_true, y_pred) r = recall_score(y_true, y_pred) f = f1_score(y_true, y_pred) print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))运行这个例子证实 50 的精确率和完美的召回率，并且 F 分数的结果约为 0.667.

Result: p=0.500, r=1.000, f=0.667

## Fbeta-Measure
F-measure 平衡了准确率和召回率。在某些问题上，我们可能对更加关注精度的 F 度量感兴趣，例如当假阳性更重要以最小化时，但假阴性仍然很重要。在其他问题上，我们可能对更关注召回的 F 度量感兴趣，例如当假阴性更重要以最小化时，但假阳性仍然很重要。解决方案是 Fbeta-Measure.

Fbeta 度量是 F 度量的抽象，其中**[调和均值](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Harmonic_mean)**计算中的精度和召回率的平衡由称为 _beta_ 的系数控制。
*   Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)

beta 参数的选择将用于 Fbeta-measure 的名称。例如，beta 值为 2 被称为 F2-measure 或 F2-score.Beta 值为 1 被称为 F1-measure 或 F1-score.
beta 参数的三个常见值如下：
*   **F0.5-Measure** (beta=0.5)：在精度上的权重更大，召回的权重更小。
*   **F1-Measure** (beta=1.0)：平衡准确率和召回率的权重。
*   **F2-Measure** (beta=2.0)：精度权重较小，召回权重较大

起初，不同 beta 值对计算的影响并不直观。让我们仔细看看这些例子中的每一个。

### F1-measure
上一节中讨论的 F-measure 是Beta值为 1的 Fbeta-measure 的示例。具体来说，F-measure和F1-measure计算的东西是一样的；例如：
F-Measure = ((1 + 1^2) * Precision * Recall) / (1^2 * Precision + Recall) F-Measure = (2 * Precision * Recall) / (Precision + Recall)考虑我们有 50 %精确率和100%召回率的情况。我们可以手动计算这种情况下的 F1 度量，如下所示：
F-Measure = (2 * Precision * Recall) / (Precision + Recall) F-Measure = (2 * 0.5 * 1.0) / (0.5 + 1.0) F-Measure = 1.0 / 1.5 F-Measure= 0.666我们可以使用scikit-learn 中的**[fbeta_score() 函数](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)**将“ _beta_ ”参数设置为 1.0来确认这个计算.下面列出了完整的示例。

# calculate the f1-measure
from sklearn.metrics import fbeta_score from sklearn.metrics import precision_score from sklearn.metrics import recall_score

# perfect precision, 50% recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] p = precision_score(y_true, y_pred) r = recall_score(y_true, y_pred) f = fbeta_score(y_true, y_pred, beta=1.0) print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))这个 F1-measure 值 0.667 与上面中计算的 F-measure 相匹配。

### F0.5-Measure
F0.5-measure 是Beta值为 0.5的 Fbeta-measure 的一个示例。它具有提高精确率的重要性和降低召回率的重要性的效果。如果最大化精确率最小化假阳性且最大化召回率最小化假阴性，那么F0.5 度量更关注最小化假阳性而不是最小化假阴性。

F0.5-Measure 计算如下：
F0.5-Measure = ((1 + 0.5^2) * Precision * Recall) / (0.5^2 * Precision + Recall) F0.5-Measure = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)考虑我们有 50% 的准确率和完美召回率的情况。我们可以手动计算这种情况下的 F0.5 度量，如下所示：
F0.5-Measure = (1.25 * Precision * Recall) / (0.25 * Precision + Recall) F0.5-Measure = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) F0.5-Measure = 0.625 / 1.125 F0.5-Measure = 0.555我们预计 0.5 的 beta 值会导致这种情况下的得分较低，因为精确率得分较低且召回率非常好。这正是我们所看到的，对于 F1 分数计算为 0.667 的相同场景，F0.5 度量达到了 0.555. 精确率在计算中发挥了更大的作用。我们可以确认这个计算；下面列出了完整的示例。

# calculate the f0.5-measure
from sklearn.metrics import fbeta_score from sklearn.metrics import f1_score from sklearn.metrics import precision_score from sklearn.metrics import recall_score y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] p = precision_score(y_true, y_pred) r = recall_score(y_true, y_pred) f = fbeta_score(y_true, y_pred, beta=0.5) print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))运行该示例确认精度和召回值，然后报告 F0.5 测量值为 0.556（四舍五入），与我们手动计算的值相同。

Result: p=0.500, r=1.000, f=0.556

## F2-measure
F2-measure 是Beta值为 2.0的 Fbeta-measure 的一个示例。它具有降低精度重要性和增加召回重要性的效果。如果最大化精度最小化误报，最大化召回率最小化漏报，那么F2 度量更关注最小化漏报而不是最小化误报。

F2-measure 计算如下：
F2-Measure = ((1 + 2^2) * Precision * Recall) / (2^2 * Precision + Recall) F2-Measure = (5 * Precision * Recall) / (4 * Precision + Recall)考虑我们有 50% 的准确率和完美召回率的情况。我们可以手动计算这种情况下的 F2 度量，如下所示：
F2-Measure = (5 * Precision * Recall) / (4 * Precision + Recall) F2-measure  = (5 * 0.5 * 1.0) / (4 * 0.5 + 1.0) F2-measure  = 2.5 / 3.0 F2-measure  = 0.833我们预计2.0的beta值会在这种情况下导致更高的分数，因为召回有一个完美的分数，这将比精度表现不佳的分数更高。正是我们看到的，对于 F1 分数计算为 0.667 的相同场景，F2 度量达到 0.833. 召回在计算中发挥了更大的作用。我们可以确认这个计算；下面列出了完整的示例。

# calculate the f2-measure
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] p = precision_score(y_true, y_pred) r = recall_score(y_true, y_pred) f = fbeta_score(y_true, y_pred, beta=2.0) print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))运行该示例确认精度和召回值，然后报告 F2-measure 为 0.883，与我们手动计算的值相同（四舍五入）。
`Result: p=0.500, r=1.000, f=0.833`

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[Keras-loss-与-metrics|Keras loss 与 metrics]]
[[Label-Smoothing|Label Smoothing]]
[[Loss-与-Metric-区别|Loss 与 Metric 区别]]
[[Metric-能否当-Loss|Metric 能否当 Loss]]
[[交叉熵损失|交叉熵损失]]
[[分类评估指标|分类评估指标]]
[[ROC-与-PR-曲线|ROC 与 PR 曲线]] — _损失 / 指标 / 不平衡_
[[不平衡专用损失|不平衡专用损失]] — _损失 / 指标 / 不平衡_
