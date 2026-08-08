---
title: Keras loss 与 metrics
url: https://stackoverflow.com/questions/51256695/loss-metrics-and-scoring-in-keras
fetch_source: stackexchange:api
fetched_at: '2026-06-28T08:17:04+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
math_repaired_at: '2026-06-27T20:24:23+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Keras loss 与 metrics

## Question

What is the difference between `loss`, `metrics` and `scoring` in building a `keras` model? Should they be different or same? In a typical model, we use all of the three for`GridSearchCV`.

Here is the snapshot of a typical model for regression which uses all the three.
```
def create_model():
 model = Sequential()
 model.add(Dense(12, input_dim=1587, activation='relu'))
 model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
 return model
model = KerasRegressor(build_fn=create_model, verbose=0)
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model,param_grid=param_grid, scoring='r2' n_jobs=-1)
grid_result = grid.fit(X, Y)
```
## Answer 1 (score: 7)
No, they are all different things used for different purposes in your code.
There are two parts in your code.
1) Keras part:
```
model.compile(loss='mean_squared_error',
               optimizer='adam',
               metrics=['mean_squared_error'])
```
a) `loss`: In the [Compilation section of the documentation here](https://keras.io/getting-started/sequential-model-guide/), you can see that:
A loss function is the objective that the model will try to minimize.

So this is actually used together with the `optimizer` to actually train the model
b) `metrics`: According to the [documentation](https://keras.io/metrics/):
A metric function is similar to a loss function, except that the results from evaluating a metric are not used when training the model.

This is only used to report the metrics so that the used (you) can judge the performance of model. It does not impact how the model is trained.
2) Grid-search part:

`scoring`: Again, check [the documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
A single string or a callable to evaluate the predictions on the test set.

This is used to find the combination of parameters which you defined in `param_grid` which gives the best `score`.

They can very well (in most cases) be different, depending on what you want.

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[Loss-与-Metric-区别|Loss 与 Metric 区别]]
[[Metric-能否当-Loss|Metric 能否当 Loss]]
[[F-beta-指标|F-beta 指标]]
[[Label-Smoothing|Label Smoothing]]
[[交叉熵损失|交叉熵损失]]
[[分类评估指标|分类评估指标]]
[[ROC-与-PR-曲线|ROC 与 PR 曲线]] — _损失 / 指标 / 不平衡_
[[不平衡专用损失|不平衡专用损失]] — _损失 / 指标 / 不平衡_
