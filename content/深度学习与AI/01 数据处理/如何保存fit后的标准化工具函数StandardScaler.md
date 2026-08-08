---
title: 如何保存fit后的标准化工具函数StandardScaler
url: https://blog.csdn.net/weixin_44022515/article/details/105973449
curated_at: '2026-06-28T20:00:00+00:00'
---

# 如何保存 fit 后的标准化工具函数 StandardScaler

训练模型时常用 `sklearn.preprocessing.StandardScaler` 标准化特征。测试/预测时必须在**与训练相同的尺度**下变换新数据；不必对测试集重新 `fit`，可将训练阶段拟合好的 scaler 序列化保存，预测时再加载。

## fit 并保存 scaler

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
import pickle

# 导入训练数据
allmatrix = pd.read_csv(mydata, header=0, index_col=0, low_memory=True).iloc[:, :2080].values
target = np.loadtxt(mylabel)

# 标准化训练数据
scaler = SS()
allmatrix = scaler.fit_transform(allmatrix)

# 保存标化器
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# ……训练模型，保存模型
```

## 预测前加载 scaler 变换

```python
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# 加载待预测数据
testdata = pd.read_csv(args.matrix, header=0, index_col=0, low_memory=True).values

# 用训练时的 scaler 变换（勿重新 fit）
scaler = pickle.load(open('scaler.pkl', 'rb'))
testdata = scaler.transform(testdata)

print("testdata shape: {}".format(testdata.shape))

# 转为 tensor 并加载模型预测
x = tf.convert_to_tensor(testdata)
network = tf.keras.models.load_model('your_model.h5', compile=True)
prediction = network.predict(x)
print(prediction)

np.savetxt("prediction_result.txt", prediction, fmt="%.4f")
```

要点：`transform` 使用的是训练集上 `fit` 得到的均值与方差；对测试/线上数据只能 `transform`，不能 `fit_transform`。

## 相关笔记

[[Pairs-Plot-Seaborn-可视化|Seaborn Pairs Plot 可视化]]
[[不平衡数据-测试集勿平衡-SO|不平衡数据：测试集勿平衡]]
