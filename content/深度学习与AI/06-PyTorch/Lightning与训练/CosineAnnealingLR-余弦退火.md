---
title: CosineAnnealingLR 余弦退火
url: https://blog.csdn.net/qq_29007291/article/details/126094939
curated_at: '2026-06-28T20:00:00+00:00'
---

# CosineAnnealingLR 余弦退火

PyTorch 学习率调度器 `CosineAnnealingLR`：学习率按余弦曲线从初值衰减到 `eta_min`，无需像 `StepLR` 那样手工设阶梯，实践中往往效果稳定。

> 下文示例基于 PyTorch 1.x 语义；新版本 API 名称一致，细节以官方文档为准。

## 构造函数

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max,
    eta_min=0,
    last_epoch=-1,
)
```

| 参数 | 含义 |
|------|------|
| `optimizer` | 绑定的优化器 |
| `T_max` | 余弦**半周期**内的 scheduler 步数（见下） |
| `eta_min` | 学习率下限 |
| `last_epoch` | 恢复训练时的 epoch 计数 |

更新公式（第 \(t\) 步，\(t=0, \ldots, T_{\max}\)）：

\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\frac{t\pi}{T_{\max}}\right)
\]

![余弦退火曲线示意](https://i-blog.csdnimg.cn/blog_migrate/12cce09aa563d8de63acda3ffc5afca2.png)

![与 StepLR 等策略对比](https://i-blog.csdnimg.cn/blog_migrate/913814077378e28663016428c6af1b36.png)

## 每个 epoch 更新一次

`scheduler.step()` 放在**每个 epoch 结束**（训练 loop 外层）：

```python
import torch
from torch import nn, optim
from torchvision import models
import matplotlib.pyplot as plt

model = models.resnet18()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-6
)

lrs = []
for epoch in range(200):
    train_one_epoch(model, optimizer)  # 省略
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

plt.plot(lrs)
plt.xlabel('epoch')
plt.ylabel('learning rate')
```

此时 `T_max` = 总 epoch 数（一个完整余弦从 \(\eta_0\) 到 \(\eta_{\min}\)）。

## 每个 iteration（batch）更新一次

若希望**每个 batch** 调一次学习率，应使用 `CosineAnnealingWarmRestarts`，或在 inner loop 里对 `CosineAnnealingLR` 逐步 `step()`，并令 `T_max` = 总 iteration 数（`epochs × steps_per_epoch`）。

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_iters, eta_min=1e-6
)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
        scheduler.step()  # 每个 batch 一步
```

## 使用注意

- `T_max` 指 scheduler **调用 `step()` 的次数**，不是"epoch 数"本身，除非你在每个 epoch 只 step 一次。
- 恢复训练时传入正确的 `last_epoch`，或保存/加载 scheduler 状态。
- 常与 warmup, weight decay 分组等策略组合；纯余弦往往已足够作为默认 schedule.

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)

[[PyTorch-Lightning-学习率调度|PyTorch Lightning 学习率调度]]
