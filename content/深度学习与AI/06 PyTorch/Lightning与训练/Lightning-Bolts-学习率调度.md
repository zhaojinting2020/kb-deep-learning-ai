---
title: Lightning Bolts 学习率调度
url: https://pytorch-lightning-bolts.readthedocs.io/en/0.3.0/learning_rate_schedulers.html
fetch_source: internet_archive
fetched_at: '2026-06-27T20:05:06+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Lightning Bolts 学习率调度

This package lists common learning rate schedulers across research domains (This is a work in progress. If you have any learning rate schedulers you want to contribute, please submit a PR!)

Note
this module is a work in progress

## Your Learning Rate Scheduler¶
We’re cleaning up many of our learning rate schedulers, but in the meantime, submit a PR to add yours here!

## Linear Warmup Cosine Annealing Learning Rate Scheduler¶
*class*`pl_bolts.optimizers.lr_scheduler.``LinearWarmupCosineAnnealingLR`(*optimizer*,*warmup_epochs*,*max_epochs*,*warmup_start_lr=0.0*,*eta_min=0.0*,*last_epoch=-1*)[source]
- Bases: - `torch.optim.lr_scheduler._LRScheduler`- Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and eta_min. - Warning - It is recommended to call - `step()`for- `LinearWarmupCosineAnnealingLR`after each iteration as calling it after each epoch will keep the starting lr at warmup_start_lr for the first epoch which is 0 in most cases.- Warning - passing epoch to - `step()`is being deprecated and comes with an EPOCH_DEPRECATION_WARNING. It calls the- `_get_closed_form_lr()`method for this scheduler instead of- `get_lr()`. Though this does not change the behavior of the scheduler, when passing epoch param to- `step()`, the user should call the- `step()`function before calling train and validation methods.- Example - >>> layer = nn.Linear(10, 1) >>> optimizer = Adam(layer.parameters(), lr=0.02) >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40) >>> # >>> # the default case >>> for epoch in range(40): ... # train(...) ... # validate(...) ... scheduler.step() >>> # >>> # passing epoch param case >>> for epoch in range(40): ... scheduler.step(epoch) ... # train(...) ... # validate(...)

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-学习率调度|Lightning Bolts 学习率调度]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-工具箱|Lightning Bolts 工具箱]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Hooks-与-Callbacks|Lightning Hooks 与 Callbacks]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Hooks-与-Callbacks|Lightning Hooks 与 Callbacks]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/PyTorch-Lightning-完全攻略-zh|PyTorch Lightning 完全攻略]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/PyTorch-Lightning-完全攻略|PyTorch Lightning 完全攻略]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/ReduceLROnPlateau-学习率调度|ReduceLROnPlateau 学习率调度]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/CosineAnnealingLR-余弦退火|CosineAnnealingLR 余弦退火]]
