---
title: ReduceLROnPlateau 学习率调度
url: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
fetch_source: agent_reach:agent_reach:jina
fetched_at: '2026-06-27T16:58:25+00:00'
polished_at: '2026-06-27T19:22:35+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# ReduceLROnPlateau 学习率调度

_class_ torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer_, _mode='min'_, _factor=0.1_, _patience=10_, _threshold=0.0001_, _threshold\_mode='rel'_, _cooldown=0_, _min\_lr=0_, _eps=1e-08_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/lr_scheduler.py#L1583)[#](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau "Link to this definition") Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.

Parameters:
*   **optimizer** ([_Optimizer_](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer "torch.optim.Optimizer")) – Wrapped optimizer.
*   **mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
*   **factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
*   **patience** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The number of allowed epochs with no improvement after which the learning rate will be reduced. For example, consider the case of having no patience (patience = 0). In the first epoch, a baseline is established and is always considered good as there’s no previous baseline. In the second epoch, if the performance is worse than the baseline, we have what is considered an intolerable epoch. Since the count of intolerable epochs (1) is greater than the patience level (0), the learning rate is reduced at the end of this epoch. From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch if the performance is worse than the baseline. If the performance improves or remains the same, the learning rate is not adjusted. Default: 10.
*   **threshold** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
*   **threshold_mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
*   **cooldown** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
*   **min_lr** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")_or_[_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.

Example
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, "min")
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>> # Note that step should be called after validate()
>>>     scheduler.step(val_loss)

Get the most recent learning rates computed by this scheduler.

Returns:
A [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of learning rates with entries for each of the optimizer’s `param_groups`, with the same types as their `group["lr"]`s.

Return type:
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")[[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") | [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")] Note The returned [`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")s are copies, and never alias the optimizer’s `group["lr"]`s.
get_lr()[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/lr_scheduler.py#L219)[#](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau.get_lr "Link to this definition") Compute the next learning rate for each of the optimizer’s `param_groups`.

Returns:
A [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of learning rates for each of the optimizer’s `param_groups` with the same types as their current `group["lr"]`s.

Return type:
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")[[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") | [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")] Note If you’re trying to inspect the most recent learning rate, use [`get_last_lr()`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau.get_last_lr "torch.optim.lr_scheduler.ReduceLROnPlateau.get_last_lr") instead.

Note The returned [`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")s are copies, and never alias the optimizer’s `group["lr"]`s.
load_state_dict(_state\_dict_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/lr_scheduler.py#L1771)[#](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau.load_state_dict "Link to this definition") Load the scheduler’s state.
state_dict()[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/lr_scheduler.py#L182)[#](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau.state_dict "Link to this definition") Return the state of the scheduler as a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)").

It contains an entry for every variable in `self.__dict__` which is not the optimizer.

Return type:
[dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), [_Any_](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")] step(_metrics_, _epoch=None_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/optim/lr_scheduler.py#L1688)[#](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau.step "Link to this definition") Perform a step.

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-学习率调度|Lightning Bolts 学习率调度]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-学习率调度|Lightning Bolts 学习率调度]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/CosineAnnealingLR-余弦退火|CosineAnnealingLR 余弦退火]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-工具箱|Lightning Bolts 工具箱]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Hooks-与-Callbacks|Lightning Hooks 与 Callbacks]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Hooks-与-Callbacks|Lightning Hooks 与 Callbacks]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/PyTorch-Lightning-完全攻略-zh|PyTorch Lightning 完全攻略]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/PyTorch-Lightning-完全攻略|PyTorch Lightning 完全攻略]]
