---
title: BCELoss 二元交叉熵
url: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
fetch_source: agent_reach:agent_reach:jina
fetched_at: '2026-06-27T16:58:42+00:00'
polished_at: '2026-06-27T19:22:35+00:00'
math_repaired_at: '2026-06-27T19:29:42+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# BCELoss 二元交叉熵

_class_ torch.nn.BCELoss(_weight=None_, _size\_average=None_, _reduce=None_, _reduction='mean'_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/loss.py#L629)[#](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss "Link to this definition") Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:
$$ℓ \left( x , y \right) = L = \left\{l_{1} , \ldots , l_{N} \right\}^{\top} , l_{n} = - w_{n} \left[ y_{n} \cdot log  x_{n} + \left( 1 - y_{n} \right) \cdot log  \left( 1 - x_{n} \right) \right] , $$

ℓ \left( x , y \right) = \left\{mean  \left( L \right) , & \text{if}\textrm{ }\text{reduction} = ‘\text{mean}’; \\ sum  \left( L \right) , & \text{if}\textrm{ }\text{reduction} = ‘\text{sum}’.
$$ℓ(x,y)={mean(L),sum(L),​if reduction=‘mean’;if reduction=‘sum’.​This is used for measuring the error of a reconstruction in for example an auto-encoder. Note that the targets $y$ should be numbers between 0 and 1.
Notice that if $x_{n}$ x n​ is either 0 or 1, one of the log terms would be mathematically undefined in the above loss equation. PyTorch chooses to set $log  \left( 0 \right) = - \infty$ lo g(0)=−∞, since $\left(lim \right)_{x \rightarrow 0} log  \left( x \right) = - \infty$ lim x→0​lo g(x)=−∞. However, an infinite term in the loss equation is not desirable for several reasons.

For one, if either $y_{n} = 0$ y n​=0 or $\left( 1 - y_{n} \right) = 0$(1−y n​)=0, then we would be multiplying 0 with infinity. Secondly, if we have an infinite loss value, then we would also have an infinite term in our gradient, since $\left(lim \right)_{x \rightarrow 0} \frac{d}{d x} log  \left( x \right) = \infty$ lim x→0​d x d​lo g(x)=∞. This would make BCELoss’s backward method nonlinear with respect to $x_{n}$ x n​, and using it for things like linear regression would not be straight-forward.

Our solution is that BCELoss clamps its log function outputs to be greater than or equal to -100. This way, we can always have a finite loss value and a linear backward method.

Parameters:
*   **weight** ([_Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")_,_ _optional_) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when `reduce` is `False`. Default: `True`
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")_,_ _optional_) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`
*   **reduction** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")_,_ _optional_) – Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:
*   Input: $\left( * \right)$(∗), where $*$∗ means any number of dimensions.
*   Target: $\left( * \right)$(∗), same shape as the input.
*   Output: scalar. If `reduction` is `'none'`, then $\left( * \right)$(∗), same shape as input.

Examples
>>> m = nn.Sigmoid()
>>> loss = nn.BCELoss()
>>> input = torch.randn(3, 2, requires_grad=True)
>>> target = torch.rand(3, 2, requires_grad=False)
>>> output = loss(m(input), target)
>>> output.backward()

forward(_input_, _target_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/loss.py#L706)[#](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss.forward "Link to this definition") Runs the forward pass.

Return type:

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/损失函数/BCEWithLogitsLoss|BCEWithLogitsLoss]]
[[content/深度学习与AI/06 PyTorch/损失函数/NLLLoss-与-CrossEntropyLoss|NLLLoss 与 CrossEntropyLoss]]
