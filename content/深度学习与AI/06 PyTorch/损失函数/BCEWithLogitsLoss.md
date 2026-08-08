---
title: BCEWithLogitsLoss
url: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
fetch_source: agent_reach:agent_reach:jina
fetched_at: '2026-06-27T16:58:44+00:00'
polished_at: '2026-06-27T19:22:35+00:00'
math_repaired_at: '2026-06-27T20:23:29+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# BCEWithLogitsLoss

_class_ torch.nn.BCEWithLogitsLoss(_weight=None_, _size\_average=None_, _reduce=None_, _reduction='mean'_, _pos\_weight=None_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/loss.py#L715)[#](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss "Link to this definition") This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:
$$ℓ \left( x , y \right) = L = \left\{l_{1} , \ldots , l_{N} \right\}^{\top} , l_{n} = - w_{n} \left[ y_{n} \cdot log  \sigma \left( x_{n} \right) + \left( 1 - y_{n} \right) \cdot log  \left( 1 - \sigma \left( x_{n} \right) \right) \right] , $$

ℓ \left( x , y \right) = \left\{mean  \left( L \right) , & \text{if}\textrm{ }\text{reduction} = ‘\text{mean}’; \\ sum  \left( L \right) , & \text{if}\textrm{ }\text{reduction} = ‘\text{sum}’.
$$ℓ(x,y)={mean(L),sum(L),​if reduction=‘mean’;if reduction=‘sum’.​This is used for measuring the error of a reconstruction in for example an auto-encoder. Note that the targets t[i] should be numbers between 0 and 1.
It’s possible to trade off recall and precision by adding weights to positive examples. In the case of multi-label classification the loss can be described as:
$$ℓ_{c} \left( x , y \right) = L_{c} = \left\{l_{1 , c} , \ldots , l_{N , c} \right\}^{\top} , l_{n , c} = - w_{n , c} \left[ p_{c} y_{n , c} \cdot log  \sigma \left( x_{n , c} \right) + \left( 1 - y_{n , c} \right) \cdot log  \left( 1 - \sigma \left( x_{n , c} \right) \right) \right] , $$

ℓ c​(x,y)=L c​={l 1,c​,…,l N,c​}⊤,l n,c​=−w n,c​[p c​y n,c​⋅lo g σ(x n,c​)+(1−y n,c​)⋅lo g(1−σ(x n,c​))], where $c$ is the class number ($c > 1$c>1 for multi-label binary classification, $c = 1$c=1 for single-label binary classification), $n$ is the number of the sample in the batch and $p_{c}$ p c​ is the weight of the positive answer for the class $c$.

For example, if a dataset contains 100 positive and 300 negative examples of a single class, then `pos_weight` for the class should be equal to $\frac{300}{100} = 3$100 300​=3. The loss would act as if the dataset contains $3 \times 100 = 300$3×100=300 positive examples.

Examples
>>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
>>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
>>> pos_weight = torch.ones([64])  # All weights are equal to 1
>>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
>>> criterion(output, target)  # -log(sigmoid(1.5))

tensor(0.20...) In the above example, the `pos_weight` tensor’s elements correspond to the 64 distinct classes in a multi-label binary classification scenario. Each element in `pos_weight` is designed to adjust the loss function based on the imbalance between negative and positive samples for the respective class. This approach is useful in datasets with varying levels of class imbalance, ensuring that the loss calculation accurately accounts for the distribution in each class.

Parameters:
*   **weight** ([_Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – a manual rescaling weight given to the loss of each batch element. The dimension of weight supports [broadcasting to a common shape](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics) with respect to the output (and target) shape.
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")_,_ _optional_) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when `reduce` is `False`. Default: `True`
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")_,_ _optional_) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`
*   **reduction** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")_,_ _optional_) – Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`
*   **pos_weight** ([_Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – a weight of positive examples to be broadcasted with target. Must be a tensor with equal size along the class dimension to the number of classes. Pay close attention to PyTorch’s broadcasting semantics in order to achieve the desired operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of size [B, C, H, W] will apply different pos_weights to each element of the batch or [C, H, W] the same pos_weights across the batch. To apply the same positive weight along all spatial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1]. Default: `None`

Shape:
*   Input: $\left( * \right)$(∗), where $*$∗ means any number of dimensions.
*   Target: $\left( * \right)$(∗), same shape as the input.
*   Output: scalar. If `reduction` is `'none'`, then $\left( * \right)$(∗), same shape as input.

Examples
>>> loss = nn.BCEWithLogitsLoss()
>>> input = torch.randn(3, requires_grad=True)
>>> target = torch.empty(3).random_(2)
>>> output = loss(input, target)
>>> output.backward()

forward(_input_, _target_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/loss.py#L832)[#](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss.forward "Link to this definition") Runs the forward pass.

Return type:

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[BCELoss-二元交叉熵|BCELoss 二元交叉熵]]
[[NLLLoss-与-CrossEntropyLoss|NLLLoss 与 CrossEntropyLoss]]
