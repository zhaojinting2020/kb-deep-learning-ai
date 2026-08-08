---
title: PyTorch 命名张量
url: https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#:~:text=PyTorch%20%E4%B8%AD%E7%9A%84
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:46:10+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# PyTorch 命名张量

*   [PyTorch 中的命名张量简介（原型）](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#fzexwx "PyTorch 中的命名张量简介（原型）")
    *   [基础知识：命名维度](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#d7lh0l "基础知识：命名维度")         *   [访问器和归约](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#61bp1p "访问器和归约")         *   [名称推断](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#cxr5nd "名称推断")             *   [广播](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#1hetjv "广播")             *   [矩阵乘法](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#ffuuhq "矩阵乘法")         *   [新行为：按名称显式广播](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#duwwy8 "新行为：按名称显式广播")         *   [新行为：按名称展平或取消展平维度](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#cuvcqi "新行为：按名称展平或取消展平维度")         *   [Autograd 支持](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#v9akb "Autograd 支持")         *   [其他受支持的（和不受支持的）功能](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#wr1xp "其他受支持的（和不受支持的）功能")         *   [更长的例子：多头关注](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#1yhvjb "更长的例子：多头关注")         *   [运行示例](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#9lxxof "运行示例")         *   [总结](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md#dekun1 "总结")
## [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)PyTorch 中的命名张量简介（原型）
> 原文：[https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html](https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html)

**作者**： [Richard Zou](https://github.com/zou3519)
命名张量旨在通过允许用户将显式名称与张量维度相关联来使张量更易于使用。在大多数情况下，采用尺寸参数的操作将接受尺寸名称，而无需按位置跟踪尺寸。此外，命名张量使用名称来自动检查运行时是否正确使用了 API，从而提供了额外的安全性。名称也可以用于重新排列尺寸，例如，支持“按名称广播”而不是“按位置广播”。本教程旨在作为 1.3 启动中将包含的功能的指南。到最后，您将能够：
*   创建具有命名尺寸的张量，以及删除或重命名这些尺寸
*   了解操作如何传播维度名称的基础
*   了解命名尺寸如何在两个关键区域实现更清晰的代码：
    *   广播操作    *   重塑和展开尺寸最后，我们将通过使用命名张量编写一个多头注意力模块来将其付诸实践。

PyTorch 中的命名张量受 [Sasha Rush](https://tech.cornell.edu/people/alexander-rush/) 的启发并与之合作。 Sasha 在他的 [2019 年 1 月博客文章](http://nlp.seas.harvard.edu/NamedTensor)中提出了最初的想法和概念证明。

## [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)基础知识：命名维度
PyTorch 现在允许张量具有命名维度； 工厂函数采用新的名称参数，该参数将名称与每个维度相关联。这适用于大多数工厂函数，例如
*   `tensor`
*   `empty`
*   `ones`
*   `zeros`
*   `randn`
*   `rand`

这里我们用名字构造一个张量：
1.   `import torch`
2.   `imgs = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))`
3.   `print(imgs.names)`

出：
1.   `('N', 'C', 'H', 'W')`

与[命名张量的原始博客文章](http://nlp.seas.harvard.edu/NamedTensor)不同，命名维度是有序的：`tensor.names[i]`是`tensor`的第`i`个维度的名称。重命名`Tensor`尺寸的方法有两种：
1.   `# Method #1: set the .names attribute (this changes name in-place)`
2.   `imgs.names = ['batch', 'channel', 'width', 'height']`
3.   `print(imgs.names)`
5.   `# Method #2: specify new names (this changes names out-of-place)`
6.   `imgs = imgs.rename(channel='C', width='W', height='H')`
7.   `print(imgs.names)`

出：
1.   `('batch', 'channel', 'width', 'height')`
2.   `('batch', 'C', 'W', 'H')`

删除名称的首选方法是调用`tensor.rename(None)`：
1.   `imgs = imgs.rename(None)`
2.   `print(imgs.names)`

出：
1.   `(None, None, None, None)`

未命名的张量（没有命名尺寸的张量）仍然可以正常工作，并且在其`repr`中没有名称。
1.   `unnamed = torch.randn(2, 1, 3)`
2.   `print(unnamed)`
3.   `print(unnamed.names)`

出：
1.   `tensor(`[[[-0.7420, -0.3646,  0.1424]]`,`
3.   ``[[-0.6065, -1.4888,  0.2935]]`])`
4.   `(None, None, None)`

命名张量不需要命名所有尺寸。
1.   `imgs = torch.randn(3, 1, 1, 2, names=('N', None, None, None))`
2.   `print(imgs.names)`

出：
1.   `('N', None, None, None)`

由于命名张量可以与未命名张量共存，因此我们需要一种不错的方式来编写可识别命名张量的代码，该代码可用于命名张量和未命名张量。使用`tensor.refine_names(*names)`优化尺寸并将未命名的暗淡提升为已命名的暗淡。细化维度定义为“重命名”，并具有以下限制：
*   可以将`None`暗号细化为任何名称
*   命名的维度只能精简为具有相同的名称。
1.   `imgs = torch.randn(3, 1, 1, 2)`
2.   `named_imgs = imgs.refine_names('N', 'C', 'H', 'W')`
3.   `print(named_imgs.names)`
5.   `# Refine the last two dims to 'H' and 'W'. In Python 2, use the string '...'`
6.   `# instead of ...`
7.   `named_imgs = imgs.refine_names(..., 'H', 'W')`
8.   `print(named_imgs.names)`
10.   `def catch_error(fn):`
11.   `try:`
12.   `fn()`
13.   `assert False`
14.   `except RuntimeError as err:`
15.   `err = str(err)`
16.   `if len(err) > 180:`
17.   `err = err[:180] + "..."`
18.   `print(err)`
20.   `named_imgs = imgs.refine_names('N', 'C', 'H', 'W')`
22.   `# Tried to refine an existing name to a different name`
23.   `catch_error(lambda: named_imgs.refine_names('N', 'C', 'H', 'width'))`

出：
1.   `('N', 'C', 'H', 'W')`
2.   `(None, None, 'H', 'W')`
3.   `refine_names: cannot coerce Tensor['N', 'C', 'H', 'W'] to Tensor['N', 'C', 'H', 'width'] because 'W' is different from 'width' at index 3`

大多数简单的操作都会传播名称。命名张量的最终目标是所有操作以合理，直观的方式传播名称。在 1.3 版本发布时，已添加了对许多常用操作的支持。例如，这里是`.abs()`：
1.   `print(named_imgs.abs().names)`

出：
1.   `('N', 'C', 'H', 'W')`
### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)访问器和归约
可以使用尺寸名称来引用尺寸而不是位置尺寸。这些操作还传播名称。索引（基本索引和高级索引）尚未实现，但仍在规划中。使用上面的`named_imgs`张量，我们可以执行以下操作：
1.   `output = named_imgs.sum('C')  # Perform a sum over the channel dimension`
2.   `print(output.names)`
4.   `img0 = named_imgs.select('N', 0)  # get one image`
5.   `print(img0.names)`

出：
1.   `('N', 'H', 'W')`
2.   `('C', 'H', 'W')`
### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)名称推断
名称在称为**名称推断**的两步过程中在操作上传播：
1.   **检查名称**：运算符可以在运行时执行自动检查，以检查某些尺寸名称是否匹配。
2.   **传播名称**：名称推断将输出名称传播到输出张量。让我们看一个非常小的例子，添加 2 个一维张量，不进行广播。
1.   `x = torch.randn(3, names=('X',))`
2.   `y = torch.randn(3)`
3.   `z = torch.randn(3, names=('Z',))`

**检查名称**：首先，我们将检查这两个张量的名称是否相匹配。当且仅当两个名称相等（字符串相等）或至少一个为`None`（`None`本质上是一个特殊的通配符名称）时，两个名称才匹配。因此，这三者中唯一会出错的是`x + z`：
1.   `catch_error(lambda: x + z)`

出：
1.   `Error when attempting to broadcast dims ['X'] and dims ['Z']: dim 'X' and dim 'Z' are at the same position from the right but do not match.`

**传播名称**：通过返回两个名称中最精确的名称来统一这两个名称。使用`x + y`时，`X`比`None`更精细。
1.   `print((x + y).names)`

出：
1.   `('X',)`

大多数名称推断规则都很简单明了，但是其中一些可能具有意想不到的语义。让我们来看看您可能会遇到的一对：广播和矩阵乘法。

#### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)广播
命名张量不会改变广播行为； 他们仍然按位置广播。但是，在检查两个尺寸是否可以广播时，PyTorch 还会检查这些尺寸的名称是否匹配。这导致命名张量防止广播操作期间意外对齐。在下面的示例中，我们将`per_batch_scale`应用于`imgs`。
1.   `imgs = torch.randn(2, 2, 2, 2, names=('N', 'C', 'H', 'W'))`
2.   `per_batch_scale = torch.rand(2, names=('N',))`
3.   `catch_error(lambda: imgs * per_batch_scale)`

出：
1.   `Error when attempting to broadcast dims ['N', 'C', 'H', 'W'] and dims ['N']: dim 'W' and dim 'N' are at the same position from the right but do not match.`

如果没有`names`，则`per_batch_scale`张量与`imgs`的最后一个尺寸对齐，这不是我们想要的。我们确实想通过将`per_batch_scale`与`imgs`的批量尺寸对齐来执行操作。有关如何按名称对齐张量的信息，请参见新的“按名称显式广播”功能，如下所述。

#### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)矩阵乘法
`torch.mm(A, B)`在`A`的第二个暗角和`B`的第一个暗角之间执行点积，返回具有`A`的第一个暗角和`B`的第二个暗角的张量。 （其他`matmul`函数，例如`torch.matmul`，`torch.mv`和`torch.dot`的行为类似）。
1.   `markov_states = torch.randn(128, 5, names=('batch', 'D'))`
2.   `transition_matrix = torch.randn(5, 5, names=('in', 'out'))`
4.   `# Apply one transition`
5.   `new_state = markov_states @ transition_matrix`
6.   `print(new_state.names)`

出：
1.   `('batch', 'out')`

如您所见，矩阵乘法不会检查收缩尺寸是否具有相同的名称。接下来，我们将介绍命名张量启用的两个新行为：按名称的显式广播以及按名称的展平和展平尺寸

### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)新行为：按名称显式广播
有关使用多个维度的主要抱怨之一是需要`unsqueeze`“虚拟”维度，以便可以进行操作。例如，在之前的每批比例示例中，使用未命名的张量，我们将执行以下操作：
1.   `imgs = torch.randn(2, 2, 2, 2)  # N, C, H, W`
2.   `per_batch_scale = torch.rand(2)  # N`
4.   `correct_result = imgs * per_batch_scale.view(2, 1, 1, 1)  # N, C, H, W`
5.   `incorrect_result = imgs * per_batch_scale.expand_as(imgs)`
6.   `assert not torch.allclose(correct_result, incorrect_result)`

通过使用名称，我们可以使这些操作更安全（并且易于与尺寸数量无关）。我们提供了一个新的`tensor.align_as(other)`操作，可以对张量的尺寸进行排列以匹配`other.names`中指定的顺序，并在适当的地方添加一个尺寸的尺寸（`tensor.align_to(*names)`也可以）：
1.   `imgs = imgs.refine_names('N', 'C', 'H', 'W')`
2.   `per_batch_scale = per_batch_scale.refine_names('N')`
4.   `named_result = imgs * per_batch_scale.align_as(imgs)`
5.   `# note: named tensors do not yet work with allclose`
6.   `assert torch.allclose(named_result.rename(None), correct_result)`
### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)新行为：按名称展平或取消展平维度
一种常见的操作是展平和展平尺寸。现在，用户可以使用`view`，`reshape`或`flatten`来执行此操作； 用例包括将批量尺寸展平以将张量发送到必须采用一定数量尺寸的输入的运算符（即`conv2d`采用 4D 输入）。为了使这些操作比查看或整形更具语义意义，我们引入了一种新的`tensor.unflatten(dim, namedshape)`方法并更新`flatten`以使用名称：`tensor.flatten(dims, new_dim)`。
`flatten`只能展平相邻的尺寸，但也可以用于不连续的维度。必须将名称和形状传递到`unflatten`中，该形状是`(dim, size)`元组的列表，以指定如何展开维度。可以在`flatten`期间保存`unflatten`的尺寸，但我们尚未这样做。
1.   `imgs = imgs.flatten(['C', 'H', 'W'], 'features')`
2.   `print(imgs.names)`
4.   `imgs = imgs.unflatten('features', (('C', 2), ('H', 2), ('W', 2)))`
5.   `print(imgs.names)`

出：
1.   `('N', 'features')`
2.   `('N', 'C', 'H', 'W')`
### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)Autograd 支持
Autograd 当前会忽略所有张量上的名称，只是将它们视为常规张量。梯度计算是正确的，但是我们失去了名称赋予我们的安全性。在路线图上引入名称以自动微分的处理。
1.   `x = torch.randn(3, names=('D',))`
2.   `weight = torch.randn(3, names=('D',), requires_grad=True)`
3.   `loss = (x - weight).abs()`
4.   `grad_loss = torch.randn(3)`
5.   `loss.backward(grad_loss)`
7.   `correct_grad = weight.grad.clone()`
8.   `print(correct_grad)  # Unnamed for now. Will be named in the future`
10.   `weight.grad.zero_()`
11.   `grad_loss = grad_loss.refine_names('C')`
12.   `loss = (x - weight).abs()`
13.   `# Ideally we'd check that the names of loss and grad_loss match, but we don't`
14.   `# yet`
15.   `loss.backward(grad_loss)`
17.   `print(weight.grad)  # still unnamed`
18.   `assert torch.allclose(weight.grad, correct_grad)`

出：
1.   `tensor([0.5398, 0.7907, 0.7784])`
2.   `tensor([0.5398, 0.7907, 0.7784])`
### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)其他受支持的（和不受支持的）功能
[有关 1.3 发行版支持的功能的详细分类，请参见此处](https://pytorch.org/docs/stable/named_tensor.html)。特别是，我们要指出当前不支持的三个重要函数：
*   通过`torch.save`或`torch.load`保存或加载命名张量
*   通过`torch.multiprocessing`进行多重处理
*   JIT 支持； 例如，以下将错误
1.   `imgs_named = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))`
3.   `@torch.jit.script`
4.   `def fn(x):`
5.   `return x`
7.   `catch_error(lambda: fn(imgs_named))`

出：
1.   `NYI: Named tensors are currently unsupported in TorchScript. As a  workaround please drop names via `tensor = tensor.rename(None)`.`

解决方法是，在使用尚不支持命名张量的任何东西之前，请通过`tensor = tensor.rename(None)`删除名称。

### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)更长的例子：多头关注
现在，我们将通过一个完整的示例来实现一个常见的 PyTorch `nn.Module`：多头注意。我们假设读者已经熟悉多头注意； 要进行复习，请查看[此说明](https://nlp.seas.harvard.edu/2018/04/03/attention.html)或[此说明](http://jalammar.github.io/illustrated-transformer/)。我们采用 [ParlAI](https://github.com/facebookresearch/ParlAI) 来实现多头注意力的实现； 具体来说[此处](https://github.com/facebookresearch/ParlAI/blob/f7db35cba3f3faf6097b3e6b208442cd564783d9/parlai/agents/transformer/modules.py#L907)。阅读该示例中的代码； 然后，与下面的代码进行比较，注意有四个标记为（I），（II），（III）和（IV）的位置，使用命名张量可以使代码更易读； 在代码块之后，我们将深入探讨其中的每一个。
1.   `import torch.nn as nn`
2.   `import torch.nn.functional as F`
3.   `import math`
5.   `class MultiHeadAttention(nn.Module):`
6.   `def __init__(self, n_heads, dim, dropout=0):`
7.   `super(MultiHeadAttention, self).__init__()`
8.   `self.n_heads = n_heads`
9.   `self.dim = dim`
11.   `self.attn_dropout = nn.Dropout(p=dropout)`
12.   `self.q_lin = nn.Linear(dim, dim)`
13.   `self.k_lin = nn.Linear(dim, dim)`
14.   `self.v_lin = nn.Linear(dim, dim)`
15.   `nn.init.xavier_normal_(self.q_lin.weight)`
16.   `nn.init.xavier_normal_(self.k_lin.weight)`
17.   `nn.init.xavier_normal_(self.v_lin.weight)`
18.   `self.out_lin = nn.Linear(dim, dim)`
19.   `nn.init.xavier_normal_(self.out_lin.weight)`
21.   `def forward(self, query, key=None, value=None, mask=None):`
22.   `# (I)`
23.   `query = query.refine_names(..., 'T', 'D')`
24.   `self_attn = key is None and value is None`
25.   `if self_attn:`
26.   `mask = mask.refine_names(..., 'T')`
27.   `else:`
28.   `mask = mask.refine_names(..., 'T', 'T_key')  # enc attn`
30.   `dim = query.size('D')`
31.   `assert dim == self.dim, \`
32.   `f'Dimensions do not match: {dim} query vs {self.dim} configured'`
33.   `assert mask is not None, 'Mask is None, please specify a mask'`
34.   `n_heads = self.n_heads`
35.   `dim_per_head = dim // n_heads`
36.   `scale = math.sqrt(dim_per_head)`
38.   `# (II)`
39.   `def prepare_head(tensor):`
40.   `tensor = tensor.refine_names(..., 'T', 'D')`
41.   `return (tensor.unflatten('D', [('H', n_heads), ('D_head', dim_per_head)])`
42.   `.align_to(..., 'H', 'T', 'D_head'))`
44.   `assert value is None`
45.   `if self_attn:`
46.   `key = value = query`
47.   `elif value is None:`
48.   `# key and value are the same, but query differs`
49.   `key = key.refine_names(..., 'T', 'D')`
50.   `value = key`
51.   `dim = key.size('D')`
53.   `# Distinguish between query_len (T) and key_len (T_key) dims.`
54.   `k = prepare_head(self.k_lin(key)).rename(T='T_key')`
55.   `v = prepare_head(self.v_lin(value)).rename(T='T_key')`
56.   `q = prepare_head(self.q_lin(query))`
58.   `dot_prod = q.div_(scale).matmul(k.align_to(..., 'D_head', 'T_key'))`
59.   `dot_prod.refine_names(..., 'H', 'T', 'T_key')  # just a check`
61.   `# (III)`
62.   `attn_mask = (mask == 0).align_as(dot_prod)`
63.   `dot_prod.masked_fill_(attn_mask, -float(1e20))`
65.   `attn_weights = self.attn_dropout(F.softmax(dot_prod / scale,`
66.   `dim='T_key'))`
68.   `# (IV)`
69.   `attentioned = (`
70.   `attn_weights.matmul(v).refine_names(..., 'H', 'T', 'D_head')`
71.   `.align_to(..., 'T', 'H', 'D_head')`
72.   `.flatten(['H', 'D_head'], 'D')`
73.   `)`
75.   `return self.out_lin(attentioned).refine_names(..., 'T', 'D')`

（I）细化输入张量维度
1.   `def forward(self, query, key=None, value=None, mask=None):`
2.   `# (I)`
3.   `query = query.refine_names(..., 'T', 'D')`

`query = query.refine_names(..., 'T', 'D')`用作可执行的文档，并将输入尺寸提升为名称。它检查最后两个维度是否可以调整为`['T', 'D']`，以防止在以后出现潜在的无声或混乱的尺寸不匹配错误。
（II）在`prepare_head`中操纵尺寸
1.   `# (II)`
2.   `def prepare_head(tensor):`
3.   `tensor = tensor.refine_names(..., 'T', 'D')`
4.   `return (tensor.unflatten('D', [('H', n_heads), ('D_head', dim_per_head)])`
5.   `.align_to(..., 'H', 'T', 'D_head'))`

首先要注意的是代码如何清楚地说明输入和输出尺寸：输入张量必须以`T`和`D`变暗结束，输出张量应以`H`，`T`和`D_head`维度结束。要注意的第二件事是代码清楚地描述了正在发生的事情。 `prepare_head`获取键，查询和值，并将嵌入的维度拆分为多个头部，最后将维度顺序重新排列为`[..., 'H', 'T', 'D_head']`。 ParlAI 使用`view`和`transpose`操作实现以下`prepare_head`：
1.   `def prepare_head(tensor):`
2.   `# input is [batch_size, seq_len, n_heads * dim_per_head]`
3.   `# output is [batch_size * n_heads, seq_len, dim_per_head]`
4.   `batch_size, seq_len, _ = tensor.size()`
5.   `tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)`
6.   `tensor = (`
7.   `tensor.transpose(1, 2)`
8.   `.contiguous()`
9.   `.view(batch_size * n_heads, seq_len, dim_per_head)`
10.   `)`
11.   `return tensor`

我们命名的张量变量使用的操作虽然较为冗长，但比`view`和`transpose`具有更多的语义含义，并包含以名称形式出现的可执行文档。
（III）按名称显式广播
1.   `def ignore():`
2.   `# (III)`
3.   `attn_mask = (mask == 0).align_as(dot_prod)`
4.   `dot_prod.masked_fill_(attn_mask, -float(1e20))`

`mask`通常具有暗淡`[N, T]`（在自我关注的情况下）或`[N, T, T_key]`（对于编码器注意的情况），而`dot_prod`具有暗淡的`[N, H, T, T_key]`。为了使`mask`与`dot_prod`正确广播，我们通常会在自注意的情况下将的调暗`1`和`-1`压下，在编码器的情况下，我们将`unsqueeze`调暗`unsqueeze` 。使用命名张量，我们只需使用`align_as`将`attn_mask`与`dot_prod`对齐，而不必担心`unsqueeze`变暗的位置。
（IV）使用`align_to`和`flatten`进行更多尺寸操作
1.   `def ignore():`
2.   `# (IV)`
3.   `attentioned = (`
4.   `attn_weights.matmul(v).refine_names(..., 'H', 'T', 'D_head')`
5.   `.align_to(..., 'T', 'H', 'D_head')`
6.   `.flatten(['H', 'D_head'], 'D')`
7.   `)`

在这里，与（II）一样，`align_to`和`flatten`在语义上比`view`和`transpose`更有意义（尽管更冗长）。

### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)运行示例
1.   `n, t, d, h = 7, 5, 2 * 3, 3`
2.   `query = torch.randn(n, t, d, names=('N', 'T', 'D'))`
3.   `mask = torch.ones(n, t, names=('N', 'T'))`
4.   `attn = MultiHeadAttention(h, d)`
5.   `output = attn(query, mask=mask)`
6.   `# works as expected!`
7.   `print(output.names)`

出：
1.   `('N', 'T', 'D')`

以上工作正常。此外，请注意，在代码中我们根本没有提到批量维度的名称。实际上，我们的`MultiHeadAttention`模块与批量尺寸的存在无关。
1.   `query = torch.randn(t, d, names=('T', 'D'))`
2.   `mask = torch.ones(t, names=('T',))`
3.   `output = attn(query, mask=mask)`
4.   `print(output.names)`

出：
1.   `('T', 'D')`
### [](https://geekdaxue.co/read/Pytorch-document-turtorial/docs-1.7-42.md)总结
感谢您的阅读！ 命名张量仍在发展中。如果您有反馈和/或改进建议，请通过创建 [ISSUE](https://github.com/pytorch/pytorch/issues) 来通知我们。
**脚本的总运行时间**：（0 分钟 0.094 秒）
[下载 Python 源码：`named_tensor_tutorial.py`](https://geekdaxue.co/read/Pytorch-document-turtorial/$docs-_downloads-1e94d0ce96a0c8097f002bcbe94c35d7-named_tensor_tutorial.py) [下载 Jupyter 笔记本：`named_tensor_tutorial.ipynb`](https://geekdaxue.co/read/Pytorch-document-turtorial/$docs-_downloads-90d6df7aa4b65bb035e19943c6f92ea0-named_tensor_tutorial.ipynb) [由 Sphinx 画廊](https://sphinx-gallery.readthedocs.io/)生成的画廊

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-einsum-爱因斯坦求和|PyTorch einsum 爱因斯坦求和]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制-qq|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-显示为图片|Tensor 显示为图片]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-标准化|Tensor 标准化]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/TensorFlow-2-与-Keras-入门|TensorFlow 2 与 Keras 入门]]
