---
title: Conv2d 二维卷积
url: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
fetch_source: agent_reach:agent_reach:jina
fetched_at: '2026-06-27T16:58:33+00:00'
polished_at: '2026-06-27T19:22:35+00:00'
math_repaired_at: '2026-06-27T20:23:29+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Conv2d 二维卷积

_class_ torch.nn.Conv2d(_in\_channels_, _out\_channels_, _kernel\_size_, _stride=1_, _padding=0_, _dilation=1_, _groups=1_, _bias=True_, _padding\_mode='zeros'_, _device=None_, _dtype=None_)[source](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/conv.py#L388)[#](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d "Link to this definition") Applies a 2D convolution over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size $\left( N , C_{\text{in}} , H , W \right)$(N,C in​,H,W) and output $\left( N , C_{\text{out}} , H_{\text{out}} , W_{\text{out}} \right)$(N,C out​,H out​,W out​) can be precisely described as:
$$ \text{out} \left( N_{i} , C_{\text{out}_{j}} \right) = \text{bias} \left( C_{\text{out}_{j}} \right) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight} \left( C_{\text{out}_{j}} , k \right) \star \text{input} \left( N_{i} , k \right) $$ out(N i​,C out j​​)=bias(C out j​​)+k=0∑C in​−1​weight(C out j​​,k)⋆input(N i​,k) where $\star$⋆ is the valid 2D [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) operator, $N$ is a batch size, $C_{\text{in}}$ C in​ and $C_{\text{out}}$ C out​ correspond to `in_channels` and `out_channels` respectively, $H$ and $W$ are the input height and width in pixels. See the Shape section below for how $H_{\text{out}}$ H out​ and $W_{\text{out}}$ W out​ are derived from `kernel_size`, `stride`, `padding`, and `dilation`.
This module supports [TensorFloat32](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).

On certain ROCm devices, when using float16 inputs this module will use [different precision](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#fp16-on-mi200) for backward.
*   `stride` controls the stride for the cross-correlation, a single number or a tuple.
*   `padding` controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or an int / a tuple of ints giving the amount of implicit padding applied on both sides.
*   `dilation` controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation` does.
*   `groups` controls the connections between inputs and outputs. `in_channels` and `out_channels` must both be divisible by `groups`. For example,

> *   At groups=1, all inputs are convolved to all outputs.
>
>     *   At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
>
>     *   At groups= `in_channels`, each input channel is convolved with its own set of filters (of size $\frac{\text{out}_\text{channels}}{\text{in}_\text{channels}}$in_channels out_channels​).

The parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:
> *   a single `int` – in which case the same value is used for the height and width dimension
>
> *   a `tuple` of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension

Note When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”.

In other words, for an input of size $\left( N , C_{i n} , L_{i n} \right)$(N,C in​,L in​), a depthwise convolution with a depthwise multiplier K can be performed with the arguments $\left( C_{\text{in}} = C_{\text{in}} , C_{\text{out}} = C_{\text{in}} \times \text{K} , . . . , \text{groups} = C_{\text{in}} \right)$(C in​=C in​,C out​=C in​×K,...,groups=C in​).

Note In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting `torch.backends.cudnn.deterministic = True`. See [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) for more information.

Note `padding='valid'` is the same as no padding. `padding='same'` pads the input so the output has the shape as the input. However, this mode doesn’t support any stride values other than 1.

Note This module supports complex data types i.e. `complex32, complex64, complex128`.

Parameters:
*   **in_channels** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Number of channels in the input image
*   **out_channels** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Number of channels produced by the convolution
*   **kernel_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")_or_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")) – Size of the convolving kernel
*   **stride** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")_or_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")_,_ _optional_) – Stride of the convolution. Default: 1
*   **padding** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")_,_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")_or_[_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")_,_ _optional_) – Padding added to all four sides of the input. Default: 0
*   **dilation** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")_or_[_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")_,_ _optional_) – Spacing between kernel elements. Default: 1
*   **groups** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")_,_ _optional_) – Number of blocked connections from input channels to output channels. Default: 1
*   **bias** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")_,_ _optional_) – If `True`, adds a learnable bias to the output. Default: `True`
*   **padding_mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")_,_ _optional_) – `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'zeros'`

Shape:
$$ H_{o u t} = \lfloor \frac{H_{i n} + 2 \times \text{padding} \left[ 0 \right] - \text{dilation} \left[ 0 \right] \times \left( \text{kernel}_\text{size} \left[ 0 \right] - 1 \right) - 1}{\text{stride} \left[ 0 \right]} + 1 \rfloor $$ H o u t​=⌊stride[0]H in​+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​+1⌋ $$ W_{o u t} = \lfloor \frac{W_{i n} + 2 \times \text{padding} \left[ 1 \right] - \text{dilation} \left[ 1 \right] \times \left( \text{kernel}_\text{size} \left[ 1 \right] - 1 \right) - 1}{\text{stride} \left[ 1 \right]} + 1 \rfloor $$ W o u t​=⌊stride[1]W in​+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1​+1⌋Variables:
*   **weight** ([_Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – the learnable weights of the module of shape $\left( \text{out}_\text{channels} , \frac{\text{in}_\text{channels}}{\text{groups}} ,$(out_channels,groups in_channels​,$\text{kernel}_\text{size}[\text{0}] , \text{kernel}_\text{size}[\text{1}] \right)$kernel_size[0],kernel_size[1]). The values of these weights are sampled from $\mathcal{U} \left( - \sqrt{k} , \sqrt{k} \right)$U(−k​,k​) where $k = \frac{g r o u p s}{C_{\text{in}} * \prod_{i = 0}^{1} \text{kernel}_\text{size} \left[ i \right]}$k=C in​∗∏i=0 1​kernel_size[i]g ro u p s​
*   **bias** ([_Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – the learnable bias of the module of shape (out_channels). If `bias` is `True`, then the values of these weights are sampled from $\mathcal{U} \left( - \sqrt{k} , \sqrt{k} \right)$U(−k​,k​) where $k = \frac{g r o u p s}{C_{\text{in}} * \prod_{i = 0}^{1} \text{kernel}_\text{size} \left[ i \right]}$k=C in​∗∏i=0 1​kernel_size[i]g ro u p s​

Examples
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/卷积与视觉模型/CycleGAN-训练流程|CycleGAN 训练流程]]
[[content/深度学习与AI/06 PyTorch/卷积与视觉模型/MobileNet-五类花-Colab|MobileNet 五类花 Colab]]
