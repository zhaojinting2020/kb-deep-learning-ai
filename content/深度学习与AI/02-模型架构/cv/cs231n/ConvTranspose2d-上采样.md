---
title: ConvTranspose2d 上采样
url: https://blog.csdn.net/qq_27261889/article/details/86304061
fetch_source: csdn:content_views+follow
fetched_at: '2026-06-28T09:45:27+00:00'
curated_at: '2026-06-28T10:00:00+00:00'
---

# ConvTranspose2d 上采样

生成模型需要逐步**放大特征图尺寸**；深度学习中常用 **`nn.ConvTranspose2d`** 实现上采样，DCGAN（[论文](https://arxiv.org/abs/1511.06434)）即采用此法。

## 名称对照

| 场景 | 名称 |
|------|------|
| PyTorch API | `ConvTranspose2d`（[文档](https://pytorch.org/docs/stable/nn.html?highlight=convtranspose#torch.nn.ConvTranspose2d)） |
| 论文 | fractionally-strided convolutions |
| 不推荐 | deconvolution（并非数学意义上的"去卷积"） |

下文用 **逆卷积** 指代"给定卷积结果，反求输入特征图"这一操作。

## 逆卷积是什么

**正卷积**：图像 A（Height × Width × Channels）经 kernel, stride, padding 得到 B——用一次卷积把 A 变成 B.

**逆卷积**：已知 B 与相同卷积核设置，求特征图 y，使得"y 经该卷积得到 x（原 A）"。卷积核尺寸与正卷积相同，但目标是恢复空间分辨率。

### 数值例子

- 输入 x：4×4×`channels_in`
- 卷积核：3×3，stride=1，padding=0
- 输出 y：2×2×`channels_out`

逆卷积的输入是 y，在相同核设置下恢复 4×4 的 x（见原文配图：卷积与逆卷积的像素连接关系）。

## 代码中的三步实现

给定特征图 **a** 与卷积超参，逆卷积可拆为：

1. 把 **a** 变换为 **a′**（主要在空间维插零扩尺寸）
2. 得到新的卷积超参（加撇号区分）
3. 在 **a′** 上做**普通卷积**，结果即为逆卷积输出

### 符号与公式

设原特征图尺寸 Height, Width；卷积核 Size，步长 Stride，padding.

**新特征图尺寸**（在元素之间插入零之前的目标大小）：

$$
\text{Height}' = \text{Height} + (\text{Stride} - 1) \times (\text{Height} - 1)
$$

$$
\text{Width}' = \text{Width} + (\text{Stride} - 1) \times (\text{Width} - 1)
$$

**插零**：在相邻元素之间插入 `(Stride − 1)` 个 0（fractionally-strided / sub-pixel 的含义）。

**新卷积超参**（与正卷积对称，使 shape 互逆）：

$$
\text{padding}' = \text{Size} - 1 - \text{padding}, \quad \text{Stride}' = 1, \quad \text{Size}' = \text{Size}
$$

然后在 a′ 上以 `(Size', Stride', padding')` 做标准卷积。

> 原文后续"逆卷积与卷积的关系", "参数详解"为 CSDN 付费章节；PyTorch 官方输出尺寸公式：
>
> $$H_{out} = (H_{in}-1)\times stride - 2\times padding + dilation\times(k-1) + output\_padding + 1$$
>
> （Width 同理；`output_padding` 用于 stride>1 时的 shape 歧义。）

## 与上采样的关系

逆卷积 / 转置卷积是一种**可学习的上采样**：区别于最近邻, 双线性插值等固定插值，核权重由网络学习。GAN 解码器, 语义分割 head 中广泛使用。
