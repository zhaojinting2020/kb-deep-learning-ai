---
title: Tensor 显示为图片
url: https://zhuanlan.zhihu.com/p/615144176
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T20:05:22+00:00'
polished_at: '2026-06-27T20:26:50+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Tensor 显示为图片

![Image 1: Pytorch 显示 Tensor 格式的图片](https://picx.zhimg.com/v2-f999c78836a9a48ce5ecbfc504dfcc68_720w.jpg?source=172ae18b)

<p class="kb-image-caption">图例</p>

[![Image 2: 月落天白](https://picx.zhimg.com/v2-d16e6f8257405327c9eadac5abc87d4f_l.jpg?source=32738c0c&needBackground=1)](https://www.zhihu.com/people/yue-luo-tian-bai-8)

<p class="kb-image-caption">图例</p>

[月落天白](https://www.zhihu.com/people/yue-luo-tian-bai-8) 6 人赞同了该文章使用 `transforms.ToPILImage()` 将 `Tensor` 格式的图片转为 `PIL` 格式，然后 `show()`

# 函数接受 CHW 格式的 Tensor 或 HWC 格式的 ndarray
# 对于 3 通道的输入会自动选择 RGB 模式, 其他特殊模式需要指定
# 4 通道自动 RGBA，2 通道自动 LA，1 通道根据数据类型自动选择(int, float等)
trans_to_pil = transforms.ToPILImage(mode="RGB") img_pil = trans_to_pil(img_tensor) img_pil.show()
参考：
[ToPILImage — Torchvision 0.15 documentation (pytorch.org)](https://link.zhihu.com/?target=https%3A//pytorch.org/vision/stable/generated/torchvision.transforms.ToPILImage.html%23torchvision.transforms.ToPILImage) [Concepts - Pillow (PIL Fork) 9.5.0.dev0 documentation](https://link.zhihu.com/?target=https%3A//pillow.readthedocs.io/en/latest/handbook/concepts.html%23modes)

## 相关笔记
[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-标准化|Tensor 标准化]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-einsum-爱因斯坦求和|PyTorch einsum 爱因斯坦求和]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制-qq|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-命名张量|PyTorch 命名张量]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/TensorFlow-2-与-Keras-入门|TensorFlow 2 与 Keras 入门]]
