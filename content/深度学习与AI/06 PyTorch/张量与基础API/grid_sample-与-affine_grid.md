---
title: grid_sample 与 affine_grid
url: https://blog.csdn.net/qq_34914551/article/details/110186855
curated_at: '2026-06-28T20:00:00+00:00'
---

# grid_sample 与 affine_grid

PyTorch 的 `affine_grid` 与 `grid_sample` 常成对使用，实现可微分的 2D 仿射变换，类似 OpenCV 的 `getRotationMatrix2D` + `warpAffine`。

## 背景

- `affine_grid` 可用于 ROI Pooling, STN（Spatial Transformer Network）等场景
- 核心作用：根据仿射参数生成采样网格，再交给 `grid_sample` 从源图像双线性采样
- 仿射变换原理可参考：[仿射变换的基本原理](https://blog.csdn.net/qq_34914551/article/details/107132145)
- `grid_sample` 采样细节可参考：[grid sample（CSDN）](https://blog.csdn.net/qq_34914551/article/details/107559031)

## affine_grid

```python
def affine_grid(theta, size, align_corners=None):
    """
    theta: N×2×3 张量，N 为 batch size
    size:  输出网格/图像尺寸 (N, C, H, W)
    """
```

平面仿射变换通常用 3×3 矩阵，第三行与透视有关；PyTorch 的 `theta` 只取前两行（2×3），对应旋转, 缩放与平移共 6 个自由度：

- 前两列：旋转与缩放
- 最后一列：平移

`affine_grid` 根据 `theta` 和目标 `size` 生成 grid；grid 中每个位置记录"应从源图像哪个坐标采样"。该 grid 传入 `grid_sample` 即完成仿射变换。

## 实验：旋转 30°

```python
import cv2
import numpy as np
import torch
import torch.nn.functional as F

img = cv2.imread('./00000.jpg')
img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
size = img.shape[:2]
size = (1, 3,) + size

angle = 30 * np.pi / 180  # NumPy 三角函数使用弧度
theta = np.array([
    np.cos(angle), np.sin(-angle), 0,
    np.sin(angle),  np.cos(angle),  0,
], dtype=np.float32).reshape(1, 2, 3)

theta = torch.from_numpy(theta)
grid = F.affine_grid(theta, size=torch.Size(size), align_corners=False)
img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
warp_img = F.grid_sample(img_tensor, grid).squeeze().permute(1, 2, 0).numpy()
warp_img = np.clip(warp_img, 0, 255).astype(np.uint8)

img_vis = np.concatenate((img, warp_img), axis=0)  # 上：原图；下：变换后
cv2.imshow('res', img_vis)
cv2.waitKey()
```

效果：输出图像相对原图约逆时针旋转 30°；若出现拉伸，需检查 `size`, `align_corners` 及坐标归一化方式。

## 与 OpenCV 的对应关系

| PyTorch | OpenCV |
|---------|--------|
| 构造 2×3 仿射矩阵 | `getRotationMatrix2D` 等 |
| `affine_grid` + `grid_sample` | `warpAffine` |

区别在于 PyTorch 路径支持 autograd，可嵌入网络端到端训练。

## 参考

- 原文：[grid_sample 与 affine_grid（CSDN）](https://blog.csdn.net/qq_34914551/article/details/110186855)

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-einsum-爱因斯坦求和|PyTorch einsum 爱因斯坦求和]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-命名张量|PyTorch 命名张量]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-显示为图片|Tensor 显示为图片]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-标准化|Tensor 标准化]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
