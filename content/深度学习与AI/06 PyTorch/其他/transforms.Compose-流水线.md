---
title: transforms.Compose 流水线
url: https://blog.csdn.net/u013925378/article/details/103363232
curated_at: '2026-06-28T20:00:00+00:00'
---

# transforms.Compose 流水线

`torchvision.transforms` 提供图像预处理算子；`Compose` 将多个变换按顺序串联成一条流水线。

## 基本用法

```python
from torchvision import transforms

# ImageNet 常见归一化
self.norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# 裁剪 + 转 Tensor
transform = transforms.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),
])
```

`Compose([T1, T2, ...])` 依次对 PIL / Tensor 图像应用 `T1`, `T2`……输出作为下一步输入。

## 常用变换

| 变换 | 作用 |
|------|------|
| `Resize` | 缩放到指定尺寸 |
| `Normalize` | 按均值, 标准差归一化 Tensor |
| `ToTensor` | PIL (H×W×C, [0,255]) → Tensor (C×H×W, [0,1]) |
| `ToPILImage` | Tensor → PIL |
| `Scale` | 已弃用，请用 `Resize` |
| `CenterCrop` | 中心裁剪 |
| `RandomCrop` | 随机位置裁剪 |
| `RandomHorizontalFlip` | 50% 概率水平翻转 |
| `RandomVerticalFlip` | 50% 概率垂直翻转 |
| `RandomResizedCrop` | 随机区域 + 随机缩放裁剪 |
| `Grayscale` | 转灰度 |
| `RandomGrayscale` | 按概率转灰度 |
| `FiveCrop` | 四角 + 中心五块裁剪 |
| `TenCrop` | 五块 + 水平翻转共十块 |
| `Pad` | 填充 |
| `ColorJitter` | 随机调整亮度, 对比度, 饱和度 |

## 与 DataLoader 配合

通常在 `Dataset.__getitem__` 中对单张样本调用 `transform`；训练集与验证集可分别定义不同的 `Compose` 流水线（训练加强数据增强，验证仅做尺度/归一化）。

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
