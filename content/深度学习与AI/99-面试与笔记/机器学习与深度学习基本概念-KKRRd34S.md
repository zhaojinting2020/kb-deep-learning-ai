---
title: 机器学习与深度学习基本概念
url: https://www.feishu.cn/docx/KKRRd34SgoOLOZxFSdccwaCInBb
quality: raw
fetch_source: feishu:cli
fetched_at: '2026-06-28T08:21:28+00:00'
sheets_expanded_at: '2026-06-28T08:23:28+00:00'
---

# 机器学习的经典算法

想快速了解所有机器学习模型，请看 [挑战只用17分钟讲完所有的机器学习模型！\_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1GgkSYBEpD/?spm_id_from=333.1391.0.0&vd_source=c8041efd376e7f34e73272f6ae86b7a5)

想系统学习所有机器学习算法，请看：

# 神经网络的经典架构

经典基础架构

| 架构 | 简介 |
| --- | --- |
| MLP（多层感知机） | 最基础的前馈神经网络，适用于结构化数据。 |
| CNN（卷积神经网络） | 图像处理界的王者，用于提取局部空间特征。 |
| RNN（循环神经网络） | 处理序列数据（如文本, 音频），有记忆能力。 |
| LSTM / GRU | RNN 的改进版本，缓解长距离依赖问题。 |

[Stanford University CS231n: Deep Learning for Computer VisionDeep Learning for Computer Vision](https://cs231n.stanford.edu/)

[【重温经典】GRU循环神经网络 —— LSTM的轻量级版本，大白话讲解_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1454y1n7uY/?spm_id_from=333.337.search-card.all.click&vd_source=c8041efd376e7f34e73272f6ae86b7a5)

进阶架构（革命性）

| 架构 | 简介 |
| --- | --- |
| Transformer | 用自注意力机制替代 RNN，支持并行计算，是 NLP 和多模态主力。 |
| GAN（生成对抗网络） | 由生成器+判别器组成，擅长生成图像, 音频等。 |
| VAE（变分自编码器） | 用概率方式建模潜在空间的生成模型，比 GAN 更稳定。 |

[一小时从函数到Transformer！一路大白话彻底理解AI原理_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1NCgVzoEG9/?spm_id_from=333.337.search-card.all.click)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[3Blue1Brown - Visualizing Attention, a Transformer's Heart | Chapter 6, Deep Learning](https://www.3blue1brown.com/lessons/attention)

图像领域专属架构（CNN家族升级）

| 架构 | 简介 |
| --- | --- |
| LeNet | 最早的 CNN（1998 年），用于手写数字识别。 |
| AlexNet | 引爆深度学习的关键网络（2012）。 |
| VGG | 用重复的 3x3 卷积层堆深。 |
| GoogLeNet (Inception) | 多尺度卷积合并。 |
| ResNet | 引入残差连接，解决深层网络梯度消失问题。 |
| DenseNet | 所有层都互相连接，更高效利用特征。 |
| EfficientNet | 用搜索方法自动调节网络宽度/深度/分辨率。 |
| Vision Transformer (ViT) | 用 Transformer 来做图像处理，放弃卷积。 |

图结构相关

| 架构 | 简介 |
| --- | --- |
| GCN（图卷积网络） | 处理社交网络, 分子结构等图数据。 |
| GAT（图注意力网络） | 引入注意力机制来学习邻居节点的重要性。 |

特殊用途结构

| 架构 | 简介 |
| --- | --- |
| AutoEncoder | 编码-解码结构，用于降维, 去噪等。 |
| Siamese Network | 比较两个输入是否相似（人脸比对等）。 |
| U-Net | 图像分割专用架构，常用于医学图像。 |
| NeRF（神经辐射场） | 3D重建用网络，用于合成新视角图像。 |
| Slot Attention / Perceiver / Mamba | 新一代架构，试图替代传统 Transformer. |

小结：

- 分类/提取特征：CNN, MLP, ResNet, ViT
- 序列建模：RNN, LSTM, GRU, Transformer, Mamba
- 生成模型：GAN, VAE, AutoEncoder, NeRF
- 图结构：GCN, GAT
- 多模态/预训练：BERT, GPT, CLIP, SAM
- 特殊任务：U-Net, Siamese, Slot Attention

# 经典深度学习领域

| 领域 | 简单解释 | 典型任务 / 应用 | 代表模型 |
| --- | --- | --- | --- |
| 🖼️ 计算机视觉（CV） | 看图像 | 分类, 检测, 分割, 生成 | CNN, ViT, YOLO, UNet |
| 📢 自然语言处理（NLP） | 看文字 | 翻译, 问答, 摘要, 聊天 | RNN, LSTM, BERT, GPT |
| 🔊 语音处理（Speech） | 听声音 | 识别, 合成, 唤醒词检测 | DeepSpeech, Wav2Vec |
| 🧠 强化学习（RL） | 学策略 | 玩游戏, 控制机器人 | DQN, PPO, A3C |
| 🧬 多模态学习（Multimodal） | 图文声一起看 | 图文检索, VQA, AI生成 | CLIP, BLIP, Flamingo |
| 🧪 生成模型（Generative Models） | 生成数据 | GAN画画, Diffusion合图 | GAN, VAE, Diffusion |
| 🔍 表征学习（Representation Learning） | 学特征 | 特征提取, 无监督学习 | SimCLR, BYOL, MoCo |
| 🕸️ 图神经网络（GNN） | 学图结构 | 社交推荐, 分子建模 | GCN, GAT, GraphSAGE |

# 深度学习 / 机器学习模型开发经典流程

![image](attachments/KKRRd34SgoOLOZxFSdccwaCInBb/img_001.png)

# 深度学习/机器学习开发工具

| 步骤 | 工具 / 框架 |
| --- | --- |
| 数据准备 | pandas, numpy, label-studio |
| 数据增强 | albumentations, torchaudio.transforms |
| 模型设计 | PyTorch, TensorFlow, HuggingFace |
| 训练 & 调参 | wandb, TensorBoard, Optuna |
| 评估 | scikit-learn, seaborn, matplotlib |
| 部署 | ONNX, TensorRT, FastAPI, Flask, TFLite |

[ONNX and ONNX Runtime_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1114y1v7bL?spm_id_from=333.788.recommend_more_video.1&vd_source=c8041efd376e7f34e73272f6ae86b7a5)
