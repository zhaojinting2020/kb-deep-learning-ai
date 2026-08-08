---
title: CycleGAN 训练流程
url: https://huaweicloud.csdn.net/638088f9dacf622b8df89cde.html
curated_at: '2026-06-28T20:00:00+00:00'
---

# CycleGAN 训练流程

基于 PyTorch CycleGAN 官方实现，梳理生成器/判别器构造, 前向传播与损失计算流程（`cycle_gan_model.py`, `networks.py`, `train.py`）。

## 模型初始化

`CycleGANModel.__init__` 中定义两对生成器与判别器：

```python
self.netG_A = networks.define_G(...)  # A → B
self.netG_B = networks.define_G(...)  # B → A

if self.isTrain:
    self.netD_A = networks.define_D(...)  # 判别 fake_B
    self.netD_B = networks.define_D(...)  # 判别 fake_A
```

- `G_A`, `G_B`：结构相同，输入/输出通道互换（域 A ↔ 域 B）
- `D_A`, `D_B`：PatchGAN 判别器，结构相同

## 生成器 `define_G`

```python
def define_G(input_nc, output_nc, ngf, netG, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
```

| `netG` | 结构 |
|--------|------|
| `resnet_9blocks` / `resnet_6blocks` | 下采样 → N 个 ResnetBlock → 上采样 |
| `unet_128` / `unet_256` | U-Net（128/256 输入） |

返回经 `init_net` 权重初始化并放到 GPU 的模块。

### ResnetBlock

带 skip connection 的卷积块，缓解梯度消失：

```python
class ResnetBlock(nn.Module):
    def forward(self, x):
        return x + self.conv_block(x)
```

单块流程：`padding → Conv2d → Norm → ReLU → [Dropout] → Conv2d → Norm`。

### ResnetGenerator 整体

```
输入 → ReflectionPad → Conv7×7 → Norm → ReLU
     → 下采样 ×2（Conv stride=2）
     → ResnetBlock ×N
     → 上采样 ×2（ConvTranspose2d）
     → ReflectionPad → Conv7×7 → Tanh → 输出
```

## 判别器 NLayerDiscriminator（PatchGAN）

```
Conv → LeakyReLU → [Conv → Norm → LeakyReLU]×(n_layers-1) → Conv1×1 → 输出 patch 图
```

输出为 **patch 级** 真/假图，而非单标量。

## 前向传播

```python
def forward(self):
    self.fake_B = self.netG_A(self.real_A)   # G_A(A)
    self.rec_A   = self.netG_B(self.fake_B)  # G_B(G_A(A))
    self.fake_A = self.netG_B(self.real_B)   # G_B(B)
    self.rec_B   = self.netG_A(self.fake_A)  # G_A(G_B(B))
```

## 训练主循环

`train.py` 每个 epoch 遍历 dataloader，调用 `model.optimize_parameters()`：

```python
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    model.update_learning_rate()
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.optimize_parameters()
```

## `optimize_parameters`

```python
def optimize_parameters(self):
    self.forward()

    # 更新 G
    self.set_requires_grad([self.netD_A, self.netD_B], False)
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()

    # 更新 D
    self.set_requires_grad([self.netD_A, self.netD_B], True)
    self.optimizer_D.zero_grad()
    self.backward_D_A()
    self.backward_D_B()
    self.optimizer_D.step()
```

`set_requires_grad` 在更新 G 时关闭 D 的梯度，避免多余计算。

## 生成器损失 `backward_G`

三类损失：

| 损失 | 含义 |
|------|------|
| **GAN Loss** | 骗过 \(D_A\), \(D_B\) |
| **Cycle Loss** | \(\|G_B(G_A(A))-A\|\), \(\|G_A(G_B(B))-B\|\)（L1） |
| **Identity Loss**（可选） | \(\|G_A(B)-B\|\), \(\|G_B(A)-A\|\)，当 `lambda_identity > 0` |

超参：`lambda_A`, `lambda_B`（cycle 权重），`lambda_identity`（相对 cycle 的比例）。

```python
def backward_G(self):
    lambda_idt = self.opt.lambda_identity
    lambda_A = self.opt.lambda_A
    lambda_B = self.opt.lambda_B

    if lambda_idt > 0:
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    else:
        self.loss_idt_A = self.loss_idt_B = 0

    self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

    self.loss_G = (self.loss_G_A + self.loss_G_B +
                   self.loss_cycle_A + self.loss_cycle_B +
                   self.loss_idt_A + self.loss_idt_B)
    self.loss_G.backward()
```

初始化时：

```python
self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
self.criterionCycle = torch.nn.L1Loss()
self.criterionIdt = torch.nn.L1Loss()
```

## GANLoss

支持 `lsgan`（MSE）, `vanilla`（BCEWithLogits）, `wgangp`（Wasserstein，代码中 wgangp 分支未完整接入训练）。

```python
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # lsgan → MSELoss; vanilla → BCEWithLogitsLoss
```

`get_target_tensor` 将 label 扩展为与 prediction 同 shape；`__call__(prediction, target_is_real)` 计算损失。判别器 `backward_D_A` / `backward_D_B` 结构类似：对 fake 判假, 对 real 判真，分别反传。

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)

[[content/深度学习与AI/06 PyTorch/卷积与视觉模型/Conv2d-二维卷积|Conv2d 二维卷积]]

[[content/深度学习与AI/06 PyTorch/卷积与视觉模型/MobileNet-五类花-Colab|MobileNet 五类花 Colab]]
