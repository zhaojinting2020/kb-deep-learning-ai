---
title: PyTorch Lightning 完全攻略
url: https://zhuanlan.zhihu.com/p/410394574
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:43:43+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
math_repaired_at: '2026-06-27T19:29:26+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# PyTorch Lightning 完全攻略

​目录[Pytorch-Lightning](https://zhida.zhihu.com/search?content_id=179329113&content_type=Article&match_order=1&q=Pytorch-Lightning&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTUwMDksInEiOiJQeXRvcmNoLUxpZ2h0bmluZyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjE3OTMyOTExMywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.zRNpdUDV9qeZvSOVv_OyWCig86LeKux0OFXjZAe3pX0&zhida_source=entity)的官网教程写得蛮好，但是内容比较多，初学者不容易接受。这里结合个人感悟做个提炼。

## 写在最前

本教程只是官网最简范例的一个讲解，强烈建议读者学习[官方文档](https://link.zhihu.com/?target=https%3A//pytorch-lightning.readthedocs.io/en/latest/)的各个章节。里面有很多细节，我个人受益匪浅。

## Simplest example

Simplest example为最核心内容，只包含train，没有val和test.
import os import torch from pytorch_lightning import LightningModule, Trainer from pytorch_lightning.metrics.functional import accuracy from torch import nn from torch.nn import functional as F from torch.utils.data import DataLoader, random_split from torchvision import transforms from torchvision.datasets import MNIST PATH_DATASETS = os.environ.get('PATH_DATASETS', '.') AVAIL_GPUS = min(1, torch.cuda.device_count()) BATCH_SIZE = 256 if AVAIL_GPUS else 64

# 该范例只包含train，没有val和test
# 关键一：定义LightningModule的模型
class MNISTModel(LightningModule):

    # 以下四个函数是必须要有的，分别对应    # __init__：模型初始化    # forward：模型前向传递过程，主要指val和test，但我也推荐在train中使用，保持代码统一    # training_step:单次的训练过程    # configure_optimizers: 优化器定义    def __init__(self):
        super().__init__()         self.l1 = torch.nn.Linear(28 * 28, 10)     def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))     def training_step(self, batch, batch_nb):
        x, y = batch         loss = F.cross_entropy(self(x), y)
        return loss     def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) if __name__ == '__main__':

    # Init our model     mnist_model = MNISTModel()     # Init DataLoader from MNIST Dataset     train_ds = MNIST(PATH_DATASETS,                      train=True,                      download=True,                      transform=transforms.ToTensor()           		)     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)     # 关键二：实例化训练器trainer     trainer = Trainer(         gpus=AVAIL_GPUS,         max_epochs=3,         progress_bar_refresh_rate=20,     )     # 关键三,用trainer.fit(...)调用训练⚡    trainer.fit(mnist_model, train_loader)
### Simplest Example讲解

### Pytorch-Lightning中最重要的两个模块是`model`和`trainer`。
### model：继承自`LightningModule`，`LightningModule`是`nn.Module`扩展后的子类，相比于`nn.Module`，`LightningModule`将之前需要额外定义的训练过程，验证过程，优化器，等一系列功能全部集成到了LightningModule模中。`nn.Module`有的功能和属性`LightningModule`都有。
![Image 1](https://pic3.zhimg.com/v2-a2d0e0cfcaa9e4bd2dc9df4b4dbb5d9c_1440w.jpg)

<p class="kb-image-caption">图例</p>

如：用DataParallel还是用[DDP](https://zhida.zhihu.com/search?content_id=179329113&content_type=Article&match_order=1&q=DDP&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTUwMDksInEiOiJERFAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzkzMjkxMTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.2_6VCFd888YavaqOIIhsMw7IRu1Z52mmS8ckm96AZek&zhida_source=entity)，训练多少个epoch，在多少个epoch时记录数据，保存模型，调整学习率等。在trainer上只写如何触发这个事件，不写具体如何做。具体如何做则定义在LightingModule中。[触发事件详见Callback章节](https://link.zhihu.com/?target=https%3A//pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.Callback.html%23pytorch_lightning.callbacks.Callback)，用到再查来得及。恭喜您现在已经学会了Pytorch-Lightning的Simplest example了，是不是很简单？

## Complete Example

本章将介绍一个更复杂的范例。详细介绍都在代码中文注释中。
import os import torch from pytorch_lightning import LightningModule, Trainer from torchmetrics.functional.classification.accuracy import accuracy from torch import nn from torch.nn import functional as F from torch.utils.data import DataLoader, random_split from torchvision import transforms from torchvision.datasets import MNIST PATH_DATASETS = os.environ.get('PATH_DATASETS', '.') AVAIL_GPUS = min(1, torch.cuda.device_count()) BATCH_SIZE = 256 if AVAIL_GPUS else 64
class LitMNIST(LightningModule):
    #-----------------------模型相关------------------------ 	# __init__：初始化可能用到的参数，网络的Pytorch model     # forward：模型前向传递过程，主要指val和test，但我也推荐在train中使用，保持代码统一    # training_step：单次的训练过程    # validation_step：单次的验证过程    # test_step：单次的测试过程，常与validation_step同    # configure_optimizers：优化器定义    #-----------------------数据相关------------------------     # prepare_data：下载，train和test如果都有的话则都要，val常从train中分出    # setup：预设数据，设置如何构造Dataset，利用fit和test的区别进行区分。

    # train_dataloader：返回train的dataloader     # val_dataloader：返回val的dataloader     # test_dataloader：返回test的dataloader     # NOTE：三个dataloader也可以在LightingModule外定义，    #       但为了迁移时更好的稳定性，官网推荐当前的这种方法。
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
        super().__init__()         # Set our init args as class attributes         self.data_dir = data_dir         self.hidden_size = hidden_size         self.learning_rate = learning_rate         # Hardcode some dataset specific attributes         self.num_classes = 10         self.dims = (1, 28, 28)         channels, width, height = self.dims         self.transform = transforms.Compose([             transforms.ToTensor(),             transforms.Normalize((0.1307, ), (0.3081, )),         ])         # Define PyTorch model         self.model = nn.Sequential(             nn.Flatten(),             nn.Linear(channels * width * height, hidden_size),             nn.ReLU(),             nn.Dropout(0.1),             nn.Linear(hidden_size, hidden_size),             nn.ReLU(),             nn.Dropout(0.1),             nn.Linear(hidden_size, self.num_classes),         )     def forward(self, x):
        x = self.model(x)

        return F.log_softmax(x, dim=1)     def training_step(self, batch, batch_idx):
        x, y = batch         logits = self(x)         loss = F.nll_loss(logits, y)
        return loss     def validation_step(self, batch, batch_idx):
        x, y = batch         logits = self(x)         loss = F.nll_loss(logits, y)         preds = torch.argmax(logits, dim=1)         acc = accuracy(preds, y)         # Calling self.log will surface up scalars for you in TensorBoard         self.log('val_loss', loss, prog_bar=True)         self.log('val_acc', acc, prog_bar=True)
        return loss     def test_step(self, batch, batch_idx):

        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)     def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer     ####################     # DATA RELATED HOOKS     ####################     def prepare_data(self):

        # download         MNIST(self.data_dir, train=True, download=True)         MNIST(self.data_dir, train=False, download=True)     def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders         if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)             self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])         # Assign test dataset for use in dataloader(s)         if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)     def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)     def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)     def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE) if __name__ == '__main__':

    # 关键一：实例化LightingModule模型    model = LitMNIST()     # 关键二：实例化trainer     trainer = Trainer(         gpus=AVAIL_GPUS,         max_epochs=3,         progress_bar_refresh_rate=20,     )     # 训练该模型，val也包含在内    trainer.fit(model)     # 测试该模型，test不跟fit放在一起，    # 为了防止误操作，避免在train和val的过程中就使用了test的数据    trainer.test()以上既是可用的最小范例，进阶教程主要是trainer和LightingModule中各种参数的使用，[详见官网文档](https://link.zhihu.com/?target=https%3A//pytorch-lightning.readthedocs.io/en/latest/)

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/Lightning与训练/PyTorch-Lightning-完全攻略|PyTorch Lightning 完全攻略]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-学习率调度|Lightning Bolts 学习率调度]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-学习率调度|Lightning Bolts 学习率调度]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Bolts-工具箱|Lightning Bolts 工具箱]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Hooks-与-Callbacks|Lightning Hooks 与 Callbacks]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/Lightning-Hooks-与-Callbacks|Lightning Hooks 与 Callbacks]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/CosineAnnealingLR-余弦退火|CosineAnnealingLR 余弦退火]]
[[content/深度学习与AI/06 PyTorch/Lightning与训练/ReduceLROnPlateau-学习率调度|ReduceLROnPlateau 学习率调度]]
