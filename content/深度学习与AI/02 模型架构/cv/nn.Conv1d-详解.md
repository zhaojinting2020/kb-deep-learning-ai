---
title: nn.Conv1d 详解
url: https://blog.csdn.net/sunny_xsc1994/article/details/82969867
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:56:36+00:00'
curated_at: '2026-06-28T10:00:00+00:00'
---

# nn.Conv1d 详解

PyTorch 一维卷积常用于**文本分类**：在词向量序列上滑动卷积核，提取 n-gram 级局部特征。参考：[Keras 文本分类中的 Conv1D 图示](https://zhuanlan.zhihu.com/p/29201491)

## API

```python
torch.nn.Conv1d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, dilation=1, groups=1, bias=True
)
```

| 参数 | 含义 |
|------|------|
| `in_channels` | 输入通道数；文本任务中即**词向量维度** |
| `out_channels` | 输出通道数；每个 out_channel 对应一个一维卷积核 |
| `kernel_size` | 卷积核长度 k；实际感受野为 k × in_channels（沿序列维滑动） |
| `stride` / `padding` / `dilation` | 同二维卷积语义 |
| `groups` | 分组卷积 |
| `bias` | 是否加偏置 |

## 输入形状

Conv1d 在**最后一维（序列长度）** 上卷积，因此输入 layout 为：

```
(batch, in_channels, seq_len)
```

文本 embedding 通常为 `(batch, seq_len, embed_dim)`，需 **`permute(0, 2, 1)`** 后再送入 Conv1d.

### 示例

```python
import torch.nn as nn

conv1 = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=2)
input = torch.randn(32, 35, 256)   # batch=32, len=35, dim=256
input = input.permute(0, 2, 1)     # → (32, 256, 35)
out = conv1(input)

# out.size(): (32, 100, 34)  因为 35 - 2 + 1 = 34
```

## 多尺度卷积 + 池化（TextCNN 思路）

词向量维度假设为 5，序列长 7，使用 kernel_size = 2, 3, 4 各 2 个卷积核：

- 对 k=4：在序列上滑动 4×5 窗口，步长 1，得到长度 `((7-4)/1+1)=4` 的特征，再经 **MaxPool1d(kernel_size=4)** 压成 1 个标量。
- 6 路特征拼接后接全连接分类。

## TextCNN 参考实现

```python
class TextCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.use_element = config.use_element

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.embedding_size,
                    out_channels=config.feature_size,
                    kernel_size=h,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=config.max_text_len - h + 1),
            )
            for h in config.window_sizes
        ])
        self.fc = nn.Linear(
            in_features=config.feature_size * len(config.window_sizes),
            out_features=config.num_class,
        )

    def forward(self, x):
        embed_x = self.embedding(x)           # (B, L, E)
        embed_x = embed_x.permute(0, 2, 1)    # (B, E, L)
        out = [conv(embed_x) for conv in self.convs]  # 每路 (B, F, 1)
        out = torch.cat(out, dim=1)           # (B, F*num_kernels, 1)
        out = out.view(-1, out.size(1))       # (B, F*num_kernels)
        out = F.dropout(out, p=self.dropout_rate)
        out = self.fc(out)
        return out
```

数据流（batch=32, len=35, embed=256, 4 个 kernel）：

1. `(32, 35, 256)` → permute → `(32, 256, 35)`
2. 每路 conv+pool → `(32, 100, 1)`，共 4 路
3. cat → `(32, 400)` → fc → `(32, num_class)`
