---
title: Tensor 标准化
url: https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T20:05:43+00:00'
polished_at: '2026-06-27T19:21:52+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Tensor 标准化

## post by bapriddy on Feb 28, 2019
Is there a pytorch command that scales tensors like sklearn (example below)?

X = data[:,:num_inputs]

x_scaler = preprocessing.StandardScaler()
X_scaled = x_scaler.fit_transform(X)
```
From class

`sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)`

## post by ptrblck on Feb 28, 2019
You can easily clone the sklearn behavior using this small script:
x = torch.randn(10, 5) * 10
scaler = StandardScaler()
arr_norm = scaler.fit_transform(x.numpy())

# PyTorch impl

m = x.mean(0, keepdim=True)
s = x.std(0, unbiased=False, keepdim=True)
x -= m
x /= s

torch.allclose(x, torch.from_numpy(arr_norm))
```
![Image 3: :wink:](https://discuss.pytorch.org/images/emoji/apple/wink.png?v=6)

<p class="kb-image-caption">图例</p>

I have an input that has `required_grad=True`. I need to scale it, and I wondered if the solution in this post would break the graph such that the gradient is not computable later?

## post by ptrblck on Apr 26, 2020
The `x.numpy()` operation will break the computation graph, so you should use the plain PyTorch approach.

If you need gradients for the input, I would also recommend to not normalize it inplace, but create a new normalized tensor.

## post by muammar on Apr 26, 2020
![Image 8: :slight_smile:](https://discuss.pytorch.org/images/emoji/apple/slight_smile.png?v=9)

<p class="kb-image-caption">图例</p>
    # Similar to: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    def __init__(self, num_features):
        super().__init__()

        self._num_features = num_features
        self.register_buffer('mean', torch.full((self._num_features,), np.nan))
        self.register_buffer('var', torch.full((self._num_features,), np.nan))
    def _is_initialized(self):
        if torch.isnan(self.mean).any() or torch.isnan(self.var).any():
            return False
        return True
    @torch.no_grad()

    def initialize(self, input_batch):
        """
        Args:

            input_batch: Batch that should represent *all* inputs for a given
                dataset.
        Example:

            norm.initialize(torch.cat([x for (x, _) in train_loader]))
        """

        # TODO(eric.cousineau): Use an accurate running computation?
        # See: https://github.com/pytorch/pytorch/blob/480851ad/aten/src/ATen/native/Normalization.cpp#L215-L269
        assert not self._is_initialized()
        N, L = input_batch.shape

        assert L == self._num_features
        assert N > 1

        var_mean = torch.var_mean(input_batch, dim=0)
        self.var.data[:], self.mean.data[:] = var_mean
    def forward(self, x):

        if not self._is_initialized():
            raise RuntimeError("This must be initialized on the dataset!")
        y = (x - self.mean) / torch.sqrt(self.var)
        return y
```
# Terse test code

class TestStuff(unittest.TestCase):
    def test_dataset_norm(self):
        xs = torch.Tensor([
            [1., 10.],
            [2., 20.],
            [3., 30.],
            [4., 40.],
        ])
        norm = mut.DatasetNorm1d(2)

        with self.assertRaises(RuntimeError):
            norm(xs)
        norm.initialize(xs)
        ys = norm(xs)

        np.testing.assert_array_equal(ys[:, 0].numpy(), ys[:, 1].numpy())
        self.assertEqual(list(norm.state_dict().keys()), ["mean", "var"])
```
## post by Matthew_Chung on Jul 7, 2020
2 years later

## post by Sithara85 on Oct 12, 2022
1 year later

## post by meh-deh on Apr 6, 2024
[Powered by Discourse](https://discourse.org/powered-by)

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-显示为图片|Tensor 显示为图片]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-einsum-爱因斯坦求和|PyTorch einsum 爱因斯坦求和]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制-qq|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-命名张量|PyTorch 命名张量]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/TensorFlow-2-与-Keras-入门|TensorFlow 2 与 Keras 入门]]
