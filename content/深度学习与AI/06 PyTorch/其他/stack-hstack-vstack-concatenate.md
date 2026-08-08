---
title: stack / hstack / vstack / concatenate
url: https://zhuanlan.zhihu.com/p/38184917
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:45:06+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
math_repaired_at: '2026-06-27T19:29:26+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# stack / hstack / vstack / concatenate

在写代码时，经常会遇到多个矩阵数组拼接的情况，numpy里stack, hstack, [vstack](https://zhida.zhihu.com/search?content_id=7465465&content_type=Article&match_order=1&q=vstack&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NTUwOTMsInEiOiJ2c3RhY2siLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3NDY1NDY1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.h1TbFhIJTf-9fPiufyvHoDXt7_ITH8yqO6NpQ2Ibki8&zhida_source=entity), concatenate都有拼接的作用，那么这些函数是怎么执行的，他们的结果又如何呢？

### Note
### shape = [2,3,4],则第一个轴为大小为2的轴
## 1. stack(arrays, axis=0)

```mermaid
flowchart LR
  raw["C 数组 T n"]
  arr["std::array T N"]
  raw --> arr
  arr --> safe["size 编译期固定"]
  arr --> stl["迭代器与算法兼容"]
```

```
mermaid
flowchart LR   raw["C 数组 T n"]   arr["std::array T N"]   raw --> arr   arr --> safe["size 编译期固定"]   arr --> stl["迭代器与算法兼容"]
mermaid flowchart LR   raw["C 数组 T n"]   arr["std::array T N"]   raw --> arr   arr --> safe["size 编译期固定"]   arr --> stl["迭代器与算法兼容"] Join a sequence of arrays along a new axis.
The `axis` parameter specifies the index of the new axis in the dimensions of the result. For example, if ``axis=0`` it will be the first dimension and if ``axis=-1`` it will be the last dimension.

### 这个函数肯定会新建一个轴
*   如果arrays矩阵至少有两个轴，则这个函数会在axis轴新建一个轴，把已经存在的轴挤到两边，在新建的轴上堆叠矩阵。如果第一个轴是步骤2虚拟出来的轴，且这个轴到最后还是1，则会取消这个轴。
*   如果arrays矩阵只有一个轴，则先把仅存的轴作为最后一个轴，前面用1来作为第一个轴的大小，接着回到第一步。
Examples     --------     >>> arrays = [np.random.randn(3, 4) for _ in range(10)]     >>> np.stack(arrays, axis=0).shape     (10, 3, 4)     >>> np.stack(arrays, axis=1).shape     (3, 10, 4)     >>> np.stack(arrays, axis=2).shape     (3, 4, 10)     --------     >>> a = np.array([1, 2, 3])     >>> b = np.array([2, 3, 4])     >>> np.stack((a, b))     array([[1, 2, 3],            [2, 3, 4]])     >>> np.stack((a, b), axis=-1)     array([[1, 2],            [2, 3],            [3, 4]])
* * *
## 2. hstack(tup)
Stack arrays in sequence horizontally (column wise).
All arrays must have the same shape along all but the second axis.
Notes     -----     Equivalent to ``np.concatenate(tup, axis=1)`` if `tup` contains arrays that     are at least 2-dimensional.
*   如果矩阵至少有两个轴，则这个函数会沿着第二个轴扩充矩阵。如果第一个轴是步骤2虚拟出来的轴，且这个轴到最后还是1，则会取消这个轴。
*   如果矩阵只有一个轴，则先把仅存的轴作为最后一个轴，前面用1来作为第一个轴的大小，接着回到第一步。
Examples     --------     >>> a = np.array((1,2,3))   # 只有一个轴存在    >>> b = np.array((2,3,4))     >>> np.hstack((a,b))     array([1, 2, 3, 2, 3, 4])     --------     >>> a = np.array([[1],[2],[3]])     >>> b = np.array([[2],[3],[4]])     >>> np.hstack((a,b))     array([[1, 2],            [2, 3],            [3, 4]])
* * *
## 3. vstack(tup)
Stack arrays in sequence vertically (row wise).
Tuple containing arrays to be stacked. The arrays must have the same         shape along all but the first axis.
Notes     -----     Equivalent to ``np.concatenate(tup, axis=0)`` if `tup` contains arrays that     are at least 2-dimensional.
*   如果矩阵至少有两个轴，则这个函数会沿着第一个轴扩充矩阵。如果第一个轴是步骤2虚拟出来的轴，且这个轴到最后还是1，则会取消这个轴。
Examples     --------     >>> a = np.array([1, 2, 3])     >>> b = np.array([2, 3, 4])     >>> np.vstack((a,b))     array([[1, 2, 3],            [2, 3, 4]])     --------     >>> a = np.array([[1], [2], [3]])     >>> b = np.array([[2], [3], [4]])     >>> np.vstack((a,b))     array([[1],            [2],            [3],            [2],            [3],            [4]])
* * *
## 4. concatenate((a1, a2, ...), axis=0)
Join a sequence of arrays along an existing axis.
这个函数只能沿着已经存在的轴堆叠矩阵，即axis必须在shape的长度范围内，否则会报'指定轴不存在'的错误。
Examples --------
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array(`[[5, 6]]`)
>>> np.concatenate((a, b), axis=0)

array([[1, 2],        [3, 4],        [5, 6]])
>>> np.concatenate((a, b.T), axis=1)

array([[1, 2, 5],        [3, 4, 6]])
>>> a = np.random.rand(3,)
>>> np.concatenate((a, a, a), axis=0)

array([ 0.28531257,  0.97033517,  0.87735775,  0.28531257,  0.97033517,         0.87735775,  0.28531257,  0.97033517,  0.87735775])
>>> np.concatenate((a, a, a), axis=1)   # 超出axis范围，报错

AxisError  Traceback (most recent call last) <ipython-input-32-f0106f21b6f0> in <module>() ----> 1 np.concatenate([a,a,a],1) AxisError: axis 1 is out of bounds for array of dimension 1

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[PyTorch-广播机制|PyTorch 广播机制]]
[[PyTorch-模型部署基础|PyTorch 模型部署基础]]
[[transforms.Compose-流水线|transforms.Compose 流水线]]
