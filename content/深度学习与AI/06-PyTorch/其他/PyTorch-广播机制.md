---
title: PyTorch 广播机制
url: https://zhuanlan.zhihu.com/p/86997775#:~:text=torch%E7%9A%84%E5%B9%BF%E6%92%AD%E6%9C%BA%E5%88%B6
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:44:05+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
math_repaired_at: '2026-06-27T20:24:23+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# PyTorch 广播机制

# torch的广播机制(broadcast mechanism)
[![Image 4: 营长](https://pic1.zhimg.com/v2-cc9ed697ae666b9db95d3396dc54777d_l.jpg?source=32738c0c&needBackground=1)](https://www.zhihu.com/people/lizhiweiena)

<p class="kb-image-caption">图例</p>
*    在满足特定限制的前提下，**较小的数组“广播至”较大的数组**，使两者形状互相兼容。广播提供了一个[向量化数组操作](https://zhida.zhihu.com/search?content_id=107462192&content_type=Article&match_order=1&q=%E5%90%91%E9%87%8F%E5%8C%96%E6%95%B0%E7%BB%84%E6%93%8D%E4%BD%9C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3MzI1NzgsInEiOiLlkJHph4_ljJbmlbDnu4Tmk43kvZwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMDc0NjIxOTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.rmAYo4cnUHMNyxReYyBYweMJoDdMw556dCCYEp1iMko&zhida_source=entity)的机制，这样遍历就发生在C层面，而不是Python层面。广播可以避免不必要的数据复制，通常导向高效的算法实现。不过，也存在不适用广播的情形（可能导致拖慢计算过程的低效[内存使用](https://zhida.zhihu.com/search?content_id=107462192&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E4%BD%BF%E7%94%A8&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3MzI1NzgsInEiOiLlhoXlrZjkvb_nlKgiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMDc0NjIxOTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.IXCwuXoP9crhaY5QPffR7G41s7yxWWcbltgo8K2qeFw&zhida_source=entity)）。
*    可广播的一对张量需满足以下规则：
  *    每个张量至少有一个维度。
*    迭代维度尺寸时，从**尾部**的维度开始，维度尺寸

 ​ 或者**相等**， ​ 或者**其中一个张量的维度尺寸为 1** ， ​ 或者**其中一个张量不存在这个维度**。
python import torch

# 示例1：相同形状的张量总是可广播的，因为总能满足以上规则。
x = torch.empty(5, 7, 3) y = torch.empty(5, 7, 3)

# 示例2：不可广播（ a 不满足第一条规则）。
a = torch.empty((0,)) b = torch.empty(2, 2)

# 示例3：m 和 n 可广播
m = torch.empty(5, 3, 4, 1) n = torch.empty(   3, 1, 1)

# 倒数第一个维度：两者的尺寸均为1
# 倒数第二个维度：n尺寸为1
# 倒数第三个维度：两者尺寸相同
# 倒数第四个维度：n该维度不存在
# 示例4：不可广播，因为倒数第三个维度：2 != 3
p = torch.empty(5, 2, 4, 1) q = torch.empty(   3, 1, 1)
*    现在你对“可广播”这一概念已经有所了解了，让我们看下，**广播后的张量是什么样的**。
*    如果张量x和张量y是可广播的，那么广播后的张量尺寸按照如下方法计算：
  *   **如果x和y的维数不等，在维数较少的张量上添加尺寸为 1 的维度。结果维度尺寸是x和y相应维度尺寸的较大者。**

python

# 示例5：可广播
c = torch.empty(5, 1, 4, 1) d = torch.empty(   3, 1, 1) (c + d).size()  # torch.Size([5, 3, 4, 1])

# 示例6：可广播
f = torch.empty(      1) g = torch.empty(3, 1, 7) (f + g).size()  # torch.Size([3, 1, 7])

# 示例7：不可广播
o = torch.empty(5, 2, 4, 1) u = torch.empty(   3, 1, 1) (o + u).size()

# 报错
# ---------------------------------------------------------------------------
#

# RuntimeError Traceback (most recent call last)
#

# <ipython-input-17-72fb34250db7> in <module>()
# 1 o=torch.empty(5,2,4,1)
# 2 u=torch.empty(3,1,1)
# ----> 3 (o+u).size()
#

# RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
# 三维张量 (b1, n, m) = (2, 3, 4)
A = torch.randn(2, 3, 4)

# 三维张量 (b2, m, p) = (1, 4, 5)
B = torch.randn(1, 4, 5)

# 结果: (max(b1,b2), n, p) = (2, 3, 5)
C = torch.matmul(A, B) print(C.shape) # 输出: torch.Size([2, 3, 5])请问如何用你上面提到的规则解释这个例子？
2023-10-28​回复​喜欢点击查看全部评论[](https://zhuanlan.zhihu.com/p/86997775)关于作者

[![Image 25: 营长](https://pic1.zhimg.com/v2-cc9ed697ae666b9db95d3396dc54777d_l.jpg?source=32738c0c&needBackground=1)](https://www.zhihu.com/people/lizhiweiena)

<p class="kb-image-caption">图例</p>

[回答 **48**](https://www.zhihu.com/people/lizhiweiena/answers)[文章 **31**](https://www.zhihu.com/people/lizhiweiena/posts)[关注者 **542**](https://www.zhihu.com/people/lizhiweiena/followers)​关注他​发私信大家都在搜换一换[中国银行逃税近 24 亿被通报 392 万](https://www.zhihu.com/search?q=%E4%B8%AD%E5%9B%BD%E9%93%B6%E8%A1%8C%E9%80%83%E7%A8%8E%E8%BF%91+24+%E4%BA%BF%E8%A2%AB%E9%80%9A%E6%8A%A5&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)热[疑似小米员工发文置身米内 357 万](https://www.zhihu.com/search?q=%E7%96%91%E4%BC%BC%E5%B0%8F%E7%B1%B3%E5%91%98%E5%B7%A5%E5%8F%91%E6%96%87%E7%BD%AE%E8%BA%AB%E7%B1%B3%E5%86%85&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)热[杨紫首获白玉兰视后 329 万](https://www.zhihu.com/search?q=%E6%9D%A8%E7%B4%AB%E9%A6%96%E8%8E%B7%E7%99%BD%E7%8E%89%E5%85%B0%E8%A7%86%E5%90%8E&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)热[法国4比1挪威 318 万](https://www.zhihu.com/search?q=%E6%B3%95%E5%9B%BD4%E6%AF%941%E6%8C%AA%E5%A8%81&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)热[世界杯 32 强已确定 28 席 312 万](https://www.zhihu.com/search?q=%E4%B8%96%E7%95%8C%E6%9D%AF+32+%E5%BC%BA%E5%B7%B2%E7%A1%AE%E5%AE%9A+28+%E5%B8%AD&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)新[026 知乎毕业季线下作品展 308 万](https://www.zhihu.com/parker/campaign/2047734146286490237?zh_hide_nav_bar=true)活动[湖南两女子被关笼中游街 305 万](https://www.zhihu.com/search?q=%E6%B9%96%E5%8D%97%E4%B8%A4%E5%A5%B3%E5%AD%90%E8%A2%AB%E5%85%B3%E7%AC%BC%E4%B8%AD%E6%B8%B8%E8%A1%97&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)热[王俊凯的小鸡被霸凌去世了 277 万](https://www.zhihu.com/search?q=%E7%8E%8B%E4%BF%8A%E5%87%AF%E7%9A%84%E5%B0%8F%E9%B8%A1%E8%A2%AB%E9%9C%B8%E5%87%8C%E5%8E%BB%E4%B8%96%E4%BA%86&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)热[韩国网友怒骂德国故意输球 259 万](https://www.zhihu.com/search?q=%E9%9F%A9%E5%9B%BD%E7%BD%91%E5%8F%8B%E6%80%92%E9%AA%82%E5%BE%B7%E5%9B%BD%E6%95%85%E6%84%8F%E8%BE%93%E7%90%83&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content) [钟南山叮嘱学生抓住两个窗口期 254 万](https://www.zhihu.com/search?q=%E9%92%9F%E5%8D%97%E5%B1%B1%E5%8F%AE%E5%98%B1%E5%AD%A6%E7%94%9F%E6%8A%93%E4%BD%8F%E4%B8%A4%E4%B8%AA%E7%AA%97%E5%8F%A3%E6%9C%9F&search_source=Trending&utm_content=search_hot&utm_medium=organic&utm_source=zhihu&type=content)

### 推荐阅读

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/其他/PyTorch-模型部署基础|PyTorch 模型部署基础]]
[[content/深度学习与AI/06 PyTorch/其他/stack-hstack-vstack-concatenate|stack / hstack / vstack / concatenate]]
[[content/深度学习与AI/06 PyTorch/其他/transforms.Compose-流水线|transforms.Compose 流水线]]
