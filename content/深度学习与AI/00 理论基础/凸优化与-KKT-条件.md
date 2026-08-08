---
title: 凸优化与 KKT 条件
url: https://zhuanlan.zhihu.com/p/87485105
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T20:08:56+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
math_repaired_at: '2026-06-27T20:23:29+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# 凸优化与 KKT 条件

## [凸函数](https://zhida.zhihu.com/search?content_id=107570581&content_type=Article&match_order=1&q=%E5%87%B8%E5%87%BD%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NjM3MjUsInEiOiLlh7jlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMDc1NzA1ODEsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.SWYrhhW_CTihkpXPWPH748BjLl-mnPIcTtoWFzrzX20&zhida_source=entity)
### 凸函数的定义
![Image 1](https://pic1.zhimg.com/v2-39878e792ba3961f5f2e1209f78d1496_1440w.jpg)

<p class="kb-image-caption">图例</p>

证明（先拆出f(x)，然后分母是$\left( 1 - \theta \right) \left( y - x \right)$(1-\theta)(y-x) ）：
![Image 2](https://picx.zhimg.com/v2-4bec792f073d048cf0abd2f280ac4ac3_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 10](https://pic3.zhimg.com/v2-539d566a2c1231046bf2e4631d870bbc_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 11](https://pic2.zhimg.com/v2-e31b0b972b73b1561b9b098805c64195_1440w.jpg)

<p class="kb-image-caption">图例</p>
### [Jensen&#39;s Inequality](https://zhida.zhihu.com/search?content_id=107570581&content_type=Article&match_order=1&q=Jensen%26%2339%3Bs+Inequality&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NjM3MjUsInEiOiJKZW5zZW5cdTAwMjYjMzk7cyBJbmVxdWFsaXR5IiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTA3NTcwNTgxLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.VVretQmFpG85i3_wsXif4IKa-z7OK6n0UeqTzSFZHlQ&zhida_source=entity)
f(E(x))\leq{E(f(x))}

f(x) 的期望难以计算，但是 E(x) 只是一个数值， f(E(x)) 容易计算，为 E(f(x)) 提供了一个下界。

## 凸优化
### 凸优化问题的定义
![Image 12](https://pica.zhimg.com/v2-a7fc209c5fe9567a3acaf274a9718172_1440w.jpg)

<p class="kb-image-caption">图例</p>

将[Geometric Programming](https://zhida.zhihu.com/search?content_id=107570581&content_type=Article&match_order=1&q=Geometric+Programming&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NjM3MjUsInEiOiJHZW9tZXRyaWMgUHJvZ3JhbW1pbmciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMDc1NzA1ODEsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.ujN3n38as2z9TwVIzlIK5K8ARkzynFKUE7dsixDw-K8&zhida_source=entity)转化为Convex Programming
![Image 13](https://pic3.zhimg.com/v2-5dac7afdf7237cb3960842eeb1975230_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 14](https://pic3.zhimg.com/v2-a4f6f351fd9f6b4550d1c9e1333c97ac_1440w.jpg)

<p class="kb-image-caption">图例</p>

[Bisection Method](https://zhida.zhihu.com/search?content_id=107570581&content_type=Article&match_order=1&q=Bisection+Method&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NjM3MjUsInEiOiJCaXNlY3Rpb24gTWV0aG9kIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTA3NTcwNTgxLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Vz-C4C7Ok8fArRy0BCvnc_ZduMautBp4SCu8ykdh_2A&zhida_source=entity)：
用二分法寻找 \frac{df(x)}{dx}=0 的点。

[Newton&#39;s Method](https://zhida.zhihu.com/search?content_id=107570581&content_type=Article&match_order=1&q=Newton%26%2339%3Bs+Method&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODI3NjM3MjUsInEiOiJOZXd0b25cdTAwMjYjMzk7cyBNZXRob2QiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMDc1NzA1ODEsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.OAV6G14EzmflXBjuFhFlU80JCyvOYM1AgWd-G5rQR8k&zhida_source=entity)：
每次迭代执行 x_{i+1}=x_{i}-\frac{f'(x_i)}{f''(x_i)} 。例题如下：
![Image 15](https://pic4.zhimg.com/v2-0aad4c20c2bf1278451699721e8e71ad_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 16](https://pic2.zhimg.com/v2-c147215cff8af71238f47ec408ee971f_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 17](https://picx.zhimg.com/v2-4e8c84b5f9e03f4f6de6c5a7bb217b19_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 18](https://pica.zhimg.com/v2-5ac3e467dedb149b56d73e40508f6d7e_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 19](https://picx.zhimg.com/v2-9ef8fa98ca47fba79cf3d4a863b2eba1_1440w.jpg)

<p class="kb-image-caption">图例</p>
![Image 20](https://pic3.zhimg.com/v2-897d6ddc95deadfa2afb21c9515f12b0_1440w.jpg)

<p class="kb-image-caption">图例</p>
