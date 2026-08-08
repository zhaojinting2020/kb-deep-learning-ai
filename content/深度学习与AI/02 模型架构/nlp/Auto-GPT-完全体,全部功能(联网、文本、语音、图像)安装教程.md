---
title: Auto-GPT 完全体, 全部功能(联网, 文本, 语音, 图像)安装教程
url: https://zhuanlan.zhihu.com/p/622946647
fetch_source: browser_cookies
fetched_at: '2026-06-27T20:07:27+00:00'
polished_at: '2026-06-28T08:10:00+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Auto-GPT 完全体, 全部功能(联网, 文本, 语音, 图像)安装教程

Auto-GPT 是一个能自主迭代, 长时记忆, 自我提示且联网查询的 GPT 框架。最少只需在第一轮对话中输入需求，Auto-GPT 就能自己分解任务去完成。本教程面向零基础用户：可只做必要步骤获得主要功能，也可按全文配置全部能力。首发公众号：prompthere

| 前排提示 | 需要至少有 OpenAI 账号；以 Windows 为例 |
| --- | --- |

## Git 与 Python 环境

若已安装 Python，可跳过本节。

### Git 安装（非必须）

1. 访问 Git 官方网站：<https://git-scm.com/download/win>
2. 选择 **64-bit Git for Windows Setup** 下载（32 位系统选上一项）
3. 运行安装包，按向导安装（默认可）
4. 安装完成后，Win+R 输入 `cmd` 打开命令提示符，执行 `git --version`；若显示版本号则成功

### Anaconda 安装（非必须，与下一步二选一）

1. 访问 <https://www.anaconda.com/>，点击 Download
2. 运行安装包，按向导安装
3. 若电脑上没有其他 Python，勾选"Add Anaconda to PATH"相关选项
4. 等待安装完成

### Python 安装（非必须，与上一步二选一）

1. 访问 <https://www.python.org/downloads/>，点击 Windows
2. 下载 Python 3.10 及以上（如 3.10.10）
3. 安装时勾选 **Add Python 3.x to PATH**，再点 Install Now（或 Customize Installation）

## Auto-GPT 安装

### 下载项目（Git 方式）

1. 打开 <https://github.com/Significant-Gravitas/Auto-GPT>，点击 Code → 复制 HTTPS 地址
2. 新建文件夹用于安装 Auto-GPT
3. 在文件夹空白处 Shift+右键 →"在此处打开 PowerShell 窗口"
4. 执行：

```bash
git clone https://github.com/Significant-Gravitas/Auto-GPT.git
```

5. 当前目录下应出现 Auto-GPT 源码

### 下载项目（备选：直接下载压缩包）

1. 打开稳定版 <https://github.com/Significant-Gravitas/Auto-GPT/releases>，在 Assets 下载源码压缩包
2. 新建安装目录
3. 解压到该目录

### 安装依赖库

1. 进入 Auto-GPT 源码目录
2. Shift+右键 →"在此处打开 PowerShell 窗口"（或 `cd` 到该目录）
3. 执行：

```bash
pip install -r requirements.txt
```

### 更名与新建

1. 将 `.env.template` 重命名为 `.env`
2. 新建 `auto-gpt.json`（本地记忆缓存；若已存在可忽略）

## API Key 配置

用文本编辑器打开 `.env`。

### OpenAI API Key（主要功能）

1. 打开 <https://platform.openai.com/account/api-keys>
2. 创建 key 并复制
3. 粘贴到 `.env` 中对应项（**不要**加英文双引号）

OpenAI API 按 token 计费；新账号通常有试用额度。

### Pinecone API Key（记忆，非必须）

Pinecone 为向量数据库，用于长时记忆；本地 `auto-gpt.json` 也可缓存，非必须。

1. 打开 <https://app.pinecone.io/> 并注册
2. API keys → 复制 key
3. 写入 `.env`

### Google API Key（联网，非必须）

1. 打开 <https://console.cloud.google.com/>
2. "API 和服务"→"凭据"→ 创建项目 →"创建凭据"→"API 密钥"
3. 打开 <https://programmablesearchengine.google.com/controlpanel/all> →"添加"
4. 填名称，选择"在整个网络中搜索"→ 确定
5. "自定义"→ 复制"搜索引擎 ID"
6. 将 Google key 与搜索引擎 ID 写入 `.env`

### HuggingFace API Key（图像，非必须）

除 OpenAI DALL·E 外，可用 Stable Diffusion（需 HuggingFace token）。

1. <https://huggingface.co/settings/tokens> → Access Tokens → 复制
2. 写入 `.env`

### ElevenLabs API Key（语音，非必须）

1. <https://beta.elevenlabs.io/> → Profile → 复制 key（需先验证）
2. 可选 voice ID 示例：

  - Rachel: `21m00Tcm4TlvDq8ikWAM`
- Domi: `AZnzlk1XvdvUeBnXmlld`
- Bella: `EXAVITQu4vr4xnSDxMaL`
- Antoni: `ErXwobaYiN019PkySvjV`
- Elli: `MF3mGyEYCl7XYWbV9V6O`
- Josh: `TxGEqnHWrfWFTfGW9XjX`
- Arnold: `VR6AewLTigWG4xSOukaG`
- Adam: `pNInz6obpgDQGcFmaJgB`
- Sam: `yoZ06aMxZJJ28mfd3POQ`

3. 将 API key 与 voice ID 写入 `.env`

## 运行 Auto-GPT

保存 `.env` 后：

1. 在 Auto-GPT 目录打开 PowerShell（或 `cd` 到该目录）
2. 执行：

```bash
python -m autogpt
```

需要语音时：

```bash
python -m autogpt --speak
```

3. 首次运行：输入 Agent 名称
4. 输入角色描述（例：浏览网页并总结今日财经信息）
5. 设定目标（可少于 5 个；可用中英文）

运行中每步需确认：`y` 继续，`y -10` 后续 10 步自动确认，`y -N` 直至结束。注意 token 消耗。默认输出目录：`Auto-GPT/autogpt/auto_gpt_workspace`。

### OpenAI APIConnectionError 代理

在 Python 的 `site-packages/openai/api_requestor.py` 中，为请求函数增加 `proxies` 参数，并在所有调用处传入代理（见原知乎教程截图）。

## 相关笔记

[深度学习与AI（主题索引）](MOC-dl-ai.md)
[[LLMs-from-scratch-rasbt|LLMs-from-scratch（rasbt）]]
[[自回归模型_百度百科|自回归模型_百度百科]]
[[自回归模型_百度百科|自回归模型_百度百科]]
