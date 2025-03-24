## 0x00 学习路径

本文分为三个章节，各章节的学习目标如下。

-   入门篇：
    -   了解大语言模型的基础知识和常见术语。
    -   学会使用编程语言访问 OpenAI API 等常见大语言模型接口。
    -   面向非专业背景的大模型普及知识。
-   应用篇：
    -   可以在本地环境搭建开源模型的推理环境。
    -   大语言模型应用开发框架（如 LangChain、Dify等）。
    -   Prompt 工程、 RAG、Agent 等大模型应用开发范式。
-   深入篇：
    -   大模型技术原理、训练微调、数据工程、推理优化等。
    -   大模型应用范式（RAG、Agent等）前沿进展。

读者可以根据自己需要选择对应的章节，如对大语言模型的原理不感兴趣，可只关注入门篇和应用篇。  
考虑到阅读背景，本文尽可能提供中文资料或有中文翻译的资料。

标记为【必看】的是我认为只要你对这个主题感兴趣，必须要看的资料。

## 0x10 入门篇

> 在入门之前，请申请 OpenAI API，并具备良好的国际互联网访问条件。  
> 推荐注册  [https://openrouter.ai/](https://openrouter.ai/)  可一站式访问大量闭源和开源模型。

-   [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
    -   虽然是 Prompt 工程，但是内容比较简单，适合入门者。
    -   中英双语字幕：  [https://github.com/GitHubDaily/ChatGPT-Prompt-Engineering-for-Developers-in-Chinese](https://github.com/GitHubDaily/ChatGPT-Prompt-Engineering-for-Developers-in-Chinese)
-   [OpenAI Quickstart](https://platform.openai.com/docs/quickstart)  【必看】
    -   OpenAI 官方 Quickstart 文档。以及  [API Reference](https://platform.openai.com/docs/api-reference)
-   State of GPT：Andrej Karpathy 做的演示，极好的总结了 GPT 的训练和应用。 【必看】
    -   视频：  [https://www.youtube.com/watch?v=bZQun8Y4L2A](https://www.youtube.com/watch?v=bZQun8Y4L2A)
    -   PPT：  [https://karpathy.ai/stateofgpt.pdf](https://karpathy.ai/stateofgpt.pdf)
-   Deep Dive into LLMs like ChatGPT: Andrej Karpathy 最新的长达3小时的入门视频【必看】
    -   视频：[https://www.youtube.com/watch?v=7xTGNNLPyMI](https://www.youtube.com/watch?v=7xTGNNLPyMI)
    -   中英双语字幕：[https://b23.tv/vF2vS6t](https://b23.tv/vF2vS6t)

## 0x20 应用篇

-   [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system/lesson/1/introduction)
    -   中文字幕：  [https://www.bilibili.com/video/BV1gj411X72B/](https://www.bilibili.com/video/BV1gj411X72B/)
-   [Langchain](https://python.langchain.com/)
    -   Langchain 是大语言模型最火的应用框架。即使不使用，也可以借鉴。
    -   [LangChain for LLM Application Development](https://learn.deeplearning.ai/langchain/lesson/1/introduction)
        -   中文字幕：  [https://www.bilibili.com/video/BV1Ku411x78m/](https://www.bilibili.com/video/BV1Ku411x78m/)
-   [dify](https://dify.ai/)：开源的应用编排工具。
-   [GPT best practices](https://platform.openai.com/docs/guides/gpt-best-practices/gpt-best-practices)：OpenAI 官方出的最佳实践。
-   [openai-cookbook](https://github.com/openai/openai-cookbook)：OpenAI 官方 Cookbook。
-   [Brex's Prompt Engineering Guide](https://github.com/brexhq/prompt-engineering)：Prompt 工程简介

## 0x30 深入篇

### 0x31 大模型技术基础方向

-   [《动手学深度学习》](https://zh.d2l.ai/)：配合[B站李沐的视频](https://courses.d2l.ai/zh-v2/)，是我个人认为最好的深度学习入门课程。【必看】
-   [深度学习：台湾大学李宏毅](https://www.bilibili.com/video/BV1J94y1f7u5/)：台湾大学李宏毅，讲的很清楚，也比较有趣。
-   [3brown1blue 系列视频](https://www.youtube.com/watch?v=wjZofJX0v4M)：动画做的很好，可反复回顾 【必看】

### 0x32 大模型技术原理方向

-   [大语言模型综述](https://github.com/RUCAIBox/LLMSurvey)【必看】
    -   大语言模型迄今为止最好的学术向中文综述。
-   [大语言模型](https://github.com/LLMBook-zh/LLMBook-zh.github.io)【必看】
    -   大语言模型迄今为止最好的书籍。
-   [大规模语言模型：从理论到实践](https://intro-llm.github.io/)：另一本不错的中文书籍。
-   [清华大模型公开课第二季](https://www.bilibili.com/video/BV1pf421z757)：系统的了解大模型的历史、原理和前沿进展。【必看】
-   [GPT，GPT-2，GPT-3 论文精读](https://www.bilibili.com/video/BV1AF411b7xQ)：GPT 系列模型论文精读
-   [Llama3.1 论文精读](https://www.bilibili.com/video/BV1WM4m1y7Uh)：最好的开源大模型论文精读
-   [复杂推理：大语言模型的北极星能力](https://yaofu.notion.site/6dafe3f8d11445ca9dcf8a2ca1c5b199)  ：略学术，解释大语言模型能力的来源。
-   [ICML 2024 Tutorial: Physics of Language Models by Zeyuan Allen-Zhu](https://www.bilibili.com/video/BV1TPpbeVEUi/)：使用黑盒研究大模型的原理，非常有参考价值。【必看】

### 0x33 大模型训练微调方向

-   [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)：从零构建大模型。【必看】
-   [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：个人最推荐的微调大模型的工具。【必看】
-   [MAP-NEO](https://github.com/multimodal-art-projection/MAP-NEO)：唯一全过程开源的中文大模型（包括数据处理工具、预训练数据、微调数据等）
-   [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：大规模集群训练大模型的经验，前面部分对模型训练的显存占用、4D并行做了很详细的说明。[中文翻译](https://huggingface.co/spaces/Ki-Seki/ultrascale-playbook-zh-cn)。【必看】

### 0x34 大模型数据工程方向

-   [How to Generate and Use Synthetic Data for Finetuning](https://eugeneyan.com/writing/synthetic/)：如何合成微调数据。
-   [中文行业预训练语料 IndustryCorpus 2.0](https://data.baai.ac.cn/details/BAAI-IndustryCorpus-v2)：亮点是预训练数据处理流比较科学。[数据处理工具 FlagData](https://github.com/FlagOpen/FlagData/blob/main/README_zh.md)

### 0x35 大模型推理优化方向

-   [Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis](https://arxiv.org/abs/2405.08944)：大模型推理速度计算和瓶颈分析。【必看】
-   [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)：大模型量化解析。

### 0x36 大模型应用方向

-   [A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks](https://arxiv.org/abs/2407.12994): Prompt 工程综述
-   [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/pdf/2407.21059)：高级 RAG 优化方法。
-   [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)：Agent 早期的很不错的文章。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNzU3Njc3MzhdfQ==
-->