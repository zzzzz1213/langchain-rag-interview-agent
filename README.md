# LangChain 1.x RAG Interview Agent

本项目是一个基于 LangChain 1.x 构建的 AI 面试助手，支持本地知识库检索（RAG）与多轮对话记忆。

## 🌟 核心特性

- **最新框架**：全面适配 LangChain 1.x 与 LCEL 语法。
- **持久化检索**：基于 Chroma 向量数据库，实现本地专业知识精准匹配。
- **对话记忆**：集成 `ChatMessageHistory`，支持上下文语义追问。
- **模型适配**：通过 OpenAI 协议兼容 Ollama (Qwen) 等本地大模型。

## 🛠️ 环境要求

- Python 3.10+
- 依赖安装：`pip install -r requirements.txt`

## 🚀 快速启动

1. 配置 `.env` 文件中的 API Key 和模型路径。
2. 运行 `python scripts/04_agent_tools.py`。
