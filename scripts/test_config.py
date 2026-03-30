import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# 1. 加载 .env 文件
load_dotenv()

# 2. 从环境变量中读取配置
base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("LLM_MODEL")

print(f"--- 正在连接服务: {base_url} ---")
print(f"--- 使用模型: {model_name} ---")

# 3. 初始化模型
try:
    llm = OllamaLLM(base_url=base_url, model=model_name)
    response = llm.invoke("确认配置成功，请回复：配置已就绪！")
    print(f"模型反馈: {response}")
except Exception as e:
    print(f"连接失败，请检查 Ollama 是否已启动。错误信息: {e}")