import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载配置
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

def build_vector_db():
    print(f"--- 正在读取数据: {DATA_PATH} ---")
    
    # 2. 使用 JSONLoader 加载结构化数据
    # .[] 表示遍历 JSON 数组，extract 'answer' as content
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema='.[]',
        content_key="answer"
    )
    documents = loader.load()
    print(f"成功加载 {len(documents)} 条面试知识点")

    # 3. 初始化 Embedding 模型 (文字转向量的魔术棒)
    print(f"--- 正在加载 Embedding 模型: {EMBED_MODEL} ---")
    # 第一次运行会下载模型（约 80MB），请保持网络通畅
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 4. 创建并持久化向量数据库
    print(f"--- 正在生成向量库并存入: {VECTOR_DB_PATH} ---")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    print("✅ 向量库构建完成！你现在可以进行本地检索了。")

if __name__ == "__main__":
    build_vector_db()