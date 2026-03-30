import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 加载配置
load_dotenv()
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def start_chat():
    # 2. 加载 Embedding 和 向量库 (只读模式)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    # 3. 初始化 Qwen:4b
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)

    # 4. 定制“面试官”提示词模板 (这是 Agent 的灵魂)
    template = """你是一位专业的 AI Agent 开发面试官。请根据以下提供的【已知信息】来回答候选人的问题。
    如果已知信息中没有相关内容，请诚实回答“抱歉，我的知识库中暂未收录此面试点”，不要胡编乱造。
    
    【已知信息】：{context}
    【候选人提问】：{question}
    
    请用专业、严谨且富有逻辑的语言回答："""
    
    QA_PROMPT = PromptTemplate.from_template(template)

    # 构建 LCEL 链
    rag_chain = (
        {"context": vector_db.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    print(f"\n🚀 欢迎来到 Agent 面试助手 (模型: {LLM_MODEL})")
    print("输入 'exit' 或 'quit' 退出对话\n")

    while True:
        query = input("🤔 你的面试疑问: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        print("\n🔍 正在检索知识库并思考...\n")
        response = rag_chain.invoke(query)
        
        print(f"🤖 面试官回答：\n{response}\n")
        print("-" * 50)

if __name__ == "__main__":
    start_chat()