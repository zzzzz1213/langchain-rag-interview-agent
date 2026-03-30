import os
import sys
from dotenv import load_dotenv

# --- 1. 核心组件导入 (确保路径在 LangChain 1.x 中兼容) ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LCEL 运行组件
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 记忆组件
from langchain_community.chat_message_histories import ChatMessageHistory

# 备选导入路径处理 (针对不同版本的 create_retriever_tool)
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    try:
        from langchain_community.agent_toolkits import create_retriever_tool
    except ImportError:
        pass

# 加载环境变量
load_dotenv()

# --- 2. 基础模型与数据库初始化 ---

print("--- 正在验证核心库与模型 ---")
try:
    # 初始化 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0
    )
    
    # 连接 Chroma 向量库
    vector_db = Chroma(
        persist_directory=os.getenv("VECTOR_DB_PATH"), 
        embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    print("✅ 模型与向量数据库加载成功")
except Exception as e:
    print(f"❌ 初始化失败，请检查 .env 配置: {e}")
    sys.exit()


# --- 3. 定义逻辑组件 (必须在构建链条之前) ---

# 定义 Prompt 模板：加入历史记录占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的 AI 面试官。请结合【背景信息】和【对话历史】来回答用户的问题。如果背景信息中没有相关内容，请告知用户。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "--- 背景信息 ---\n{context}"),
    ("human", "{question}")
])

# 文档格式化函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- 4. 构建带记忆的 RAG 链 (LCEL 语法) ---

# 这里的逻辑是：
# 1. 从输入字典中提取 question 喂给 retriever 获得 context
# 2. 同时保留原始的 question 和 chat_history 传给 prompt
rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs, 
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --- 5. 执行对话循环 ---

if __name__ == "__main__":
    print("\n" + "="*30)
    print("🤖 进阶版（带记忆）面试助手已就绪")
    print("="*30)
    
    # 创建内存中的历史记录对象
    history = ChatMessageHistory()
    
    # 模拟多轮测试问题
    test_questions = [
        "什么是 Agent 的 Planning 能力？",
        "那它具体包含哪些技术路径？" # 这是一个追问，测试记忆能力
    ]

    for q in test_questions:
        print(f"\n👤 用户问: {q}")
        print("⏳ 正在思考并检索知识库...")
        
        try:
            # 运行链条
            res = rag_chain.invoke({
                "question": q,
                "chat_history": history.messages 
            })
            
            print(f"🤖 面试官: {res}")
            
            # --- 核心：手动更新历史记录 ---
            history.add_user_message(q)
            history.add_ai_message(res)
            
        except Exception as e:
            print(f"❌ 运行报错: {e}")

    print("\n" + "="*30)
    print("✅ 对话测试结束")