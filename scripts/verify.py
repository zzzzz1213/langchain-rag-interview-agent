import sys
print(f"Python Executable: {sys.executable}")

try:
    # 核心：不再从 langchain.agents 导入，直接去底层找
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.react.agent import create_react_agent
    from langchain_openai import ChatOpenAI
    import langchain
    
    print(f"LangChain 版本: {langchain.__version__}")
    print("✅ 【成功】跳过路径坑，导入成功！")
except Exception as e:
    print(f"❌ 依然报错: {e}")