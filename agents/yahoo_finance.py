from langchain.agents import AgentType, initialize_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.chat_models import ChatOllama


llm = ChatOllama(model="phi3:latest")
tools = [YahooFinanceNewsTool()]
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.invoke(
    "tell me short summary of news about MSFT during last week",
)

