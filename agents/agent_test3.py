
### [Start]Load LLM ##############################
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "agent-llama3",
    base_url = "http://localhost:11434/v1",
    streaming=True)
print(llm)
### [End]Load LLM ##############################

### [Start] Define Tools ##############################
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

print(get_word_length.invoke("I love you"))
tools = [get_word_length]
### [End] Define Tools ##############################


### [Start] Create Prompts ##############################
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
print(prompt)
### [End] Create Prompts ##############################

### [Start] Bind tools to LLM ##############################
llm_with_tools = llm.bind_tools(tools)
print(llm_with_tools)
### [End] Bind tools to LLM ##############################

### [Start] Create the Agent ##############################
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
print(agent)
### [End] Create the Agent ##############################


### [Start] Create AgentExecutor ##############################
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = list(agent_executor.stream({"input": "How many letters in the word eudca"}))
print(result)
### [End] Create AgentExecutor ##############################


# ### [Start] Adding memory ##############################
# from langchain_core.prompts import MessagesPlaceholder

# MEMORY_KEY = "chat_history"
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are very powerful assistant, but bad at calculating lengths of words.",
#         ),
#         MessagesPlaceholder(variable_name=MEMORY_KEY),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )
# from langchain_core.messages import AIMessage, HumanMessage

# chat_history = []
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#         "chat_history": lambda x: x["chat_history"],
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIToolsAgentOutputParser()
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# input1 = "how many letters in the word educa?"
# result = agent_executor.invoke({"input": input1, "chat_history": chat_history})

# chat_history.extend(
#     [
#         HumanMessage(content=input1),
#         AIMessage(content=result["output"]),
#     ]
# )

# result = agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})
# print(result)
# ### [End] Adding memory ##############################



### [Start] Create Prompts ##############################
### [End] Create Prompts ##############################