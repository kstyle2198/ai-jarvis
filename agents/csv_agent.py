from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI

import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "agent-llama3",
    base_url = "http://localhost:11434/v1")
print(llm)

csv_path = "D:/ai_jarvis/data/sensor_data.csv"

agent = create_csv_agent(
    llm,
    csv_path,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run("whats the temperature on March 9th")




# agent = create_csv_agent(
#     ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#     ["titanic.csv", "titanic_age_fillna.csv"],
#     verbose=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
# )
# agent.run("how many rows in the age column are different between the two dfs?")
