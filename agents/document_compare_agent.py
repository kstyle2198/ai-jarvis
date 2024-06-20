from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic.v1 import BaseModel, Field

class DocumentInput(BaseModel):
    question: str = Field()

import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "agent-llama3",
    base_url = "http://localhost:11434/v1")
print(llm)

tools = []
files = [
    {
        "name": "specification for purchase of FW Generator",
        "path": "D:/ai_jarvis/data/FWG.pdf",
    },
    {
        "name": "specification for purchase of Unit Cooler",
        "path": "D:/ai_jarvis/data/Unit_Cooler.pdf",
    },
]

from langchain_community.embeddings import OllamaEmbeddings

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )


print(tools)
print("중간완료 ----------------------------------------------------------------")

from langchain.agents import AgentType, initialize_agent
from langchain.globals import set_debug
from langchain.agents import AgentExecutor
set_debug(True)



agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)
agent.run({"input": "what is the main difference between FW generator and unit cooler?"})
print("최종 완료---------------------------------------------------------------------")