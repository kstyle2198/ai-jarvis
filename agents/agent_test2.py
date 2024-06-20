from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.llms.ollama import Ollama

load_dotenv()


from llama_index.core.tools import FunctionTool
import os
note_file = os.path.join("data", "notes.txt")
def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")
    with open(note_file, "a") as f:
        f.writelines([note + "\n"])
    return "note saved"
note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)


import os
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index
pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file=pdf_path)
canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()


from llama_index.core.prompts import PromptTemplate
instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)
context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information about world population statistics and details about a country. """

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
    )
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country",
        ),
    ),
]


llm = Ollama(model="gemma:2b", request_timeout=120.0)
response = llm.complete("What is the population of France?")
print(response)

# agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = agent.query(prompt)
#     print(result)