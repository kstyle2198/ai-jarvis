import streamlit as st
import time
import numpy as np
import requests
import json


sample_template = '''
you are an smart AI assistant in a commercial vessel like LNG Carriers.
your answer always starts with "OK, Master".
generate compact and summarized answer to {query} kindly and shortly.
if there are not enough information to generate answers, just return "Please give me more information"
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?'
'''

st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.output = ""
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.reversed_messages = ""

if "path" not in st.session_state:
    st.session_state.path = ""
    st.session_state.rag_results = []
    st.session_state.rag_docs = []
    st.session_state.rag_output = ""
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.rag_reversed_messages = ""


def stream_data(output):
    for word in output.split(" "):
        yield word + " "
        time.sleep(0.1)

@st.experimental_fragment
def select_llm():
    llm_name = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "Groq_llama3"], index=1)
    return llm_name


@st.experimental_fragment
def chatbot(llm_name):
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    with st.expander("🐳 **Custom Prompt**"):
        custom_template = st.markdown(sample_template)

    if st.button("Call Jarvis - Step by Step"):

        res1 = requests.get(url=f"http://127.0.0.1:8000/jarvis_stt")
        input_voice = res1.json()
        input_voice = input_voice["input_voice"]
        st.session_state.messages.append({"role": "user", "content": input_voice})
        if input_voice:
            if llm_name == "tinydolphin(1.1B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_tinydolphin?input_voice={input_voice}")
            elif llm_name == "moondream":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_moondream?input_voice={input_voice}")
            elif llm_name == "dolphin-phi(2.7B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_dolphinphi?input_voice={input_voice}")
            elif llm_name == "phi3(3.8B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_phi3?input_voice={input_voice}")
            else:
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_groq_llama3?input_voice={input_voice}")
            output = res2.json()
            st.session_state.output = output['output']
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.output}) 
            # st.write_stream(stream_data(output))

    st.markdown("---")
    st.session_state.reversed_messages = st.session_state.messages[::-1]
    for msg in st.session_state.reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

    if st.session_state.output:
        res = requests.get(url=f"http://127.0.0.1:8000/jarvis_tts?output={st.session_state.output}")


from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama

@st.experimental_fragment
def my_rag(pdf_text):
    with st.spinner("Processing..."):
        if st.button("💬 RAG Jarvis"):

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(pdf_text)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.from_texts(texts, embeddings, distance_strategy=DistanceStrategy.DOT_PRODUCT)
            retriever = vectorstore.as_retriever(k=4)
            st.info("VectorStore Created")

            res1 = requests.get(url=f"http://127.0.0.1:8000/jarvis_stt")
            query = res1.json()
            query = query["input_voice"]

            docs = retriever.invoke(query)
            with st.expander("Retrieved Documnets"):
                docs

            # llm = ChatOllama(model="tinydolphin:latest")  #1.1B
            # llm = ChatOllama(model="dolphin-phi:latest")  #2.7B
            llm = ChatOllama(model="phi3:latest")         #3.8
            # llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

            SYSTEM_TEMPLATE = """
                            Answer the user's questions based on the below context. 
                            If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

                            <context>
                            {context}
                            </context>
                            """
            question_answering_prompt = ChatPromptTemplate.from_messages(
                [("system",
                    SYSTEM_TEMPLATE,),
                    MessagesPlaceholder(variable_name="messages"),
                    ])
            document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
            result = document_chain.invoke(
                    {
                        "context": docs,
                        "messages": [
                            HumanMessage(content=query)
                        ],
                    }
                )

            st.session_state.rag_output = result
            st.session_state.rag_messages.append({"role": "user", "content": query})
            st.session_state.rag_docs.append(docs)
            st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output}) 
    st.markdown("---")
    st.session_state.rag_reversed_messages = st.session_state.rag_messages[::-1]
    for msg in st.session_state.rag_reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

    if st.session_state.rag_output:
        res = requests.get(url=f"http://127.0.0.1:8000/jarvis_tts?output={st.session_state.rag_output}")
    





from pathlib import Path
parent_dir = Path(__file__).parent
base_dir = str(parent_dir) + "\data"  

import os
import PyPDF2
def list_selected_files(path, 확장자):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list if file.endswith(확장자)]
        return selected_files



    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


if __name__ == "__main__":
    st.title("⚓ :blue[AI Jarvis]")
    st.markdown("---")

    tab1, tab2 = st.tabs(["**Chatbot**", "**RAG**"])
    with tab1:
        with st.container():
            llm_name = select_llm()
            chatbot(llm_name)

    with tab2:
        with st.expander("📑 File Uploader"):
            uploaded_file = st.file_uploader("📎Upload your file")
            if uploaded_file:
                temp_dir = base_dir   # tempfile.mkdtemp()  --->  import tempfile 필요, 임시저장디렉토리 자동지정함
                path = os.path.join(temp_dir, uploaded_file.name)
                with open(path, "wb") as f:
                        f.write(uploaded_file.getvalue())
            
            if st.button("Save", type='secondary'):
                st.markdown(f"path: {path}")
                st.info("Saving a file is completed")
            else:
                st.empty()

        file_list2 = list_selected_files(base_dir, "pdf")
        sel21 = st.selectbox("📌 파일을 선택해주세요", file_list2, index=None)
        if sel21:
            st.session_state.path = os.path.join(base_dir, sel21)
        
        with st.expander("✏️ PDF TEXT"):
            if st.session_state.path: 
                pdf = PyPDF2.PdfReader(st.session_state.path)
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text()
                with st.container(height=300):
                    pdf_text

        try:
            my_rag(pdf_text)

        except:
            st.empty()

