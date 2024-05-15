import streamlit as st
import time
import numpy as np
import requests
import json
from datetime import datetime, timedelta

def calculate_time_delta(start_time, end_time):
    # Calculate the time difference (time delta) in seconds
    time_difference = end_time - start_time
    seconds = time_difference.seconds
    return seconds



sample_template = '''
you are an smart AI assistant in a commercial vessel like LNG Carriers or Container Carriers.
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
    st.session_state.trans = ""
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.rag_reversed_messages = ""

def stream_data(output):
    for word in output.split(" "):
        yield word + " "
        time.sleep(0.1)

@st.experimental_fragment
def chatbot():
    with st.container():
        llm_name = st.radio("🐬 **Select LLM**", options=["moondream(1B)", "tinydolphin(1.1B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3", "Groq_llama3"], index=2, key="dsfv")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        llm_name
    with st.expander("🐳 **Custom Prompt**"):
        custom_template = st.markdown(sample_template)

    if st.button("💬 Call Jarvis"):
        res1 = requests.get(url=f"http://127.0.0.1:8000/jarvis_stt")
        start_time = datetime.now()
        input_voice = res1.json()
        input_voice = input_voice["input_voice"]
        
        st.session_state.messages.append({"role": "user", "content": input_voice})
        if input_voice:
            if llm_name == "tinydolphin(1.1B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_tinydolphin?input_voice={input_voice}")
            elif llm_name == "moondream(1B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_moondream?input_voice={input_voice}")
            elif llm_name == "dolphin-phi(2.7B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_dolphinphi?input_voice={input_voice}")
            elif llm_name == "phi3(3.8B)":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_phi3?input_voice={input_voice}")
            elif llm_name == "llama3":
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_llama3?input_voice={input_voice}")
            else:
                res2 = requests.get(url=f"http://127.0.0.1:8000/call_groq_llama3?input_voice={input_voice}")
            output = res2.json()
            st.session_state.output = output['output']
            end_time = datetime.now()

            trans_res = requests.get(url=f"http://127.0.0.1:8000/call_trans?txt={st.session_state.output}")
            trans_res = trans_res.json()
            st.session_state.trans = trans_res['output'][0]
            
            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ 응답소요시간(초) : {delta}")

            col111, col112 = st.columns(2)
            with col111: st.session_state.output
            with col112: st.session_state.trans

            st.session_state.messages.append({"role": "assistant", "content": st.session_state.output}) 

        if st.session_state.output:
            res = requests.get(url=f"http://127.0.0.1:8000/jarvis_tts?output={st.session_state.output}")

    st.markdown("---")
    st.session_state.reversed_messages = st.session_state.messages[::-1]
    for msg in st.session_state.reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

    

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
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

import os
from langchain_groq import ChatGroq
groq_api_key = os.environ['GROQ_API_KEY']


if "retriever" not in st.session_state:
    st.session_state.retirever = ""

@st.experimental_fragment
def create_vectordb(pdf_text):
    with st.spinner("Processing..."):
        if st.button("Create Vectorstore"):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splitted_texts = text_splitter.split_text(pdf_text)
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            # vectorstore = FAISS.from_texts(splitted_texts, embeddings, distance_strategy=DistanceStrategy.DOT_PRODUCT)
            # st.session_state.retirever = vectorstore.as_retriever(k=4)
            db=Chroma.from_texts(splitted_texts, embedding=embed_model, persist_directory="test_index")
        if st.session_state.retirever: 
            st.session_state.retirever
            st.info("VectorStore Created")

def do_rag():
    with st.container():
        llm_name = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "Groq_llama3"], index=0, key="dsssv")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        llm_name
    with st.spinner("Processing..."):
        if st.button("💬 RAG Jarvis"):

            res1 = requests.get(url=f"http://127.0.0.1:8000/jarvis_stt")
            start_time = datetime.now()
            query = res1.json()
            query = query["input_voice"]
            


            if llm_name == "tinydolphin(1.1B)":
                rag_res = requests.get(url=f"http://127.0.0.1:8000/call_rag_tinydolphin?query={query}")
            else:
                rag_res = requests.get(url=f"http://127.0.0.1:8000/call_rag_groq_llama3?query={query}")
            rag_res = rag_res.json()
            st.session_state.rag_output = rag_res['output']
            end_time = datetime.now()
            delta = calculate_time_delta(start_time, end_time)
            
            trans_res = requests.get(url=f"http://127.0.0.1:8000/call_trans?txt={st.session_state.rag_output}")
            trans_res = trans_res.json()
            st.session_state.trans = trans_res['output'][0]
            st.warning(f"⏱️ 응답소요시간(초) : {delta}")
            col111, col112 = st.columns(2)
            with col111: st.session_state.rag_output
            with col112: st.session_state.trans


            st.session_state.rag_messages.append({"role": "user", "content": query})
            # st.session_state.rag_docs.append(docs)
            st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output}) 

            if st.session_state.rag_output:
                res = requests.get(url=f"http://127.0.0.1:8000/jarvis_tts?output={st.session_state.rag_output}")
        
        st.markdown("---")
        st.session_state.rag_reversed_messages = st.session_state.rag_messages[::-1]
        for msg in st.session_state.rag_reversed_messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

from pathlib import Path
parent_dir = Path(__file__).parent
base_dir = str(parent_dir) + "\data"  

import os
import PyPDF2
def list_selected_files(path, 확장자):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list if file.endswith(확장자)]
        return selected_files

###################################################################################################################################################3
if __name__ == "__main__":
    st.title("⚓ :blue[AI Jarvis]")
    st.markdown("---")

    tab1, tab2 = st.tabs(["**Chatbot**", "**RAG**"])
    with tab1:
        chatbot()

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
            try:
                file_list2 = list_selected_files(base_dir, "pdf")
                sel21 = st.selectbox("📌 파일을 선택해주세요", file_list2, index=None)
                st.session_state.path = os.path.join(base_dir, sel21)
                st.session_state.path
            except: pass
            if st.session_state.path: 
                pdf = PyPDF2.PdfReader(st.session_state.path)
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text()
                create_vectordb(pdf_text)
                with st.container(height=300):
                    pdf_text
        try:            
            do_rag()

        except:
            st.empty()

