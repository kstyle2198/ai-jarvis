import asyncio
import aiohttp
import streamlit as st
from datetime import datetime
import time
from utils import CustomPDFLoader, ShowPdf

# st.set_page_config(
#         page_title="AI Jarvis",
#         layout="wide")
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

from pathlib import Path
parent_dir = Path(__file__).parent
base_dir = str(parent_dir) + "\data"  

import os
import PyPDF2
def list_selected_files(path, 확장자):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list if file.endswith(확장자)]
        return selected_files


#### 공통함수 ############
def stream_data(output):
    for word in output.split(" "):
        yield word + " "
        time.sleep(0.1)

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

async def stt():
    url = "http://127.0.0.1:8000/jarvis_stt"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"

async def tts(output):
    url = "http://127.0.0.1:8000/jarvis_tts"
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"output": output})
    except Exception as e:
        return f"Error: {str(e)}"

async def trans(txt):
    url = "http://127.0.0.1:8000/jarvis_trans"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"txt": txt}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"


#### Chatbot 함수 #############################
if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.output = ""
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.reversed_messages = ""

async def api_ollama(url, llm_name, input_voice):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"llm_name": llm_name, "input_voice": input_voice}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"

async def api_groq(url, input_voice):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"input_voice": input_voice}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"
    
async def call_jarvis(llm_name, input_voice):
    async with aiohttp.ClientSession() as session:
        if llm_name == "Groq_llama3":
            url = "http://127.0.0.1:8000/call_groq_llama3"  
        else:
            url = "http://127.0.0.1:8000/call_jarvis" 

        async with session.post(url, json={"llm_name": llm_name, "input_voice": input_voice}) as response:
            res = await response.json()

    output = res["output"]
    trans_res = await trans(output)
    trans_output = trans_res['output'][0]

    return output, trans_output
    
async def chat_main():
    with st.expander("🐳 **Custom Prompt**"):
        custom_template = st.markdown(sample_template)
    with st.container():
        llm1 = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3", "Groq_llama3"], index=1, key="dsfv", help="Internet is not needed, except for 'Groq_llama3'")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if llm1 == "tinydolphin(1.1B)":
            llm_name = "tinydolphin:latest"
        elif llm1 == "dolphin-phi(2.7B)":
            llm_name = "dolphin-phi:latest"
        elif llm1 == "phi3(3.8B)":
            llm_name = "phi3:latest"
        elif llm1 == "llama3":
            llm_name = "llama3:latest"
        else:
            llm_name = "Groq_llama3"

    text_input = st.text_input("✏️ Send your Qeustions", placeholder="Input your Qeustions", key="dldfs")
    call_btn = st.button("💬 Call Jarvis", help="Without Text Query, Click & Say 'Jarvis' after 2~3 seconds. Jarvis will replay 'Yes, Master' and then Speak your Requests")
    if  call_btn and text_input =="":
        res = await stt()
        input_voice = res['input_voice']
        start_time = datetime.now()
        st.session_state.messages.append({"role": "user", "content": input_voice})

        if input_voice:
            output, trans_output = await call_jarvis(llm_name, input_voice)
            st.session_state.output = output
            st.session_state.trans = trans_output
            end_time = datetime.now()
            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ TimeDelta(Sec) : {delta}")
            col111, col112 = st.columns(2)
            with col111: st.write(st.session_state.output)
            with col112: st.write(st.session_state.trans)
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.output})
            if st.session_state.output:
                await tts(st.session_state.output)

    elif call_btn and text_input:
        start_time = datetime.now()
        input_voice = text_input
        st.session_state.messages.append({"role": "user", "content": input_voice})
        if input_voice:
            start_time = datetime.now()
            output, trans_output = await call_jarvis(llm_name, input_voice)
            st.session_state.output = output
            st.session_state.trans = trans_output
            end_time = datetime.now()

            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

            col111, col112 = st.columns(2)
            with col111: st.write_stream(stream_data(st.session_state.output))
            with col112: st.write(st.session_state.trans)

            st.session_state.messages.append({"role": "assistant", "content": st.session_state.output})
            if st.session_state.output:
                await tts(st.session_state.output)
            text_input = ""

    st.markdown("---")
    st.session_state.reversed_messages = st.session_state.messages[::-1]
    for msg in st.session_state.reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])


#### VectorDB 함수 #################################################    
if "retriever" not in st.session_state:
    st.session_state.retirever = ""

if "path" not in st.session_state:
    st.session_state.path = ""
    st.session_state.pages = ""
    st.session_state.retrievals = ""
    st.session_state.rag_results = []
    st.session_state.rag_doc = ""
    st.session_state.rag_output = ""
    st.session_state.trans = ""
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.rag_reversed_messages = ""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

show_pdf = ShowPdf()
custom_loader = CustomPDFLoader()



def create_vectordb(parsed_text):  # VectorDB생성 및 저장
    with st.spinner("Processing..."):
        if st.button("Create Vectorstore", help="You can add your PDFs in the VectorStore After Parsing"):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splitted_texts = text_splitter.split_documents(parsed_text)
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            db=Chroma.from_documents(splitted_texts, embedding=embed_model, persist_directory="test_index")
        if st.session_state.retirever: 
            st.session_state.retirever
            st.info("VectorStore Created")

#### RAG 함수 #################################################    

async def call_rag(llm_name, query):
    try:
        if llm_name == "Groq_llama3":
            url = "http://127.0.0.1:8000/call_rag_groq_llama3"
            res = await api_groq(url, query)
        else:
            url = "http://127.0.0.1:8000/call_rag_jarvis"
            res = await api_ollama(url, llm_name, query)
        
        retrival_output = res["output"][0]
        output = res["output"][1]
        trans_res = await trans(output)
        trans_output = trans_res['output'][0]
        
        return retrival_output, output, trans_output
    except Exception as e:
        return f"Error: {str(e)}"

    
async def rag_main():
    with st.container():
        llm2 = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3", "Groq_llama3"], index=1, key="dsssv", help="Internet is not needed, except for 'Groq_llama3'")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    with st.container():
        if llm2 == "tinydolphin(1.1B)":
            llm_name = "tinydolphin:latest"
        elif llm2 == "dolphin-phi(2.7B)":
            llm_name = "dolphin-phi:latest"
        elif llm2 == "phi3(3.8B)":
            llm_name = "phi3:latest"
        elif llm2 == "llama3":
            llm_name = "llama3:latest"
        else:
            llm_name = "Groq_llama3"

    text_input = st.text_input("✏️ Send your Queries", placeholder="Input your Query", key="dls")
    rag_btn = st.button("💬 RAG Jarvis", help="Without Text Query, Click & Say 'Jarvis' after 2~3 seconds. Jarvis will replay 'Yes, Master' and then Speak your Requests")

    if  rag_btn and text_input == "":
        res = await stt()
        query = res['input_voice']
        start_time = datetime.now()
        st.session_state.rag_messages.append({"role": "user", "content": query})

        if query:
            retrival_output, output, trans_output = await call_rag(llm_name, query)
            st.session_state.rag_doc = retrival_output
            st.session_state.rag_output = output
            st.session_state.trans = trans_output
            end_time = datetime.now()
            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

            col111, col112 = st.columns(2)
            with col111: st.write(st.session_state.rag_output)
            with col112: st.write(st.session_state.trans)
            with st.expander("Retrieval doc"):
                st.session_state.rag_doc

            st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})

            if st.session_state.rag_output:
                await tts(st.session_state.rag_output)

    elif rag_btn and text_input:
        start_time = datetime.now()
        query = text_input
        st.session_state.rag_messages.append({"role": "user", "content": query})

        retrival_output, output, trans_output = await call_rag(llm_name, query)
        st.session_state.rag_doc = retrival_output
        st.session_state.rag_output = output
        st.session_state.trans = trans_output
        end_time = datetime.now()

        delta = calculate_time_delta(start_time, end_time)
        st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

        col111, col112 = st.columns(2)
        with col111: st.write(st.session_state.rag_output)
        with col112: st.write(st.session_state.trans)
        with st.expander("Retrieval doc"):
            st.session_state.rag_doc

        st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})

        if st.session_state.rag_output:
            await tts(st.session_state.rag_output)

    st.markdown("---")
    st.session_state.rag_reversed_messages = st.session_state.rag_messages[::-1]
    for msg in st.session_state.rag_reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])


sample_template = '''
you are an smart AI assistant in a commercial vessel like LNG Carriers or Container Carriers.
your answer always starts with "OK, Master".
generate compact and summarized answer to {query} kindly and shortly.
if there are not enough information to generate answers, just return "Please give me more information"
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?'
'''





if __name__ == "__main__":
    st.title("⚓ :blue[AI Jarvis]")
    with st.expander("🚢 Note"):
        st.markdown("""
                    - This AI app is created for :green[**Local Usages without Internet**].
                    - :orange[**Chatbot**] is for Common Conversations regarding your interests like food, movie, etc.
                    - :orange[**RAG**] is for ***Domain-Specific Conversations*** using VectorStore(saving your PDFs)
                    """)
    tab1, tab2 = st.tabs(["⚾ **Chatbot**", "⚽ **RAG**"])
    with tab1:
        asyncio.run(chat_main())
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

        with st.expander("✏️ Custom Parsing"):
            try:
                file_list2 = list_selected_files(base_dir, "pdf")
                sel21 = st.selectbox("📌 Select your File", file_list2, index=None, placeholder="Select your file to parse", help="Table to Markdown and Up/Down Cropping")
                st.session_state.path = os.path.join(base_dir, sel21)
                st.session_state.path
                crop_check = st.checkbox("Crop", value=True)
                with st.spinner("processing.."):
                    if st.button("Parsing"):
                        st.session_state.pages = ""
                        st.session_state.pages = custom_loader.lazy_load(st.session_state.path, crop_check)
                st.session_state.pages
            except:
                pass
        with st.expander("🗂️ VectorStore(DB)"):
            create_vectordb(st.session_state.pages)

        with st.expander("🔎 Retrieval Test"):
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma(persist_directory="test_index", embedding_function=embed_model)
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            my_query = st.text_input("✏️ text input", placeholder="Input your target senetences for similarity search")
            with st.spinner("Processing..."):
                if st.button("Similarity Search"):
                    st.session_state.retrievals = retriever.invoke(my_query)
            st.session_state.retrievals

        try:
            asyncio.run(rag_main())
        except:
            st.empty()