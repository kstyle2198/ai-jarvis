import asyncio
import aiohttp
import streamlit as st
from datetime import datetime
import time
from utils import CustomPDFLoader, ShowPdf, ChromaViewer
import pandas as pd
import json

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


async def call_jarvis(custom_template, llm_name, input_voice):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/call_jarvis" 
        async with session.post(url, json={"template": custom_template, "llm_name": llm_name, "input_voice": input_voice}) as response:
            res = await response.json()

    output = res["output"]
    trans_res = await trans(output)
    trans_output = trans_res['output'][0]

    return output, trans_output

async def chat_main(custome_template):

    with st.container():
        llm1 = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "Gemma(2B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3(8B)"], index=1, key="dsfv", help="Internet is not needed")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if llm1 == "tinydolphin(1.1B)":
            llm_name = "tinydolphin:latest"
        elif llm1 == "Gemma(2B)":
            llm_name = "gemma:2b"
        elif llm1 == "dolphin-phi(2.7B)":
            llm_name = "dolphin-phi:latest"
        elif llm1 == "phi3(3.8B)":
            llm_name = "phi3:latest"
        elif llm1 == "llama3(8B)":
            llm_name = "llama3:latest"
        else:
            pass

    text_input = st.text_input("✏️ Send your Qeustions", placeholder="Input your Qeustions", key="dldfs")
    call_btn = st.button("💬 Chat Jarvis", help="Without Text Query, Click & Say 'Jarvis' after 2~3 seconds. Jarvis will replay 'Yes, Master' and then Speak your Requests")
    if  call_btn and text_input =="":
        res = await stt()
        input_voice = res['input_voice']
        start_time = datetime.now()
        st.session_state.messages.append({"role": "user", "content": input_voice})

        if input_voice:
            output, trans_output = await call_jarvis(custome_template, llm_name, input_voice)
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
            output, trans_output = await call_jarvis(custome_template, llm_name, input_voice)
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
    st.session_state.rag_history = ""
    st.session_state.trans = ""
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.rag_reversed_messages = ""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# show_pdf = ShowPdf()
custom_loader = CustomPDFLoader()
cv = ChromaViewer

def create_vectordb(parsed_text, chunk_size=1000, chunk_overlap=200):  # VectorDB생성 및 저장
    with st.spinner("Processing..."):
        if st.button("Create Vectorstore", help="You can add your PDFs in the VectorStore After Parsing"):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splitted_texts = text_splitter.split_documents(parsed_text)
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            db=Chroma.from_documents(splitted_texts, embedding=embed_model, persist_directory="vector_index")
        if st.session_state.retirever: 
            st.session_state.retirever
            st.info("VectorStore Created")

#### RAG 함수 #################################################    

async def api_ollama(url, custome_template, llm_name, input_voice, temp, top_k, top_p):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"template": custome_template, "llm_name": llm_name, "input_voice": input_voice, "temperature": temp, "top_k":top_k, "top_p":top_p}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"

   
async def call_rag(custome_template, llm_name, query, temp, top_k, top_p):
    try:
        url = "http://127.0.0.1:8000/call_rag_jarvis"
        res = await api_ollama(url, custome_template, llm_name, query, temp, top_k, top_p)
        retrival_output = res["output"][0]
        output = res["output"][1]
        trans_res = await trans(output)
        trans_output = trans_res['output'][0]
        return retrival_output, output, trans_output
    except Exception as e:
        return f"Error: {str(e)}"
  
async def rag_main(custome_template):

    col911, col922, col933 = st.columns(3)
    with col911: 
        temp = st.slider("🌡️ :blue[Temperature(Default:0)]", min_value=0.0, max_value=2.0)
    with col922: 
        top_k = st.slider("🎲 :blue[Probability of Nonsense(Default:0)]", min_value=0, max_value=100)
    with col933: 
        top_p = st.slider("📝 :blue[More Diverse Text(Default:0)]", min_value=0.0, max_value=1.0)

    with st.container():
        llm2 = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "Gemma(2B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3(8B)"], index=1, key="dsssv", help="Internet is not needed")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    with st.container():
        if llm2 == "tinydolphin(1.1B)":
            llm_name = "tinydolphin:latest"
        elif llm2 == "Gemma(2B)":
            llm_name = "gemma:2b"
        elif llm2 == "dolphin-phi(2.7B)":
            llm_name = "dolphin-phi:latest"
        elif llm2 == "phi3(3.8B)":
            llm_name = "phi3:latest"
        elif llm2 == "llama3(8B)":
            llm_name = "llama3:latest"
        else:
            pass

    text_input = st.text_input("✏️ Send your Queries", placeholder="Input your Query", key="dls")
    rag_btn = st.button("💬 RAG Jarvis", help="Without Text Query, Click & Say 'Jarvis'. Jarvis will replay 'Yes, Master' and then Speak your Requests")

    with st.spinner("Processing"):
        if  rag_btn and text_input == "":
            res = await stt()
            query = res['input_voice']
            start_time = datetime.now()
            st.session_state.rag_messages.append({"role": "user", "content": query})

            if query:
                retrival_output, output, trans_output = await call_rag(custome_template, llm_name, query, temp, top_k, top_p)
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

            retrival_output, output, trans_output = await call_rag(custome_template, llm_name, query, temp, top_k, top_p)
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




async def api_ollama_history(url, llm_name, input_voice, temp, top_k, top_p, history_key):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"llm_name": llm_name, "input_voice": input_voice, "temperature": temp, "top_k":top_k, "top_p":top_p, "history_key": history_key}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"
    

store = {}
async def call_rag_with_history(llm_name, query, temp, top_k, top_p, history_key):
    global store
    try:
        url = "http://127.0.0.1:8000/call_rag_jarvis_with_history"
        res = await api_ollama_history(url, llm_name, query, temp, top_k, top_p, history_key)
        retrival_output = res["output"][0]["context"]
        output = res["output"][0]["answer"]
        history = res["output"][0]["chat_history"]
        trans_res = await trans(output)
        trans_output = trans_res['output'][0]
        
        return retrival_output, output, history, trans_output
    except Exception as e:
        return f"Error: {str(e)}"
    
async def rag_main_history():
    global store
    col9111, col9222, col9333 = st.columns(3)
    with col9111: 
        temp = st.slider("🌡️ :blue[Temperature(Default:0)]", min_value=0.0, max_value=2.0, key="wedsf")
    with col9222: 
        top_k = st.slider("🎲 :blue[Probability of Nonsense(Default:0)]", min_value=0, max_value=100, key="xvvd")
    with col9333: 
        top_p = st.slider("📝 :blue[More Diverse Text(Default:0)]", min_value=0.0, max_value=1.0, key="qwer")

    with st.container():
        llm2 = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "Gemma(2B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3(8B)"], index=1, key="dsssadfsv", help="Internet is not needed")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    with st.container():
        if llm2== "tinydolphin(1.1B)":
            llm_name = "tinydolphin:latest"
        elif llm2 == "Gemma(2B)":
            llm_name = "gemma:2b"
        elif llm2 == "dolphin-phi(2.7B)":
            llm_name = "dolphin-phi:latest"
        elif llm2 == "phi3(3.8B)":
            llm_name = "phi3:latest"
        elif llm2 == "llama3(8B)":
            llm_name = "llama3:latest"
        else:
            pass

    text_input = st.text_input("✏️ Send your Queries", placeholder="Input your Query", key="dlsdfg")
    
    col31, col32, col33 = st.columns(3)
    with col31:
        rag_btn = st.button("💬 RAG Jarvis", help="Without Text Query, Click & Say 'Jarvis'. Jarvis will replay 'Yes, Master' and then Speak your Requests", key="wqwe")
    with col32:
        history_init = st.button("🗑️ Init History", help="대화 History 삭제(초기화)")
    with col33:
        history_key = st.number_input("🔑 history_key", min_value=1, step=1, key="wqeqq", help="History를 기억하는 구분 id (같은 id 범위내 대화 이력 기억)")

    if history_init:
        store ={}

    with st.spinner("Processing"):
        if  rag_btn and text_input == "":
            res = await stt()
            query = res['input_voice']
            start_time = datetime.now()
            st.session_state.rag_messages.append({"role": "user", "content": query})

            if query:
                retrival_output, output, history, trans_output = await call_rag_with_history(llm_name, query, temp, top_k, top_p, history_key)
                st.session_state.rag_doc = retrival_output
                st.session_state.rag_output = output
                st.session_state.rag_history = history
                st.session_state.trans = trans_output
                end_time = datetime.now()
                delta = calculate_time_delta(start_time, end_time)
                st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

                col81, col82 = st.columns(2)
                with col81: 
                    st.write(st.session_state.rag_output)
                    st.write(st.session_state.trans)
                with col82: st.write(st.session_state.rag_history)
                st.write(st.session_state.rag_doc)

                st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})

                if st.session_state.rag_output:
                    await tts(st.session_state.rag_output)

        elif rag_btn and text_input:
            start_time = datetime.now()
            query = text_input
            st.session_state.rag_messages.append({"role": "user", "content": query})

            retrival_output, output, history, trans_output = await call_rag_with_history(llm_name, query, temp, top_k, top_p, history_key)
            st.session_state.rag_doc = retrival_output
            st.session_state.rag_output = output
            st.session_state.rag_history = history
            st.session_state.trans = trans_output
            end_time = datetime.now()

            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

            col81, col82 = st.columns(2)
            with col81: 
                st.write(st.session_state.rag_output)
                st.write(st.session_state.trans)
            with col82: st.write(st.session_state.rag_history)
            st.write(st.session_state.rag_doc)


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


custome_templates = {
"AI_CoPilot": '''you are an smart AI assistant in a commercial vessel like LNG Carriers or Container Carriers.
your answer always starts with "OK, Master".
generate compact and summarized answer to {query} with numbering kindly and shortly.
if there are not enough information to generate answers, just return "Please give me more information"
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?'
''',
"English_Teacher": '''you are an smart AI English teacher to teach expresssions about daily life.
your answer always starts with "Ok, let's get started".
generate a compact and short answer correponding to {query}, like conversations between friends in a school.
if there are not some syntex errors in query, generated the corrected expression in kind manner.
''',
"Movie_Teller": "Not prepared yet",
"Food_Teller": "Not prepared yet"}


rag_sys_templates = {
'Common_Engineer' :"""You are a smart AI engineering advisor in Commercial Vessel like LNG Carrier.
Generate compact and summarized answer based on the {context} using numbering.
If the context or metadata doesn't contain any relevant information to the question, don't make something up and just say 'I don't know':
""",
'Navigation_Engineer':"""You are a smart AI specialist of Integrated Smartship Solution(ISS).
Generate compact and summarized answer based on the {context} using numbering.
If the context doesn't contain any relevant information to the question, don't make something up and just say 'I don't know':
""",
'Electrical_Engineer': "Not prepared yet",

}

def json_to_columns(json_str):
    json_dict = json.loads(json_str)
    return pd.Series(json_dict)


if __name__ == "__main__":
    st.title("⚓ :blue[AI Jarvis]")
    with st.expander("🚢 Note"):
        st.markdown("""
                    - This AI app is created for :green[**Local Chatbot and RAG Service without Internet**].
                    - :orange[**Chatbot**] is for Common Conversations regarding any interests like techs, movies, etc.
                    - :orange[**RAG**] is for ***Domain-Specific Conversations*** using VectorStore (embedding your PDFs)
                    - In the Dev mode,Translation API needs Internet. (will be excluded in the Production Mode)
                    """)
    tab1, tab2, tab3, tab4 = st.tabs(["⚾ **Chatbot**", "⚽ **RAG**", "🏀 **RAG_with_History**","🗄️ **VectorStore**"])
    with tab1:
        with st.expander("✔️ Select Prompt Concept", expanded=False):
            sel_template = st.radio("🖋️ Select & Edit", ["AI_CoPilot", "English_Teacher", "Movie_Teller", "Food_Teller"])
            custome_template = st.text_area("Template", custome_templates[sel_template], height=200)
        asyncio.run(chat_main(custome_template))

    with tab2:
        with st.expander("🧩 Custom Parsing & VectorStore(DB)"):
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
                sel21 = st.selectbox("📌 Select the Parsing File", file_list2, index=None, placeholder="Select your file to parse", help="Table to Markdown and Up/Down Cropping")
                st.session_state.path = os.path.join(base_dir, sel21)
                st.session_state.path

                col31, col32, col33 = st.columns(3)
                with col31: crop_check = st.checkbox("✂️ Crop", value=True)
                with col32: chunk_size = st.slider("📏 Chunk_Size", min_value=1000, max_value=3000, step=100)
                with col33: chunk_overlap = st.slider("⚗️ Chunk_Overlap", min_value=200, max_value=1000, step=50)
                
                with st.spinner("processing.."):
                    if st.button("Parsing"):
                        st.session_state.pages = ""
                        st.session_state.pages = custom_loader.lazy_load(st.session_state.path, crop_check)
                st.session_state.pages
                if st.session_state.pages:
                    create_vectordb(st.session_state.pages, chunk_size, chunk_overlap)
            except:
                pass
            

        with st.expander("🔎 Retrieval Test (Similarity Search)"):
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma(persist_directory="vector_index", embedding_function=embed_model)
            # retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            my_query = st.text_input("✏️ text input", placeholder="Input your target senetences for similarity search")
            with st.spinner("Processing..."):
                if st.button("Similarity Search"):
                    st.session_state.retrievals = vectordb.similarity_search_with_score(my_query)
            st.session_state.retrievals

        with st.expander("✔️ Select Prompt Concept", expanded=False):
            sel_template = st.radio("🖋️ Select & Edit", ["Common_Engineer", "Navigation_Engineer", "Electrical_Engineer"])
            custome_template = st.text_area("Template", rag_sys_templates[sel_template], height=200)

        try:
            asyncio.run(rag_main(custome_template))
        except:
            st.empty()

    with tab3:

        try:
            store = {}
            asyncio.run(rag_main_history())
        except:
            st.empty()


    with tab4:
        df = cv.view_collections("vector_index")
        df["title"] = df["metadatas"].apply(lambda x: x["keywords"])
        doc_list = df["title"].unique().tolist()
        doc_list = sorted(doc_list)
        
        with st.expander("📋 Document List"):
            for doc in doc_list:
                st.markdown(f"- {doc}")

        st.markdown(f"**Dataframe Shape: {df.shape}**")
        st.dataframe(df, use_container_width=True)

        



     