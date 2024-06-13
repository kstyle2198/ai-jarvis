import asyncio
import aiohttp
import streamlit as st
from datetime import datetime
import time
from utils import CustomPDFLoader, ChromaViewer, CustomPrompts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import os
import pandas as pd

if "center" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.center else "centered"
st.set_page_config(page_title="AI Jarvis", layout=layout)

st.markdown(
            """
        <style>
            .st-emotion-cache-janbn0 {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

parent_dir = Path(__file__).parent
base_dir = str(parent_dir) + "\data"  


#### [Start] 공통함수 #############################################################################
def list_selected_files(path, 확장자):
    file_list = os.listdir(path)
    selected_files = [file for file in file_list if file.endswith(확장자)]
    return selected_files

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

import re
import ast

def extract_metadata(input_string):  # retrieval docs re-ranking and add metadata
    # Use regex to extract the page_content
    page_content_match = re.search(r"page_content='(.+?)'\s+metadata=", input_string, re.DOTALL)
    if page_content_match:
        page_content = page_content_match.group(1)
    else:
        page_content = None

    # Use regex to extract the metadata dictionary
    metadata_match = re.search(r"metadata=(\{.+?\})", input_string)
    if metadata_match:
        metadata_str = metadata_match.group(1)
        # Convert the metadata string to a dictionary
        metadata = ast.literal_eval(metadata_str)
    else:
        metadata = None
    return page_content, metadata
###### [End] 공통함수 #################################################################

#### [Start] Chatbot 함수 ###################################################
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
    try:
        trans_res = await trans(output)
        trans_output = trans_res['output'][0]
    except:
        trans_output = "Translation Does Not Work without Internet"
    return output, trans_output

async def call_jarvis_ko(custom_template, llm_name, input_voice):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/call_jarvis" 
        async with session.post(url, json={"template": custom_template, "llm_name": llm_name, "input_voice": input_voice}) as response:
            res = await response.json()
    output = res["output"]
    return output

async def chat_main(custome_template):
    with st.container():
        llm1 = st.radio("🐬 **Select LLM**", options=["Gemma(2B)", "Phi3(3.8B)", "Llama3(8B)", "Ko-Llama3-q4(8B)"], index=0, key="dsfv", help="Bigger LLM returns better answers but takes more time")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if llm1 == "Gemma(2B)": llm_name = "gemma:2b"
        elif llm1 == "Phi3(3.8B)": llm_name = "phi3:latest"
        elif llm1 == "Llama3(8B)": llm_name = "llama3:latest"
        elif llm1 == "Ko-Llama3-q4(8B)": llm_name = "HD-ko-llama3-q4:latest"
        else: pass
    text_input = st.text_input("✏️ Send your Qeustions", placeholder="Input your Qeustions", key="wqdssd")
    call_btn = st.button("💬 Chat Jarvis", help="")
    with st.spinner("Processing..."):
        if  call_btn and text_input =="":
            time.sleep(1)
            st.info("🗨️ Say 'Jarvis' and then Speak (with 1~2 seconds gap)")
            res = await stt()
            input_voice = res['input_voice']
            start_time = datetime.now()
            st.session_state.messages.append({"role": "user", "content": input_voice})

            if input_voice:
                start_time = datetime.now()
                if llm1 == "Ko-Llama3-q4(8B)":  # 한국어 LLM인 경우
                    output = await call_jarvis_ko(custome_template, llm_name, input_voice)
                    trans_output = "번역없음"
                else:
                    output, trans_output = await call_jarvis(custome_template, llm_name, input_voice)

                st.session_state.output = output
                st.session_state.trans = trans_output
                end_time = datetime.now()
                delta = calculate_time_delta(start_time, end_time)
                st.warning(f"⏱️ TimeDelta(Sec) : {delta}")
                with st.container():
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
                if llm1 == "Ko-Llama3-q4(8B)":  # 한국어 LLM인 경우
                    output = await call_jarvis_ko(custome_template, llm_name, input_voice)
                    trans_output = "번역없음"
                else:
                    output, trans_output = await call_jarvis(custome_template, llm_name, input_voice)    
                st.session_state.output = output
                st.session_state.trans = trans_output
                end_time = datetime.now()
                delta = calculate_time_delta(start_time, end_time)
                st.warning(f"⏱️ TimeDelta(Sec) : {delta}")
                with st.container():
                    col111, col112 = st.columns(2)
                    with col111: st.write(st.session_state.output)
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
##### [End] Chatbot 함수 ############################################################################
        
#### [Start] VectorDB 함수 #################################################    
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

if "doc_list" not in st.session_state:
    st.session_state.doc_list = []

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
            st.info("VectorStore is Updated")
###### [End] VectorDB 함수 ###############################################################################################

#### [Start] RAG_without_History 함수 ####################################################################################    
async def api_ollama(url, custome_template, llm_name, input_voice, temp, top_k, top_p, doc, compress, re_rank, multi_q):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"template": custome_template, "llm_name": llm_name, "input_voice": input_voice, "temperature": temp, "top_k":top_k, "top_p":top_p, "doc": doc, "compress": compress, "re_rank": re_rank, "multi_q":multi_q}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"
   
async def call_rag(custome_template, llm_name, query, temp, top_k, top_p, doc, compress, re_rank, multi_q):
    try:
        url = "http://127.0.0.1:8000/call_rag_jarvis"
        res = await api_ollama(url, custome_template, llm_name, query, temp, top_k, top_p, doc, compress, re_rank, multi_q)
        retrival_output = res["output"][0]
        output = res["output"][1]
        try:
            trans_res = await trans(output)
            trans_output = trans_res['output'][0]
        except:
            trans_output = "Translation Does Not Work without Internet"
        return retrival_output, output, trans_output
    except Exception as e:
        return f"Error: {str(e)}"
  
async def rag_main(custome_template, doc=None, compress=False, re_rank=False, multi_q=False):
    with st.expander("🧪 Hyper-Parameters"):
        col911, col922, col933 = st.columns(3)
        with col911: temp = st.slider("🌡️ :blue[Temperature]", min_value=0.0, max_value=2.0, value=0.8, help="The temperature of the model. Increasing the temperature will make the model answer more creatively(Original Default: 0.8)")
        with col922: top_k = st.slider("🎲 :blue[Top-K(Proba of Nonsense)]", min_value=0, max_value=100, value=10, help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.(Original Default: 40)")
        with col933: top_p = st.slider("📝 :blue[Top-P(More Diverse Text)]", min_value=0.0, max_value=1.0, value=0.5, help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.(Original Default: 0.9)")

    with st.container():
        llm2 = st.radio("🐬 **Select LLM**", options=["Gemma(2B)", "Phi3(3.8B)", "Llama3(8B)"], index=1, key="dsssv", help="Bigger LLM returns better answers but takes more time")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    with st.container():
        if llm2 == "Gemma(2B)": llm_name = "gemma:2b"
        elif llm2 == "Phi3(3.8B)": llm_name = "phi3:latest"
        elif llm2 == "Llama3(8B)": llm_name = "llama3:latest"
        else: pass

    text_input = st.text_input("✏️ Send your Queries", placeholder="Input your Query", key="dls")
    rag_btn = st.button("💬 RAG Jarvis", help="")

    with st.spinner("Processing"):
        if  rag_btn and text_input == "":
            time.sleep(1)
            st.info("🗨️ Say 'Jarvis' and then Speak (with 1~2 seconds gap)")

            res = await stt()
            query = res['input_voice']
            start_time = datetime.now()
            st.session_state.rag_messages.append({"role": "user", "content": query})

            if query:
                retrival_output, output, trans_output = await call_rag(custome_template, llm_name, query, temp, top_k, top_p, doc, compress, re_rank, multi_q)
                st.session_state.rag_doc = retrival_output
                st.session_state.rag_output = output
                st.session_state.trans = trans_output
                end_time = datetime.now()
                delta = calculate_time_delta(start_time, end_time)
                st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

                col111, col112 = st.columns(2)
                with col111: st.write(st.session_state.rag_output)
                with col112: st.write(st.session_state.trans)

                st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})

                if st.session_state.rag_output:
                    await tts(st.session_state.rag_output)

        elif rag_btn and text_input:
            start_time = datetime.now()
            query = text_input
            st.session_state.rag_messages.append({"role": "user", "content": query})

            retrival_output, output, trans_output = await call_rag(custome_template, llm_name, query, temp, top_k, top_p, doc, compress, re_rank, multi_q)
            st.session_state.rag_doc = retrival_output
            st.session_state.rag_output = output
            st.session_state.trans = trans_output
            end_time = datetime.now()

            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

            col111, col112 = st.columns(2)
            with col111: st.write(st.session_state.rag_output)
            with col112: st.write(st.session_state.trans)
            
            st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})
            if st.session_state.rag_output:
                await tts(st.session_state.rag_output)

    st.markdown("---")
    st.session_state.rag_reversed_messages = st.session_state.rag_messages[::-1]
    with st.expander("Retrieval Documents(Metadata) & Images"):
        meta_list = []
        img_dict = dict()
        for d in st.session_state.rag_doc:
            page_content, metadata = extract_metadata(d)
            meta_list.append(metadata)
            if metadata["keywords"] not in img_dict.keys():
                img_dict[metadata["keywords"]] =[]
                img_dict[metadata["keywords"]].append(metadata["page_number"])
            else:
                img_dict[metadata["keywords"]].append(metadata["page_number"])
        meta_list
        base_img_path = "./images/"
        for k in img_dict.keys():
            path = base_img_path + str(k)
            imgs = list_selected_files(path, "png")

        sel2_img = [x for x in imgs if int(x.split("_")[0]) in img_dict[k]]
        image_show_check = st.checkbox("Show Images", value=True)
        if image_show_check:
            for i in sel2_img:
                path = base_img_path +str(k) +"/"+str(i)
                st.image(path, caption=path)

    ##### [Start] DATAFRAME 생성 및 저장 ---------------------------------------
    with st.expander("Save Response Results(CSV file)"):
        df = pd.DataFrame(st.session_state.rag_reversed_messages)
        st.dataframe(df.head(), use_container_width=True)
        file_name = st.text_input("Input your file name", placeholder="Input your unique file name")
        if st.button("Save"):
            df.to_csv(f"./results/{file_name}.csv")
            st.info("File is Saved")
    ##### [End] DATAFRAME 생성 및 저장 ---------------------------------------

    for msg in st.session_state.rag_reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
##### [End] RAG_without_History 함수 ############################################################################

##### [Start] RAG with History ##########################################################################################
async def api_ollama_history(url, custome_template, llm_name, input_voice, temp, top_k, top_p, history_key, doc, compress, re_rank, multi_q):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"template": custome_template, "llm_name": llm_name, "input_voice": input_voice, "temperature": temp, "top_k":top_k, "top_p":top_p, "history_key": history_key, "doc":doc, "compress": compress, "re_rank": re_rank, "multi_q":multi_q}) as response:
                res = await response.json()
        return res
    except Exception as e:
        return f"Error: {str(e)}"
    
store = {}
async def call_rag_with_history(custome_template, llm_name, query, temp, top_k, top_p, history_key, doc, compress, re_rank, multi_q):
    global store
    try:
        url = "http://127.0.0.1:8000/call_rag_jarvis_with_history"
        res = await api_ollama_history(url, custome_template, llm_name, query, temp, top_k, top_p, history_key, doc, compress, re_rank, multi_q)
        if re_rank: retrival_output = res["output"][0]["retrieved_docs"]
        else:  retrival_output = res["output"][0]["context"]
        output = res["output"][0]["answer"]
        history = res["output"][0]["chat_history"]
        try:
            trans_res = await trans(output)
            trans_output = trans_res['output'][0]
        except:
            trans_output = "Translation Does Not Work without Internet"
        return retrival_output, output, history, trans_output
    except Exception as e:
        return f"Error: {str(e)}"
    
async def rag_main_history(custome_template, doc, compress=False, re_rank=False, multi_q=False):
    global store
    with st.expander("🧪 Hyper-Parameters"):
        col9111, col9222, col9333 = st.columns(3)
        with col9111: temp = st.slider("🌡️ :blue[Temperature]", min_value=0.0, max_value=2.0, value=0.8, key="wedsf", help="The temperature of the model. Increasing the temperature will make the model answer more creatively.(Original Default: 0.8)")
        with col9222: top_k = st.slider("🎲 :blue[Top-K(Proba of Nonsense)]", min_value=0, max_value=100, value=10, key="xvvd", help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.(Original Default: 40)")
        with col9333: top_p = st.slider("📝 :blue[Top-P(More Diverse Text)]", min_value=0.0, max_value=1.0, value=0.5, key="qwer", help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Original Default: 0.9)")

    with st.container():
        llm2 = st.radio("🐬 **Select LLM**", options=["Gemma(2B)", "Phi3(3.8B)", "Llama3(8B)"], index=1, key="dsssadfsv", help="Bigger LLM returns better answers but takes more time")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    with st.container():
        if llm2 == "Gemma(2B)": llm_name = "gemma:2b"
        elif llm2 == "Phi3(3.8B)": llm_name = "phi3:latest"
        elif llm2 == "Llama3(8B)": llm_name = "llama3:latest"
        else: pass

    text_input = st.text_input("✏️ Send your Queries", placeholder="Input your Query", key="dlsdfg")
    
    col31, col32, col33 = st.columns(3)
    with col31: rag_btn = st.button("💬 RAG Jarvis", help="", key="wqwe")
    with col32: history_init = st.button("🗑️ Init History", help="Remove Conversation History(Init)")
    with col33: history_key = st.number_input("🔑 history_key", min_value=1, step=1, key="wqeqq", help="History to be remembered under the same key(id)")

    if history_init: store ={}

    with st.spinner("Processing"):
        if  rag_btn and text_input == "":
            time.sleep(1)
            st.info("🗨️ Say 'Jarvis' and then Speak (with 1~2 seconds gap)")

            res = await stt()
            query = res['input_voice']
            start_time = datetime.now()
            st.session_state.rag_messages.append({"role": "user", "content": query})

            if query:
                retrival_output, output, history, trans_output = await call_rag_with_history(custome_template, llm_name, query, temp, top_k, top_p, history_key, doc, compress, re_rank, multi_q)
                st.session_state.rag_doc = retrival_output
                st.session_state.rag_output = output
                st.session_state.rag_history = history
                st.session_state.trans = trans_output
                end_time = datetime.now()
                delta = calculate_time_delta(start_time, end_time)
                st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

                col81, col82 = st.columns(2)
                with col81: st.write(st.session_state.rag_output)
                with col82: st.write(st.session_state.trans)
                
                with st.expander("History"):
                    st.write(st.session_state.rag_history)

                st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})

                if st.session_state.rag_output:
                    await tts(st.session_state.rag_output)

        elif rag_btn and text_input:
            start_time = datetime.now()
            query = text_input
            st.session_state.rag_messages.append({"role": "user", "content": query})

            retrival_output, output, history, trans_output = await call_rag_with_history(custome_template, llm_name, query, temp, top_k, top_p, history_key, doc, compress, re_rank, multi_q)
            st.session_state.rag_doc = retrival_output
            st.session_state.rag_output = output
            st.session_state.rag_history = history
            st.session_state.trans = trans_output
            end_time = datetime.now()
            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ TimeDelta(Sec) : {delta}")

            col81, col82 = st.columns(2)
            with col81: st.write(st.session_state.rag_output)
            with col82: st.write(st.session_state.trans)
            with st.expander("History"):
                st.write(st.session_state.rag_history)

            st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})
            if st.session_state.rag_output:
                await tts(st.session_state.rag_output)

    st.markdown("---")
    st.session_state.rag_reversed_messages = st.session_state.rag_messages[::-1]
    with st.expander("Retrieval Documents(Metadata) & Images"):
        meta_list = []
        img_dict = dict()
        for d in st.session_state.rag_doc:
            page_content, metadata = extract_metadata(d)
            meta_list.append(metadata)
            if metadata["keywords"] not in img_dict.keys():
                img_dict[metadata["keywords"]] =[]
                img_dict[metadata["keywords"]].append(metadata["page_number"])
            else:
                img_dict[metadata["keywords"]].append(metadata["page_number"])
        meta_list
        base_img_path = "./images/"
        for k in img_dict.keys():
            path = base_img_path + str(k)
            imgs = list_selected_files(path, "png")

        sel2_img = [x for x in imgs if int(x.split("_")[0]) in img_dict[k]]
        image_show_check = st.checkbox("Show Images", value=True)
        if image_show_check:
            for i in sel2_img:
                path = base_img_path +str(k) +"/"+str(i)
                st.image(path, caption=path)

    ##### [Start] DATAFRAME 생성 및 저장 ---------------------------------------
    with st.expander("Save Response Results(CSV file)"):
        df = pd.DataFrame(st.session_state.rag_reversed_messages)
        st.dataframe(df.head(), use_container_width=True)
        file_name = st.text_input("Input your file name", placeholder="Input your unique file name")
        if st.button("Save"):
            df.to_csv(f"./results/{file_name}.csv")
            st.info("File is Saved")
    ##### [End] DATAFRAME 생성 및 저장 ---------------------------------------

    for msg in st.session_state.rag_reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
######### [End] RAG with History #####################################################################################

######## [Start] Templates #######################################################################################
cp = CustomPrompts()
custom_templates = cp.custom_template()
rag_sys_templates = cp.rag_sys_template()
######## [End] Templates ###################################################################################

if __name__ == "__main__":
    st.title("⚓ AI Jarvis")
    st.checkbox("Wide Layout", key="center", value=st.session_state.get("center", False))
    with st.expander("🚢 Note"):
        st.markdown("""
                    - This AI app is created for :green[**Local Chatbot and RAG Service using sLLM without Internet**].
                    - :orange[**Chatbot**] is for Common Conversations regarding any interests like techs, movies, etc.
                    - :orange[**RAG**] is for ***Domain-Specific Conversations*** using VectorStore (embedding your PDFs)
                    - In the Dev mode, Translation API needs Internet. (will be excluded in the Production Mode)
                    """)
    tab1, tab2, tab3, tab4 = st.tabs(["⚾ **Chatbot**", "⚽ **RAG**", "🗄️ **VectorStore**", "⚙️ **Prompt_Engineering**"])

    with tab1:
        with st.expander("✔️ Select Prompt Concept", expanded=False):
            sel_template = st.radio("🖋️ Select & Edit", ["AI_CoPilot", "한글_테스트", "English_Teacher", "Movie_Teller", "Food_Teller"], help="Define the roll of LLM")
            custome_template = st.text_area("📒 Template", custom_templates[sel_template], height=200)
        asyncio.run(chat_main(custome_template))

    with tab2:
        col71, col72, col73, col74, col75,  = st.columns([4, 5, 4, 5, 4])
        with col71: history_check = st.checkbox("History", help="If checked, LLM will remember our conversation history")
        with col72: sel_doc_check = st.checkbox("Select Docs", help="If not checked, search every documents. if checked, search only selected documents")
        with col73: re_rank_check = st.checkbox("Re Rank", help="Apply Re-Rank")
        with col74: compress_check = st.checkbox("Compress", help="Apply Contextual Compressor")
        with col75: multi_check = st.checkbox("Multi Q", help="Apply Multi-Query")
        if sel_doc_check:
            with st.expander("📚 Specify the Target Documents", expanded=True):
                sel_doc = st.multiselect("📌 Target Search Documents", st.session_state.doc_list)
        else: sel_doc = None
        with st.expander("✔️ Select Prompt Concept", expanded=False):
            sel_template = st.radio("🖋️ Select & Edit", ["Common_Engineer", "Navigation_Engineer", "Electrical_Engineer"], help="Define the roll of LLM")
            custome_template = st.text_area("📒 Template", rag_sys_templates[sel_template], height=200)
        try:
            if history_check:
                store = {}
                asyncio.run(rag_main_history(custome_template, sel_doc, compress_check, re_rank_check, multi_check))
            else: asyncio.run(rag_main(custome_template, sel_doc, compress_check, re_rank_check, multi_check))
        except:
            st.empty()

    from docx2pdf import convert  # 첨부파일이 word이면.. pdf로 변환
    import pythoncom
    pythoncom.CoInitialize()


    with tab3:
        with st.expander("🧩 Custom Parsing & VectorStore(DB)"):
            uploaded_file = st.file_uploader("📎Upload your file")
            if uploaded_file:
                temp_dir = base_dir   # tempfile.mkdtemp()  --->  import tempfile 필요, 임시저장디렉토리 자동지정함
                path = os.path.join(temp_dir, uploaded_file.name)
                path
            
            if st.button("Save", type='secondary', help="첨부파일이 word(.docx)인 경우, pdf로 변환하여 저장함"):
                with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                if path.split(".")[-1] == "docx":  # 첨부파일이 word면 pdf로 변환후 저장
                    new_path = path.split(".")[:-1]
                    new_path = new_path[0]+".pdf"
                    new_path
                    convert(path, new_path)
                st.markdown(f"path: {path}")
                st.info("Saving a file is completed")
            else: st.empty()

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
                        st.info("Parsing is Completed")
                st.session_state.pages

                if st.session_state.pages:
                    create_vectordb(st.session_state.pages, chunk_size, chunk_overlap)    
            except:
                st.success("There is no selected file")

        df = cv.view_collections("vector_index")
        df["title"] = df["metadatas"].apply(lambda x: x["keywords"])
        doc_list = df["title"].unique().tolist()
        st.session_state.doc_list = sorted(doc_list)

        with st.expander("📋 Document List"):
            for doc in st.session_state.doc_list:
                st.markdown(f"- {doc}")

        with st.expander("🔎 Retrieval Test (Similarity Search)"):
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma(persist_directory="vector_index", embedding_function=embed_model)
            my_query = st.text_input("✏️ text input", key="qwqqq", placeholder="Input your target senetences for similarity search")
            with st.spinner("Processing..."):
                if st.button("Similarity Search"):
                    st.session_state.retrievals = vectordb.similarity_search_with_score(my_query)
            st.session_state.retrievals

        st.markdown(f"**Dataframe Shape: {df.shape}**")
        st.dataframe(df, use_container_width=True)

    with tab4:
        st.warning("Under Construction")
        st.empty()
        



     