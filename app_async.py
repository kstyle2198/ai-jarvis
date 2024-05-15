import asyncio
import aiohttp
import streamlit as st
from datetime import datetime

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

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()
    
@st.experimental_fragment
async def call_jarvis(llm_name, input_voice):
    urls = {
        "tinydolphin(1.1B)": f"http://127.0.0.1:8000/call_tinydolphin?input_voice={input_voice}",
        "moondream(1B)": f"http://127.0.0.1:8000/call_moondream?input_voice={input_voice}",
        "dolphin-phi(2.7B)": f"http://127.0.0.1:8000/call_dolphinphi?input_voice={input_voice}",
        "phi3(3.8B)": f"http://127.0.0.1:8000/call_phi3?input_voice={input_voice}",
        "llama3": f"http://127.0.0.1:8000/call_llama3?input_voice={input_voice}",
        "Groq_llama3": f"http://127.0.0.1:8000/call_groq_llama3?input_voice={input_voice}"
    }
    url = urls.get(llm_name)
    async with aiohttp.ClientSession() as session:
        if url:
            res2 = await fetch(session, url)
            output = res2['output']

            trans_url = f"http://127.0.0.1:8000/call_trans?txt={output}"
            trans_res = await fetch(session, trans_url)
            trans_output = trans_res['output'][0]

            return output, trans_output
        return None, None

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

sample_template = '''
you are an smart AI assistant in a commercial vessel like LNG Carriers or Container Carriers.
your answer always starts with "OK, Master".
generate compact and summarized answer to {query} kindly and shortly.
if there are not enough information to generate answers, just return "Please give me more information"
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?'
'''

@st.experimental_fragment
async def chat_main():
    with st.expander("🐳 **Custom Prompt**"):
        custom_template = st.markdown(sample_template)

    with st.container():
        llm_name = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3", "Groq_llama3"], index=1, key="dsfv")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    
    if st.button("💬 Call Jarvis"):
        async with aiohttp.ClientSession() as session:
            res1 = await fetch(session, "http://127.0.0.1:8000/jarvis_stt")
            start_time = datetime.now()
            input_voice = res1.get("input_voice")

            st.session_state.messages.append({"role": "user", "content": input_voice})
            if input_voice:
                output, trans_output = await call_jarvis(llm_name, input_voice)
                st.session_state.output = output
                st.session_state.trans = trans_output
                end_time = datetime.now()

                delta = calculate_time_delta(start_time, end_time)
                st.warning(f"⏱️ 응답소요시간(초) : {delta}")

                col111, col112 = st.columns(2)
                with col111: st.write(st.session_state.output)
                with col112: st.write(st.session_state.trans)

                st.session_state.messages.append({"role": "assistant", "content": st.session_state.output})

                if st.session_state.output:
                    tts_url = f"http://127.0.0.1:8000/jarvis_tts?output={st.session_state.output}"
                    await fetch(session, tts_url)

    st.markdown("---")
    st.session_state.reversed_messages = st.session_state.messages[::-1]
    for msg in st.session_state.reversed_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
    

if "retriever" not in st.session_state:
    st.session_state.retirever = ""

if "path" not in st.session_state:
    st.session_state.path = ""
    st.session_state.rag_results = []
    st.session_state.rag_docs = []
    st.session_state.rag_output = ""
    st.session_state.trans = ""
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.rag_reversed_messages = ""


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

@st.experimental_fragment
def create_vectordb(pdf_text):
    with st.spinner("Processing..."):
        if st.button("Create Vectorstore"):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splitted_texts = text_splitter.split_text(pdf_text)
            embed_model = OllamaEmbeddings(model="nomic-embed-text")
            db=Chroma.from_texts(splitted_texts, embedding=embed_model, persist_directory="test_index")
        if st.session_state.retirever: 
            st.session_state.retirever
            st.info("VectorStore Created")

@st.experimental_fragment
async def call_rag(llm_name, query):
    urls = {
        "tinydolphin(1.1B)": f"http://127.0.0.1:8000/call_rag_tinydolphin?query={query}",
        "dolphin-phi(2.7B)": f"http://127.0.0.1:8000/call_rag_dolphin_phi?query={query}",
        "phi3(3.8B)": f"http://127.0.0.1:8000/call_rag_phi3?query={query}",
        "llama3": f"http://127.0.0.1:8000/call_rag_llama3?query={query}",
        "Groq_llama3": f"http://127.0.0.1:8000/call_rag_groq_llama3?query={query}"
    }
    url = urls.get(llm_name)
    async with aiohttp.ClientSession() as session:
        if url:
            rag_res = await fetch(session, url)
            output = rag_res['output']

            trans_url = f"http://127.0.0.1:8000/call_trans?txt={output}"
            trans_res = await fetch(session, trans_url)
            trans_output = trans_res['output'][0]

            return output, trans_output
        return None, None
    
async def rag_main():
    with st.container():
        llm_name = st.radio("🐬 **Select LLM**", options=["tinydolphin(1.1B)", "dolphin-phi(2.7B)", "phi3(3.8B)", "llama3", "Groq_llama3"], index=1, key="dsssv")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    with st.spinner("Processing..."):
        if st.button("💬 RAG Jarvis"):
            async with aiohttp.ClientSession() as session:
                res1 = await fetch(session, "http://127.0.0.1:8000/jarvis_stt")
                start_time = datetime.now()
                query = res1.get("input_voice")

                if query:
                    output, trans_output = await call_rag(llm_name, query)
                    st.session_state.rag_output = output
                    st.session_state.trans = trans_output
                    end_time = datetime.now()

                    delta = calculate_time_delta(start_time, end_time)
                    st.warning(f"⏱️ 응답소요시간(초) : {delta}")

                    col111, col112 = st.columns(2)
                    with col111: st.write(st.session_state.rag_output)
                    with col112: st.write(st.session_state.trans)

                    st.session_state.rag_messages.append({"role": "user", "content": query})
                    st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.rag_output})

                    if st.session_state.rag_output:
                        tts_url = f"http://127.0.0.1:8000/jarvis_tts?output={st.session_state.rag_output}"
                        await fetch(session, tts_url)

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


if __name__ == "__main__":
    st.title("⚓ :blue[HD Jarvis]")
    st.markdown("---")

    tab1, tab2 = st.tabs(["⚾ **Chatbot**", "⚽ **RAG**"])
    with tab1:
        asyncio.run(chat_main())
    with tab2:
        with st.expander("📑 File Uploader anc VectorStore"):
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
            asyncio.run(rag_main())
        except:
            st.empty()