import asyncio
import requests
import aiohttp
import streamlit as st
from datetime import datetime
import time
from utils import CustomPdfParser, ChromaViewer, CustomPrompts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from pathlib import Path
import os
import pandas as pd

from docx2pdf import convert  # 첨부파일이 word이면.. pdf로 변환
import pythoncom   ## window에서만 설정 필요
pythoncom.CoInitialize()   ## window에서만 설정 필요


st.set_page_config(page_title="AI Captain", layout="wide")

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
parent_dir = str(parent_dir).replace("\\", "/")
base_dir = str(parent_dir) + "/data"  


#### [Start] 공통함수 #############################################################################
def list_selected_files(path, 확장자):
    file_list = os.listdir(path)
    selected_files = [file for file in file_list if file.endswith(확장자)]
    return selected_files

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

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

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.output = ""

if "time_delta" not in st.session_state:
    st.session_state.time_delta = ""
    st.session_state.rag_time_delta = ""
    
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.reversed_messages = ""

if "doc_list" not in st.session_state:
    st.session_state.doc_list = []

if "path" not in st.session_state:
    st.session_state.path = ""
    st.session_state.pages = ""
    st.session_state.retrievals = ""
    st.session_state.rag_doc = ""
    st.session_state.query = ""
    st.session_state.queries = []
    st.session_state.rag_output = ""
    st.session_state.rag_results = []
    st.session_state.rag_history = ""

    st.session_state.trans = ""
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.rag_reversed_messages = ""
    st.session_state.result_df = pd.DataFrame()

if "doc_list" not in st.session_state:
    st.session_state.doc_list = []

if "llms" not in st.session_state:
    st.session_state.llms = []

if "context_list" not in st.session_state:
    st.session_state.context_list = []

custom_parser = CustomPdfParser()
cv = ChromaViewer

#### [Start] Chatbot 함수 ###################################################
def call_jarvis(custom_template, llm_name, input_voice):
    url = "http://127.0.0.1:8000/call_jarvis" 
    json = {"template": custom_template, "llm_name": llm_name, "input_voice": input_voice}
    response = requests.post(url, json=json)
    res = response.json()
    output = res["output"]
    return output

##### [End] Chatbot 함수 ############################################################################

#### [Start] create_vectordb 함수 ###################################################
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
##### [End] create_vectordb 함수 ############################################################################

#### [Start] RAG 함수 ###################################################
def call_rag(custome_template, llm_name, query, temp, top_k, top_p, doc, compress, re_rank, multi_q):
    url = "http://127.0.0.1:8000/call_rag_jarvis"
    json = json={"template": custome_template, "llm_name": llm_name, "input_voice": query, "temperature": temp, "top_k":top_k, "top_p":top_p, "doc": doc, "compress": compress, "re_rank": re_rank, "multi_q":multi_q}
    response = requests.post(url, json=json)
    res = response.json()
    retrival_output = res["output"][0]
    output = res["output"][1]
    return retrival_output, output


##### [End] RAG 함수 ############################################################################


######## [Start] Templates #######################################################################################
cp = CustomPrompts()
custom_templates = cp.custom_template()
rag_sys_templates = cp.rag_sys_template()
######## [End] Templates ###################################################################################

if __name__ == "__main__":

    with st.sidebar:
        st.title("⚓ AI Captain")
        service_type = st.radio("🐬 Services", options=["VectorStore", "Rag", "Open Chat"])
        st.markdown("---")

        if service_type == "Open Chat":
            st.markdown("### LLM")
            llm1 = st.radio("🐬 **Select LLM**", options=["Gemma(2B)", "Phi3(3.8B)", "Llama3.1(8B)", "Gemma2(9B)"], index=0, key="dsfv", help="Bigger LLM returns better answers but takes more time")
            st.markdown("")
            st.markdown("### Prompts")
            sel_template = st.radio("🖋️ Prompt", ["AI_CoPilot", "Medical Assistant"], help="Define the roll of LLM")
            with st.expander("✔️ Prompt Details", expanded=False):
              custome_template = st.text_area("📒 Template", custom_templates[sel_template], height=200)

        elif service_type == "Rag":
            llm1 = st.radio("🐬 **Select LLM**", options=["Phi3(3.8B)", "Llama3.1(8B)", "Gemma2(9B)"], index=0, key="ddfdfsfv", help="Bigger LLM returns better answers but takes more time")
            st.markdown("")
            sel_template = st.radio("🖋️ Prompt", ["Junior_Engineer", "Senior_Engineer", "Korean_Engineer"], help="Define the roll of LLM")
            with st.expander("✔️ Prompt Details", expanded=False):
              custome_template = st.text_area("📒 Template", rag_sys_templates[sel_template], height=200)
            st.markdown("")
            st.markdown("🐳 RAG Options")
            sel_doc_check = st.checkbox("Select Docs", help="If not checked, search every documents. if checked, search only selected documents")
            re_rank_check = st.checkbox("Re Rank", help="Apply Re-Rank", value=True)
            compress_check = st.checkbox("Compress", help="Apply Contextual Compressor")
            multi_check = st.checkbox("Multi Q", help="Apply Multi-Query")
            history_check = st.checkbox("History", help="If checked, LLM will remember the previous query", disabled=True)

            st.markdown("")
            with st.expander("🧪 Hyper-Params"):
                temp = st.slider("🌡️ :blue[Temperature]", min_value=0.0, max_value=2.0, value=0.8, help="The temperature of the model. Increasing the temperature will make the model answer more creatively(Original Default: 0.8)")
                top_k = st.slider("🎲 :blue[Top-K(Proba of Nonsense)]", min_value=0, max_value=100, value=10, help="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative.(Original Default: 40)")
                top_p = st.slider("📝 :blue[Top-P(More Diverse Text)]", min_value=0.0, max_value=1.0, value=0.5, help="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.(Original Default: 0.9)")

        else:
            st.write("***:orange[You can add your own documents in the VectorStore]***")
            uploaded_file = st.file_uploader("📎Upload your file")
            if uploaded_file:
                temp_dir = base_dir   # tempfile.mkdtemp()  --->  import tempfile 필요, 임시저장디렉토리 자동지정함
                path = os.path.join(temp_dir, uploaded_file.name)
                path = str(path).replace("\\", "/")
                path

            if st.button("Save", type='secondary', help="첨부파일이 word(.docx)인 경우, pdf로 변환하여 저장함"):
                with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                if path.split(".")[-1] == "docx":  # 첨부파일이 word면 pdf로 변환후 저장
                    new_path = path.split(".")[:-1]
                    new_path = new_path[0]+".pdf"
                    # new_path
                    convert(path, new_path)
                st.markdown(f"path: {path}")
                st.info("Saving a file is completed")
            else: st.empty()

            try:
                file_list2 = list_selected_files(base_dir, "pdf")
                sel21 = st.selectbox("📌 Select the Parsing File", file_list2, index=None, placeholder="Select your file to parse", help="Table to Markdown and Up/Down Cropping")
                st.session_state.path = os.path.join(base_dir, sel21)
                st.session_state.path = str(st.session_state.path).replace("\\", "/")
                # st.session_state.path
                with st.expander("Splitting Options"):
                    crop_check = st.checkbox("✂️ Crop", value=True)
                    chunk_size = st.slider("📏 Chunk_Size", min_value=1000, max_value=3000, step=100)
                    chunk_overlap = st.slider("⚗️ Chunk_Overlap", min_value=200, max_value=1000, step=50)
                
                with st.spinner("processing.."):
                    if st.button("Parsing"):
                        st.session_state.pages = ""
                        status, temp_pages = custom_parser.pdf_parsing(st.session_state.path, crop_check)
                        status
                        if status:
                            st.session_state.pages = temp_pages
                        else:  # 깡통 PDF인 경우, OCR 파싱 적용
                            st.session_state.pages = custom_parser.ocr_parsing(st.session_state.path)
                        st.info("Parsing is Completed")
                st.markdown(f"Length of Splitted Document: {len(st.session_state.pages)}")

                if st.session_state.pages:
                    create_vectordb(st.session_state.pages, chunk_size, chunk_overlap)    
            except:
                pass


    
    with st.expander("🧭 **Notice**"):
        st.markdown("""
                    - This app is :green[**LLM-driven AI Agent for trouble-shooting in Commercial Vessels**].
                    - :orange[**VectorStore**] is the Storage of your knowledge (embedding your own PDFs)
                    - :orange[**RAG**] is for ***Domain-Specific Conversations*** within VectorStore
                    - :orange[**Chatbot**] is for Common Conversations regarding any interests like techs, medical help, etc.
                    
                    """)
        
    if service_type != "VectorStore":
        text_input1 = st.chat_input("Say something")
    
    if service_type == "Open Chat":
        st.markdown("### 🍓 Open Chat Service")
        if llm1 == "Gemma(2B)": llm_name = "gemma:2b"
        elif llm1 == "Phi3(3.8B)": llm_name = "phi3:latest"
        elif llm1 == "Llama3.1(8B)": llm_name = "llama3.1:latest"
        elif llm1 == "Gemma2(9B)": llm_name = "gemma2:latest"
        elif llm1 == "Ko-Llama3-q4(8B)": llm_name = "ko_llama3_bllossom:latest"
        else: pass

        if text_input1:
            start_time = datetime.now()
            st.session_state.messages.append({"role": "user", "content": text_input1})
            output1 = call_jarvis(custom_templates[sel_template], llm_name, text_input1)
            st.session_state.messages.append({"role": "assistant", "content": output1})
            end_time = datetime.now()
            st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            
        else:
            pass
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
        if st.session_state.time_delta: 
            st.warning(f"⏱️ TimeDelta(Sec) : {st.session_state.time_delta}")
    



    elif service_type == "Rag":
        st.markdown("### 🍀 Rag Service")
        if sel_doc_check:
            with st.expander("📚 Specify the Target Documents", expanded=True):
                sel_doc = st.multiselect("📌 Target Search Documents", st.session_state.doc_list)
        else: sel_doc = None

        if llm1 == "Phi3(3.8B)": llm_name = "phi3:latest"
        elif llm1 == "Llama3.1(8B)": llm_name = "llama3.1:latest"
        elif llm1 == "Gemma2(9B)": llm_name = "gemma2:latest"
        elif llm1 == "Ko-Llama3-q4(8B)": llm_name = "ko_llama3_bllossom:latest"
        else: pass
        st.session_state.llms.append(llm1)


        if text_input1:
            start_time = datetime.now()
            st.session_state.rag_messages.append({"role": "user", "content": text_input1})
            retrival_output, rag_output = call_rag(custome_template, llm_name, text_input1, temp, top_k, top_p, sel_doc, compress_check, re_rank_check, multi_check)
            st.session_state.rag_messages.append({"role": "assistant", "content": rag_output})
            st.session_state.queries.append(text_input1)
            st.session_state.results.append(rag_output)
            st.session_state.rag_doc = retrival_output
            
            end_time = datetime.now()
            st.session_state.rag_time_delta = calculate_time_delta(start_time, end_time)
            
        else:
            pass
        
        for msg in st.session_state.rag_messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
        if st.session_state.rag_time_delta: 
            st.warning(f"⏱️ TimeDelta(Sec) : {st.session_state.rag_time_delta}")
        
            if st.session_state.rag_doc:
                with st.expander("Retrieval Documents(Metadata) & Images"):
                    col111, col222 = st.columns(2)
                    with col111:
        
                        meta_list = []
                        contexts_list = []
                        img_dict = dict()
                        st.session_state.rag_doc

                        for d in st.session_state.rag_doc:
                            page_content, metadata = extract_metadata(d)
                            meta_list.append(metadata)
                            contexts_list.append(page_content)
                            if metadata["keywords"] not in img_dict.keys():
                                img_dict[metadata["keywords"]] =[]
                                img_dict[metadata["keywords"]].append(metadata["page_number"])
                            else:
                                img_dict[metadata["keywords"]].append(metadata["page_number"])
                    with col222:
                        ### [Start] Img Show ---------------------------------------
                        base_img_path = "./images/"
                        target_imgs = []
                        for k in img_dict.keys():
                            img_ids = img_dict[k]
                            for img_id in img_ids:
                                imgfile_name = base_img_path + str(k) + "/" +str(k) + "_" + str(img_id)+".png"
                                target_imgs.append(imgfile_name)
                        # print(f"target_imgs: {target_imgs}")
                        col31, col32 =st.columns(2)
                        with col31: image_show_check = st.checkbox("Show Images", value=True)
                        with col32: page_num = st.number_input(f"Page Order (max:{len(target_imgs)-1})", min_value=0, max_value=len(target_imgs)-1)
                        if image_show_check:
                            st.image(target_imgs[page_num], caption=target_imgs[page_num], width=600)
                        ### [End] Img Show ---------------------------------------



    elif service_type == "VectorStore":
        st.markdown("### 🍇 VectorStore Manager")

        try:
            df = cv.view_collections("vector_index")
            df["title"] = df["metadatas"].apply(lambda x: x["keywords"])
            doc_list = df["title"].unique().tolist()
            st.session_state.doc_list = sorted(doc_list)

            with st.expander(f"📋 Document List ({len(st.session_state.doc_list)})"):
                with st.container():
                    for doc in st.session_state.doc_list:
                        st.markdown(f"- {doc}")

            with st.expander("🔎 Retrieval Test (Similarity Search)", expanded=True):
                embed_model = OllamaEmbeddings(model="nomic-embed-text")
                vectordb = Chroma(persist_directory="vector_index", embedding_function=embed_model)
                my_query = st.text_input("✏️ text input", key="qwqqq", placeholder="Input your target sentences for similarity search")
                with st.spinner("Processing..."):
                    if st.button("Similarity Search"):
                        st.session_state.retrievals = vectordb.similarity_search_with_score(my_query)
                st.session_state.retrievals

            with st.expander(f"**Dataframe Shape: {df.shape}**"):
                st.dataframe(df, use_container_width=True)
        except:
            st.info("There is no VectorStore")

        st.markdown("---")
        st.subheader("Delete Embeded Documents")
        delete_doc = st.selectbox("Target Document", st.session_state.doc_list, index=None)
        # delete_doc
        try:
            del_ids = vectordb.get(where={'keywords':delete_doc})["ids"]
            # del_ids
            if st.button("Delete All Ids"):
                vectordb.delete(del_ids)
                st.info("All Selected Ids were Deleted")
        except:
            st.empty()




    else:
        st.markdown("공사중")

