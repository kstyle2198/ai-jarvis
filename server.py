from fastapi import FastAPI, Response
import uvicorn
import json

from RealtimeTTS import TextToAudioStream, SystemEngine
from RealtimeSTT import AudioToTextRecorder
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.chat_history import BaseChatMessageHistory
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import re
import ast
import logging

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

selected_voice = "David"   # David, Hazel
language = "en"

# selected_voice = "Heami"   
# language = "ko"

############# [Start] 공통함수 + STT + TTS ###################################################################################################
def extract_metadata(input_string):
    # Use regex to extract the page_content
    page_content_match = re.search(r"page_content='(.+?)'\s+metadata=", input_string, re.DOTALL)
    if page_content_match: page_content = page_content_match.group(1)
    else: page_content = None
    # Use regex to extract the metadata dictionary
    metadata_match = re.search(r"metadata=(\{.+?\})", input_string)
    if metadata_match:
        metadata_str = metadata_match.group(1)
        # Convert the metadata string to a dictionary
        metadata = ast.literal_eval(metadata_str)
    else: metadata = None
    return page_content, metadata

def say_hello():
    global language
    TextToAudioStream(SystemEngine(voice=selected_voice ,print_installed_voices=False)).feed("Yes, Captain").play(language=language)

def recording_finished():
    print("Speech end detected... transcribing...")

async def jarvis_stt():
    global language
    with AudioToTextRecorder(spinner=False, 
                            model="small",   #'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
                            language=language, 
                            wake_words="jarvis", 
                            on_wakeword_detected=say_hello, 
                            on_recording_stop=recording_finished,
                            enable_realtime_transcription=False,
                            ) as recorder:
        input_voice = await asyncio.to_thread(recorder.text)
        print(input_voice)
        return input_voice

async def jarvis_tts(input_txt):
    global language
    def dummy_generator(input_txt):
        yield input_txt
        print(input_txt)
    tts = TextToAudioStream(SystemEngine(voice=selected_voice, print_installed_voices=False))
    # Assuming feed and play are blocking calls, run them in a separate thread
    await asyncio.to_thread(tts.feed, dummy_generator(input_txt))
    await asyncio.to_thread(tts.play, language=language)
##### [End] 공통함수 + STT + TTS #############################################################

##### [Start] Re-ranking Function ########################################################
def re_rank_documents(re_rank, docs, query):
    meta_list = []
    for doc in docs:
        meta_list.append(doc.metadata)
    if re_rank:
        model_path = "./models/cross_encoder"
        cross_encoder = CrossEncoder(model_name=model_path, max_length=512) # device="cpu"  cross-encoder/ms-marco-TinyBERT-L-2
        docs = cross_encoder.rank(
                query,
                [doc.page_content for doc in docs],
                top_k=3,
                return_documents=True,)
        print(docs)
        corpus_ids = []
        for doc in docs:
            corpus_ids.append(doc["corpus_id"]) 
        print("+"*50)
        print(f"re-ranked docs ids : {corpus_ids}")
        print("+"*50)
        rearranged_docs = []
        for idx, rr_reranked_doc in zip(corpus_ids, docs):
            result = Document(
                page_content=rr_reranked_doc["text"],
                metadata = meta_list[idx]
                )
            rearranged_docs.append(result)
        return rearranged_docs
    else: pass
##### [End] Re-ranking Function ########################################################

########[Start] Chatbot Functions ####################################################################################
from functools import partial
def create_prompt(template, **kwargs):
    return template.format(**kwargs)

async def jarvis_chat(template, llm_name, input_voice):
    create_greeting_prompt = partial(create_prompt, template)
    prompt = create_greeting_prompt(query=input_voice)
    llm = ChatOllama(model=llm_name)
    prompt = ChatPromptTemplate.from_template(prompt)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = await asyncio.to_thread(chain.invoke, query)
    return sentence
########[End] Chatbot Functions ####################################################################################

########## [Start] Rag Functions ########################################################################################
async def jarvis_rag(custom_template, model_name, query, temperature, top_k, top_p, doc=None, compress=False, re_rank=False, multi_q=False):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model=model_name, temperature=temperature, top_k=top_k, top_p=top_p)
    vectorstore = Chroma(persist_directory="vector_index", embedding_function=embed_model)
    if doc == None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
    else:
        # retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {"keywords":doc}}) 
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {"keywords": {'$in': doc}}}) 
        # retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {'$or': [{"keywords":"FWG"}, {"keywords":"ISS"}]}}) 
    docs = await asyncio.to_thread(retriever.invoke, query)
    print(f"Number of Base Retrieval Docs: {len(docs)}")
    #### Multi Query ############################################################################
    # if multi_q: 
    #     print("proceed Multi-Query")
    #     retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, include_original=True)
    #     print(retriever)
    # else: pass

    if multi_q:
        print("proceed Multi-Query")
        prompt = PromptTemplate.from_template(
            """You are an AI language model assistant. 
        Your task is to generate five different versions of the given user question, including given user question, to retrieve relevant documents from a vector database. 
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
        Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`

        #ORIGINAL QUESTION: 
        {question}
        """
        )
        chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        multi_queries = chain.invoke({"question": query})
        print(f"multi_queries: {multi_queries}")

        retriever = MultiQueryRetriever.from_llm(
            llm=chain, retriever=vectorstore.as_retriever()
        )
        docs = retriever.invoke(input=query)
        print(f"Multi Query Retrieval Docs : {len(docs)}")
    else: pass
    ###############################################################################################
    ##### Contextual Compressor ##################################################################
    if compress:
        print("proceed Contextual Compressor")
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        docs = await asyncio.to_thread(compression_retriever.invoke, query)   # compression_retriever.invoke(query)
        print(f"Number of Contextual Compressor Retrieval Docs: {len(docs)}")
    else: pass
    ############################################################################################
    ###### Re Ranking ##############################################################
    if re_rank: 
        print("proceed Re-Ranking")
        docs = re_rank_documents(re_rank, docs, query)
        print(f"Number of Re-Ranking Retrieval Docs: {len(docs)}")
    else: pass
    ############################################################################
    question_answering_prompt = ChatPromptTemplate.from_messages(
                [("system",
                    custom_template,),
                    MessagesPlaceholder(variable_name="messages"),
                    ])
    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    result = await asyncio.to_thread(
        document_chain.invoke,
        {
            "context": docs,
            "messages": [
                HumanMessage(content=query)
            ],
        }
    )
    return docs, result
########## [End] Rag Functions ########################################################################################

########## [Start] Rag with History Functions ########################################################################################
store = {}
def jarvis_rag_with_history(custom_template, model_name, query, temperature, top_k, top_p, history_key, doc=None, compress=False, re_rank=False, multi_q=False):
    global store
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model=model_name, temperature=temperature, top_k=top_k, top_p=top_p)

    vectorstore = Chroma(persist_directory="vector_index", embedding_function=embed_model)
    if doc == None: retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
    else: retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {"keywords": {'$in': doc}}})  
    retrieved_docs = retriever.invoke(query)
    print(f"Number of Base Retrieval Docs: {len(retrieved_docs)}")
    ######## Multi Query ##############################################################
    if multi_q:
        print("proceed Multi-Query")
        prompt = PromptTemplate.from_template(
            """You are an AI language model assistant. 
        Your task is to generate five different versions of the given user question, including given user question, to retrieve relevant documents from a vector database. 
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
        Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`

        #ORIGINAL QUESTION: 
        {question}
        """
        )
        chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        multi_queries = chain.invoke({"question": query})
        print(f"multi_queries: {multi_queries}")

        retriever = MultiQueryRetriever.from_llm(
            llm=chain, retriever=vectorstore.as_retriever()
        )
        docs = retriever.invoke(input=query)
        print(f"Multi Query Retrieval Docs : {len(docs)}")
    else: pass
    ###############################################################################################
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
        )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", custom_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",)
    
    ##### Contextual Compressor ##################################################################
    if compress:
        print("proceed Contextual Compressor")
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        retrieved_docs = compression_retriever.invoke(query)   
        print(f"Number of Contextual Compressor Retrieval Docs: {len(retrieved_docs)}")
    else: pass
    ############################################################################################

    #### Re-rank documents ########################################################################
    if re_rank: 
        print("proceed Re-Ranking")
        retrieved_docs = re_rank_documents(re_rank, retrieved_docs, query)
        print(f"Number of Re-Ranking Retrieval Docs: {len(retrieved_docs)}")
    else: pass
    ###############################################################################################

    result = conversational_rag_chain.invoke(
        {"input": query, "retrieved_docs": retrieved_docs},
        config={
            "configurable": {"session_id": history_key}
            }, )
    result["retrieved_docs"] = retrieved_docs
    result["chat_history"] = store[history_key]
    return result, store

async def async_jarvis_rag_with_history(*args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, jarvis_rag_with_history, *args, **kwargs)
########## [End] Rag with History Functions ########################################################################################


############[Start] Translation #########################################################################################################
import asyncio
from aiogoogletrans import Translator

async def trans(txt, input_lang, target_lang):
    trans = Translator()
    result = await trans.translate(txt, src=input_lang, dest=target_lang)
    return result.text

async def trans_main(txt):
    input_lang = 'en'
    target_langs = ['ko']
    translations = await asyncio.gather(*[trans(txt, input_lang, target_lang) for target_lang in target_langs])
    return translations
############[End] Translation #########################################################################################################

############## [Start] Schemas ############################################################################################################
from pydantic import BaseModel
from typing import Optional

class OllamaRequest(BaseModel):
    template: str
    llm_name: str
    input_voice: str

class RagOllamaRequest(BaseModel):
    template: str
    llm_name: str
    input_voice: str
    temperature: float
    top_k: int
    top_p: float
    doc: Optional[list]
    compress: bool
    re_rank: bool
    multi_q: bool

class RagOllamaRequestHistory(BaseModel):
    template: str
    llm_name: str
    input_voice: str
    temperature: float
    top_k: int
    top_p: float
    history_key: int
    doc: Optional[list]
    compress: bool
    re_rank: bool
    multi_q: bool

class TTSRequest(BaseModel):
    output: str

class TRANSRequest(BaseModel):
    txt: str
############## [End] Schemas ############################################################################################################

###### [Start] FastAPI Endpoint ###########################################################################################################
from fastapi import FastAPI, Response
import json
import asyncio
import logging

app = FastAPI(
    title="HD-CoPilot",
    description="Local RAG AI Agent without Internet",
    version="0.0")

### [Start] Configure logging #######################
logging.basicConfig(level=logging.DEBUG, filename='jarvis_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
'''
** log levels
- DEBUG: Use this level for detailed information useful for debugging purposes. It’s like wearing your detective hat and delving deep into the inner workings of your application.
- INFO: This level is perfect for general information about what’s happening within the application. Think of it as your application’s way of saying, “Hey, everything’s running smoothly!”
- WARNING: Uh-oh, something doesn’t seem quite right. Use this level to indicate potential issues that could lead to problems down the road.
- ERROR: Houston, we have a problem! Use this level to signify errors that need immediate attention but won’t necessarily crash the application.
- CRITICAL: Brace yourselves; things are about to hit the fan! Reserve this level for critical errors that could potentially bring your entire application crashing down.

** parameters
- level=logging.DEBUG: This sets the logging level to DEBUG, meaning all log messages will be captured.
- filename='app.log': This specifies the name of the log file. You can choose any name you like.
- filemode='a': This sets the file mode to append, so new log messages will be added to the end of the file.
- format='%(asctime)s - %(levelname)s - %(message)s': This defines the format of the log messages, including the timestamp, log level, and message.
'''
@app.get("/")
async def read_root():
    logger.debug("Root endpoint accessed")
    logger.info("Testing Info")
    logger.warn("Testing Warning")
    logger.error("Testing Error")
    return {"message": "Hello World"}

#### [End] Configure logging ########################################

@app.post("/jarvis_stt")
async def call_jarvis_stt():
    logger.info(f"Jarvis STT requested")
    res = await jarvis_stt()
    result = {"input_voice": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/jarvis_tts")
async def call_jarvis_tts(request: TTSRequest):
    logger.info(f"Jarvis TTS requested")
    await jarvis_tts(request.output)
    return {"status": "completed"}

@app.post("/jarvis_trans")
async def jarvis_trans_main(request: TRANSRequest):
    logger.info(f"Google Trans requested")
    res = await trans_main(request.txt)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_jarvis")
async def call_jarvis_chat(request: OllamaRequest):
    logger.info(f"Common Chat - {request.input_voice} requested")
    res = await jarvis_chat(request.template, request.llm_name, request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_rag_jarvis")
async def call_jarvis_rag(request: RagOllamaRequest):
    logger.info(f"RAG without History - {request.input_voice} requested")
    res = await jarvis_rag(request.template, request.llm_name, request.input_voice, request.temperature, request.top_k, request.top_p, request.doc, request.compress, request.re_rank, request.multi_q)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_rag_jarvis_with_history")
async def call_jarvis_rag_with_history(request: RagOllamaRequestHistory):
    logger.info(f"RAG with History - {request.input_voice} requested")
    res = await async_jarvis_rag_with_history(request.template, request.llm_name, request.input_voice, request.temperature, request.top_k, request.top_p, request.history_key, request.doc, request.compress, request.re_rank, request.multi_q)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')
###### [End] FastAPI Endpoint ###########################################################################################################


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)


