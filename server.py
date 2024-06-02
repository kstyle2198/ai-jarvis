from fastapi import FastAPI, Response
import uvicorn
import os
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
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory



selected_voice = "David"   # David, Hazel
language = "en"


def say_hello():
    TextToAudioStream(SystemEngine(voice=selected_voice ,print_installed_voices=False)).feed("Yes, Master").play(language="en")

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
        ready_msg = 'Say "Jarvis" then speak.'
        print(ready_msg)
        
        # Assuming recorder.text() is a blocking call, run it in a separate thread
        input_voice = await asyncio.to_thread(recorder.text)
        print(input_voice)
        return input_voice

async def jarvis_tts(input_txt, language=language):
    def dummy_generator(input_txt):
        yield input_txt
        print(input_txt)
    
    tts = TextToAudioStream(SystemEngine(voice=selected_voice, print_installed_voices=False))
    
    # Assuming feed and play are blocking calls, run them in a separate thread
    await asyncio.to_thread(tts.feed, dummy_generator(input_txt))
    await asyncio.to_thread(tts.play, language=language)


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



async def jarvis_rag(custom_template, model_name, query, temperature, top_k, top_p, doc=None):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="vector_index", embedding_function=embed_model)
    if doc == None:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3}) 
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3, "filter": {"keywords":doc}}) 
        # retriever = vectordb.as_retriever(search_kwargs={"k": 3, "filter": {"keywords": {'$in': ["Unit_Cooler", "FWG"]}}}) 
        # retriever = vectordb.as_retriever(search_kwargs={"k": 3, "filter": {'$or': [{"keywords":"FWG"}, {"keywords":"ISS"}]}}) 

    docs = await asyncio.to_thread(retriever.invoke, query)

    llm = ChatOllama(model=model_name, temperature=temperature, top_k=top_k, top_p=top_p)
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


store = {}
def jarvis_rag_with_history(custom_template, model_name, query, temperature, top_k, top_p, history_key, doc=None):
    global store
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model=model_name, temperature=temperature, top_k=top_k, top_p=top_p)
    vectorstore = Chroma(persist_directory="vector_index", embedding_function=embed_model)
    if doc == None:
        retriever = vectorstore.as_retriever() 
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"keywords":doc}}) 

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

    result = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": history_key}
            },  
            )
    
    result["chat_history"] = store[history_key]

    return result, store

async def async_jarvis_rag_with_history(*args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, jarvis_rag_with_history, *args, **kwargs)


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

##########################################################################################################################################
from fastapi import FastAPI, Response
import json
from pydantic import BaseModel
import asyncio
from typing import Optional

app = FastAPI(
    title="AI-Jarvis",
    description="Local RAG Chatbot without Internet",
    version="0.0"
    )

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
    doc: Optional[str]

class RagOllamaRequestHistory(BaseModel):
    template: str
    llm_name: str
    input_voice: str
    temperature: float
    top_k: int
    top_p: float
    history_key: int
    doc: Optional[str]

class TTSRequest(BaseModel):
    output: str

class TRANSRequest(BaseModel):
    txt: str

@app.post("/jarvis_stt")
async def call_jarvis_stt():
    res = await jarvis_stt()
    result = {"input_voice": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/jarvis_tts")
async def call_jarvis_tts(request: TTSRequest):
    await jarvis_tts(request.output)
    return {"status": "completed"}

@app.post("/jarvis_trans")
async def jarvis_trans_main(request: TRANSRequest):
    res = await trans_main(request.txt)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_jarvis")
async def call_jarvis_chat(request: OllamaRequest):
    res = await jarvis_chat(request.template, request.llm_name, request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_rag_jarvis")
async def call_jarvis_rag(request: RagOllamaRequest):
    res = await jarvis_rag(request.template, request.llm_name, request.input_voice, request.temperature, request.top_k, request.top_p, request.doc)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_rag_jarvis_with_history")
async def call_jarvis_rag_with_history(request: RagOllamaRequestHistory):
    res = await async_jarvis_rag_with_history(request.template, request.llm_name, request.input_voice, request.temperature, request.top_k, request.top_p, request.history_key, request.doc)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)


