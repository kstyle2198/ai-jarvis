from fastapi import FastAPI, Response
import uvicorn
import os
import json

from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine
from RealtimeSTT import AudioToTextRecorder
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings



import os
groq_api_key = os.environ['GROQ_API_KEY']

selected_voice = "David"
language = "en"

custom_template = '''
you are an smart AI assistant in a commercial vessel like LNG Carriers or Container Carriers.
your answer always starts with "OK, Master".
generate compact and summarized answer to {query} kindly and shortly.
if there are not enough information to generate answers, just return "Please give me more information"
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?'
'''



def say_hello():
    TextToAudioStream(SystemEngine(voice=selected_voice ,print_installed_voices=False)).feed("Yes, Master").play(language="en")

def recording_finished():
    print("Speech end detected... transcribing...")


async def jarvis_stt():
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

async def jarvis_chat(llm_name, input_voice):
    global custom_template
    llm = ChatOllama(model=llm_name)
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    
    # Assuming invoke is a blocking operation, use asyncio.to_thread
    sentence = await asyncio.to_thread(chain.invoke, query)
    return sentence


async def jarvis_groq_llama3_8b(input_voice):
    global custom_template
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()

    # Assuming invoke method is blocking, use asyncio.to_thread to run it in a separate thread
    sentence = await asyncio.to_thread(chain.invoke, query)
    return sentence



async def jarvis_rag(model_name, query):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="test_index", embedding_function=embed_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Assuming invoke method is blocking, use asyncio.to_thread to run it in a separate thread
    docs = await asyncio.to_thread(retriever.invoke, query)

    llm = ChatOllama(model=model_name)
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

    # Assuming invoke method is blocking, use asyncio.to_thread to run it in a separate thread
    result = await asyncio.to_thread(
        document_chain.invoke,
        {
            "context": docs,
            "messages": [
                HumanMessage(content=query)
            ],
        }
    )
    return result

async def jarvis_rag_groq_llama3(query):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="test_index", embedding_function=embed_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Assuming invoke method is blocking, use asyncio.to_thread to run it in a separate thread
    docs = await asyncio.to_thread(retriever.invoke, query)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')
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

    # Assuming invoke method is blocking, use asyncio.to_thread to run it in a separate thread
    result = await asyncio.to_thread(
        document_chain.invoke,
        {
            "context": docs,
            "messages": [
                HumanMessage(content=query)
            ],
        }
    )
    return result

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
from fastapi import FastAPI, Request, Response
import json
from pydantic import BaseModel
import asyncio

app = FastAPI()

class OllamaRequest(BaseModel):
    llm_name: str
    input_voice: str

class GroqRequest(BaseModel):
    input_voice: str

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

@app.post("/call_jarvis")
async def call_jarvis_chat(request: OllamaRequest):
    res = await jarvis_chat(request.llm_name, request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_groq_llama3")
async def call_groq_llama3_8b(request: GroqRequest):
    res = await jarvis_groq_llama3_8b(request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_rag_jarvis")
async def call_jarvis_rag(request: OllamaRequest):
    res = await jarvis_rag(request.llm_name, request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_rag_groq_llama3")
async def call_rag_groq_llama3(request: GroqRequest):
    res = await jarvis_rag_groq_llama3(request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/call_trans")
async def call_trans_main(request: TRANSRequest):
    res = await trans_main(request.txt)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)


