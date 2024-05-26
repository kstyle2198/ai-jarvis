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

async def jarvis_chat(custom_template, llm_name, input_voice):

    template = custom_template
    create_greeting_prompt = partial(create_prompt, template)
    prompt = create_greeting_prompt(query=input_voice)

    llm = ChatOllama(model=llm_name)
    prompt = ChatPromptTemplate.from_template(prompt) #(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    
    sentence = await asyncio.to_thread(chain.invoke, query)
    return sentence


async def jarvis_rag(custom_template, model_name, query):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="vector_index", embedding_function=embed_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = await asyncio.to_thread(retriever.invoke, query)

    llm = ChatOllama(model=model_name)
    SYSTEM_TEMPLATE = custom_template
    question_answering_prompt = ChatPromptTemplate.from_messages(
                [("system",
                    SYSTEM_TEMPLATE,),
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
    template: str
    llm_name: str
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
async def call_jarvis_rag(request: OllamaRequest):
    res = await jarvis_rag(request.template, request.llm_name, request.input_voice)
    result = {"output": res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)


