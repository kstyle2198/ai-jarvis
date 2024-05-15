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

def tts_generator(input_txt, language=language):
    def dummy_generator(input_txt):
        yield input_txt
        print(input_txt)
    TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=False)).feed(dummy_generator(input_txt)).play(language=language)

def jarvis_stt():
    
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
        print(recorder.text(), end=" \n", flush=True)
        input_voice = recorder.transcribe()
        return input_voice
    
def jarvis_tts(sentence):
    tts_generator(sentence, language=language)

def jarvis_tinydophin(input_voice):
    global custom_template
    llm = ChatOllama(model="tinydolphin:latest")
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = chain.invoke(query)
    return sentence
        
def jarvis_moondream(input_voice):
    global custom_template
    llm = ChatOllama(model="moondream:latest")
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = chain.invoke(query)
    return sentence

def jarvis_dophin_phi(input_voice):
    global custom_template
    llm = ChatOllama(model="dolphin-phi:latest")
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = chain.invoke(query)
    return sentence

def jarvis_ph3_4b(input_voice):
    global custom_template
    llm = ChatOllama(model="phi3:latest")
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = chain.invoke(query)
    return sentence

def jarvis_llama3_8b(input_voice):
    global custom_template
    llm = ChatOllama(model="llama3:latest")
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = chain.invoke(query)
    return sentence

def jarvis_groq_llama3_8b(input_voice):
    global custom_template
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')
    prompt = ChatPromptTemplate.from_template(custom_template)
    query = {"query": input_voice}
    chain = prompt | llm | StrOutputParser()
    sentence = chain.invoke(query)
    return sentence


def jarvis_rag_tinydolphin(query):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="test_index", embedding_function = embed_model)
    retirever = vectordb.as_retriever(search_kwargs = {"k" : 3})
    docs = retirever.invoke(query)

    llm = ChatOllama(model="tinydolphin:latest")
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
    result = document_chain.invoke(
            {
                "context": docs,
                "messages": [
                    HumanMessage(content=query)
                ],
            }
        )
    return result

def jarvis_rag_groq_llama3(query):
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="test_index", embedding_function = embed_model)
    retirever = vectordb.as_retriever(search_kwargs = {"k" : 3})
    docs = retirever.invoke(query)

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
    result = document_chain.invoke(
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
app = FastAPI()


@app.get("/jarvis_stt")
def call_jarvis_stt():
    res = jarvis_stt()
    result = {"input_voice":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/jarvis_tts")
def call_jarvis_tts(output):
    res = jarvis_tts(output)
    return res

@app.get("/call_tinydolphin")
def call_tinydolphin(input_voice):
    res = jarvis_tinydophin(input_voice)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_moondream")
def call_moondream(input_voice):
    res = jarvis_moondream(input_voice)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_dolphinphi")
def call_dophin_phi(input_voice):
    res = jarvis_dophin_phi(input_voice)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_phi3")
def call_phi3_4b(input_voice):
    res = jarvis_ph3_4b(input_voice)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_llama3")
def call_llama3_8b(input_voice):
    res = jarvis_llama3_8b(input_voice)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_groq_llama3")
def call_groq_llama3_8b(input_voice):
    res = jarvis_groq_llama3_8b(input_voice)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')



@app.get("/call_rag_tinydolphin")
def call_rag_tinydolphin(query):
    res = jarvis_rag_tinydolphin(query)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_rag_groq_llama3")
def call_rag_groq_llama3(query):
    res = jarvis_rag_groq_llama3(query)
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get("/call_trans")
def call_trans_main(txt):
    res = asyncio.run(trans_main(txt))
    result = {"output":res}
    json_str = json.dumps(result, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)


