from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
import time
import uvicorn
import os
import json

import logging
from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine
from RealtimeSTT import AudioToTextRecorder
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


selected_voice = "David"
language = "ko"


def stream_data(answer):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.1)

def recording_started():
    print("recording_started now...")

def recording_finished():
    print("Speech end detected... transcribing...")

def tts_generator(input_txt, language=language):
    def dummy_generator(input_txt):
        yield input_txt
        print(input_txt)
    TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=False)).feed(dummy_generator(input_txt)).play(language=language)

def say_hello():
    TextToAudioStream(SystemEngine(voice=selected_voice ,print_installed_voices=False)).feed("Yes, Master").play(language="en")

import os
groq_api_key = os.environ['GROQ_API_KEY']


custom_template = '''
you are an smart AI assistant in a commercial vessel like LNG Carriers or Container Carriers.
your answer always starts with "OK, Master".
generate compact and summarized answer to {query} kindly and shortly.
if there are not enough information to generate answers, just return "Please give me more information"
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?'
'''


def jarvis_stt():
    with AudioToTextRecorder(spinner=False, 
                            model="small",   #'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
                            language=language, 
                            wake_words="jarvis", 
                            on_wakeword_detected=say_hello, 
                            on_recording_stop=recording_finished,
                            enable_realtime_transcription=False,
                            ) as recorder:
        print('Say "Jarvis" then speak.')
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


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)


