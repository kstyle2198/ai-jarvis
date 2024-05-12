import logging
from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine
from RealtimeSTT import AudioToTextRecorder
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

### STT and TTS Functions ###################################################
def recording_started():
    print("recording_started now...")

def recording_finished():
    print("Speech end detected... transcribing...")

'''
[Microsoft Zira Desktop - English (United States), 
Microsoft David Desktop - English (United States), 
Microsoft Hazel Desktop - English (Great Britain), 
Microsoft Haruka Desktop - Japanese, 
Microsoft Heami Desktop - Korean, 
Microsoft Huihui Desktop - Chinese (Simplified)]
Susan, George, Mark
'''

selected_voice = "David"

def tts_generator(input_txt, selected_voice=selected_voice, language='en'):
    def dummy_generator(input_txt):
        yield input_txt
        print(input_txt)
    TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=False)).feed(dummy_generator(input_txt)).play(language=language)

def say_hello():
    TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=False)).feed("Yes, Master").play(language='en')

# def tts_generator_coqui(input_txt, language):
#     engine = CoquiEngine(voices_path="./RealtimeTTS/coqui_voices", 
#                          voice ='ks_jung',
#                          speed = 1,
#                          thread_count = 6, 
#                          stream_chunk_size = 20,
#                          full_sentences = True,
#                          comma_silence_duration = 0, 
#                          sentence_silence_duration = 0, 
#                          default_silence_duration = 0,
#                          )
#     def dummy_generator(input_txt):
#         yield input_txt
#     stream = TextToAudioStream(engine)
#     print ("Starting to play stream")
#     stream.feed(dummy_generator(input_txt)).play(log_synthesized_text=True, language=language)
#     engine.shutdown()

# llm = ChatOllama(model="llama3:latest")   # parameters 8B quantization 4-bit
# llm = ChatOllama(model="neural-chat:latest")  # parameters 7B  quantization 4-bit
# llm = ChatOllama(model="phi3:latest")     # parameters 3.8B quantization 4-bit
llm = ChatOllama(model="dolphin-phi:latest")     # parameters 2.7B quantization 4-bit
# llm = ChatOllama(model="tinyllama:latest")   # parameters 1.1B quantization 4-bit
# llm = ChatOllama(model="tinydolphin:latest")   # parameters 1.1B quantization 4-bit


import os
# groq_api_key = os.environ['GROQ_API_KEY']
# llm = ChatGroq(
#                 groq_api_key=groq_api_key, 
#                 model_name='llama3-8b-8192'
#                 )


custom_template = '''
you are an AI assistant in a company.
your answer always starts with "OK, Master".
generate compact answer to {query} kindly and shortly.
if the query does not give you enough information, return a question for additional information.
for example, 'could you give me more detailed informations about it?
'''
prompt = ChatPromptTemplate.from_template(custom_template)

if __name__ == "__main__":

    while True:
        with AudioToTextRecorder(spinner=False, 
                                #  level=logging.DEBUG, 
                                 model="tiny.en",   #'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
                                 language="en", 
                                 wake_words="jarvis", 
                                 on_wakeword_detected=say_hello, 
                                #  on_recording_start=recording_started,
                                 on_recording_stop=recording_finished,
                                 enable_realtime_transcription=False,
                                 ) as recorder:
            print('Say "Jarvis" then speak.')
            print(recorder.text(), end=" \n", flush=True)

            input_voice = recorder.transcribe()
            if input_voice:
                chain = prompt | llm | StrOutputParser()
                query = {"query": input_voice}
                sentence = chain.invoke(query)
            
                tts_generator(sentence, language='en')
                # tts_generator_coqui(sentence, language='en')
                # print(sentence)
                

