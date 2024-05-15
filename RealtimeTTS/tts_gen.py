from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine

def tts_generator(input_txt, selected_voice, language):
    def dummy_generator(input_txt):
        yield input_txt
        print(input_txt)
    TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=False)).feed(dummy_generator(input_txt)).play(language=language)

'''
[Microsoft Zira Desktop - English (United States), 
Microsoft David Desktop - English (United States), 
Microsoft Hazel Desktop - English (Great Britain), 
Microsoft Haruka Desktop - Japanese, 
Microsoft Heami Desktop - Korean, 
Microsoft Huihui Desktop - Chinese (Simplified)]
'''

def tts_generator_coqui(input_txt, language):
    engine = CoquiEngine(voices_path="D:/ai_jarvis/RealtimeTTS/coqui_voices", 
                         voice ='ks_jung',
                         speed = 1,
                         thread_count = 12, 
                         stream_chunk_size = 40,
                         full_sentences = True,
                         comma_silence_duration = 0.1, 
                         sentence_silence_duration = 0.1, 
                         default_silence_duration = 0.1,
                         )
    def dummy_generator(input_txt):
        yield input_txt

    stream = TextToAudioStream(engine)
    print ("Starting to play stream")
    stream.feed(dummy_generator(input_txt)).play(log_synthesized_text=True, language=language)
    engine.shutdown()


if __name__ == "__main__":

    input_txt = "But each of vector similarity search and keyword search has its own strengths and weaknesses. Vector similarity search is good, for example, at dealing with queries that contain typos, which usually don’t change the overall intent of the sentence.However, vector similarity search is not as good at precise matching on keywords, abbreviations, and names, which can get lost in vector embeddings along with the surrounding words. Here, keyword search performs better."
    selected_voice = "David"
    language = 'en'


    input_txt = '''
    본 레포트의 분석 목적은 조선 내업의 정반공정 최적화이다. 내업 정반 공정 최적화는 조선업의 오랜 도전과제였고, 과거에도 다양한 시도가 이루어진 바 있다. 
    그럼에도 솔루션의 과도한 복잡성 및 생산현장에서 발생하는 다양하고 새로운 변수들을 고려하기 어려움으로 인해 여전히 상당 부분 담당자의 경험적인 능력치에 의존하여 업무가 수행되고 있다
    '''
    selected_voice = "Heami"
    language = 'ko'


    tts_generator(input_txt, selected_voice, language)

    # import logging
    # logging.basicConfig(level=logging.INFO)    
    # engine = CoquiEngine(level=logging.INFO)
    # def dummy_generator():
    #     yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "

    # stream = TextToAudioStream(engine)
    
    # print ("Starting to play stream")
    # stream.feed(dummy_generator()).play(log_synthesized_text=True)

    # engine.shutdown()

    # tts_generator_coqui(input_txt, language)





