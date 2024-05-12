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


    # input_txt = '''
    # 1회 톱타자 박민우를 볼넷으로 내보내더니, 2사 2루에서 데이비슨에게 좌측 담장을 넘어가는 투런 홈런을 허용했다. 
    # 이후 권희동과 서호철에게 연속 안타를 맞더니 2사 1,2루에서 김성욱에게 3볼-1스트라이크에서 스리런 홈런을 허용했다.
    # '''
    # selected_voice = "Heami"
    # language = 'ko'


    # tts_generator(input_txt, selected_voice, language)

    # import logging
    # logging.basicConfig(level=logging.INFO)    
    # engine = CoquiEngine(level=logging.INFO)
    # def dummy_generator():
    #     yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "

    # stream = TextToAudioStream(engine)
    
    # print ("Starting to play stream")
    # stream.feed(dummy_generator()).play(log_synthesized_text=True)

    # engine.shutdown()

    tts_generator_coqui(input_txt, language)





