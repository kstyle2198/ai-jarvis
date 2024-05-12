from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine
import logging


def tts_generator(input_txt, selected_voice="David", language='en'):
    def dummy_generator(input_txt):
        yield input_txt
    TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=True)).feed(dummy_generator(input_txt)).play(language=language)

if __name__ == '__main__':
    # recorder = AudioToTextRecorder(spinner=False, model="tiny.en", language="en") 
    # print("Say something...")
    # while (True): print(recorder.text(), end=" ", flush=True)


    def recording_started():
        print("Speak now...\n")

    selected_voice = "David"
    
    def say_hello():
        TextToAudioStream(SystemEngine(voice=selected_voice,print_installed_voices=False)).feed("Yes, Master").play(language='en')


    def recording_finished():
        print("Speech end detected... transcribing...")


    with AudioToTextRecorder(spinner=False, 
                            #  level=logging.DEBUG, 
                            model="small.en", 
                            language="en", 
                            wake_words="jarvis", 
                            on_wakeword_detected=say_hello, 
                            # on_recording_start=say_hello,
                            on_recording_stop=recording_finished
                            ) as recorder:
        print('Say "Jarvis" then speak.')
        print(recorder.text(), end=" \n", flush=True)


