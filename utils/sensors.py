'''
정해진 테이블에서 값을 불러오는 것만 가능
'''

import pandas as pd

df = pd.read_csv("../data/sensor_data.csv")
df = df.astype(str)
print(df.head(2))


from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering, pipeline
google_tapes_path = "D:/ai_jarvis/models/google_tapas"

model = AutoModelForTableQuestionAnswering.from_pretrained(pretrained_model_name_or_path=google_tapes_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=google_tapes_path)
nlp = pipeline('table-question-answering', model=model, tokenizer=tokenizer)

question_list = [
"what date shows the highest temperature and its temperature",
"what date shows the lowest temperature",
"what is the temperature on date 2024-03-07",
"what is the engine pressure on date 2024-03-07",
"what date shows the highest engine pressure",
]

for question in question_list:
    result = nlp({'table': df, 'query': question})
    print(question)
    print(result['cells'][0].strip())
    print("-"*50)