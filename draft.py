import pandas as pd
import os
import re
import numpy as np
import torchaudio
from transformers import Wav2Vec2Processor, TFHubertModel

raw_audio_dir = '/media/data_sdh/HuBERT-Feature/new2.csv'

# the main CSV file from SondeHealth
df = pd.read_csv('/media/data_sdh/HuBERT-Feature/new2.csv')

# determine the columns of audio responses
column_indices = [0] # session_id
column_indices = column_indices + [i for i in range(3, 53)]  # audio responses

# there're some empty cells within response columns --> Replace these empty cells by '--'; otherwise just change it to '.wav'
for i in range(len(df)):
    for j in column_indices[1:]:
        if pd.isnull(df.iloc[i, j]):
            df.iloc[i, j] = '--'
        else:
            re_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
            x = df.iloc[i, j]
            df.iloc[i, j] = re.findall(re_pattern, x)[0] + '.wav'

# generating .npy files
for i in range(len(df)):
    session_id = df.iloc[0, 1]
    if
    for j in column_indices[1:]:





# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
# hubert_model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
#

#column_names =['session_id'] + ['response' + str(i) for i in range(1, 51)]  # session_id, response1,...,response50







#
#
# for col in column_indices[2:]:  # skip column 'session_id'
#     responses = df[col]
#     re_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
#     wav_files = list(
#         map(lambda x: (re.findall(re_pattern, x)[0] + '.wav') if type(x) is str else '__', responses)
#     )
#     df[col] = wav_files
#
#
#
#

#
# response_tasks_map = {}
# for i in range(1, 10):      # task1 - task10
#     response_tasks_map['response' + str(i)] = i
# for i in range(10, 35):     #  task 10 (Confrontational naming)
#     response_tasks_map['response' + str(i)] = 10
# for i in range(35, 45):     # task 11 (non-word)
#     response_tasks_map['response' + str(i)] = 11
# response_tasks_map['response46'] = 12    # task 12 (sentence repeat)
# response_tasks_map['response48'] = 12
#
#
#
#
# for sess in sessions:
#     for r in list(response_tasks_map.keys()):
#
# features = dt_generator.generate_training_features(hubert_processor=processor, hubert_model=hubert_model)