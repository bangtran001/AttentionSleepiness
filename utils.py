# import numpy as np
# from numpy import random
# from scipy.special import softmax
#
# # encoder representations of four different words
# word_1 = np.array([1, 0, 0])
# word_2 = np.array([0, 1, 0])
# word_3 = np.array([1, 1, 0])
# word_4 = np.array([0, 0, 1])
#
# # stacking the word embeddings into a single array
# words = np.array([word_1, word_2, word_3, word_4])
#
# # generating the weight matrices
# random.seed(42)
# W_Q = random.randint(3, size=(3, 3))
# W_K = random.randint(3, size=(3, 3))
# W_V = random.randint(3, size=(3, 3))
#
# # generating the queries, keys and values
# Q = words @ W_Q
# K = words @ W_K
# V = words @ W_V
#
# # scoring the query vectors against all key vectors
# scores = Q @ K.transpose()
#
# # computing the weights by a softmax operation
# weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
#
# # computing the attention by a weighted sum of the value vectors
# attention = weights @ V
#
# print(attention)
#
# exit(0)


# ----------------
# GENERAL VARIABLES
# ----------------
hubert_embedding_dir = "../../ICASSP2022/pickle/hubert-embedding"  # --> embedding [1, 1024]
hubert_feature_dir = "../../ICASSP2022/pickle/hubert-feature"      # --> features [T, 1024]


corpus_file = "csv/corpus.csv"

# this dictionary maps ressponseX --> task X
response_task_map = {}
for i in range(1, 10):      # task1 - task10
    response_task_map['response' + str(i)] = i
for i in range(10, 35):     #  task 10 (Confrontational naming)
    response_task_map['response' + str(i)] = 10
for i in range(35, 45):     # task 11 (non-word)
    response_task_map['response' + str(i)] = 11
response_task_map['response46'] = 12    # task 12 (sentence repeat)
response_task_map['response48'] = 12

# this is list of audio responses
audio_responses = ['response' + str(i) for i in range(1, 49)]
audio_responses.remove('response45')
audio_responses.remove('response47')

#-----------------

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence


'''
This function read corpus.csv file and generate csv file which containing columns:
    1. session_id: The id of session that the audio belongs to
    3. responseX | reponseY: the HuBERT embedding/feature file (numpy format)
    4. sss_label:  The self-assessment of sleepiness   

    @:selected_task: The specific task that we want to generate csv file (int: 1-12)
    @:csv_output: The csv output file name 
'''
def read_corpus(selected_task = None, csv_output='dataset.csv'):
    df = pd.read_csv(corpus_file)  # load the csv file
    # the response columns of selected task
    if selected_task != 0:
        responses = [k for k, v in response_task_map.items() if v == selected_task]
    else:
        responses = list(response_task_map.keys())
    # dropping unused columns
    for r in df.columns:
        if r not in responses + ['session_id', 'sss']:
            df.drop(r, axis=1, inplace=True)
    # dropping sessions containing empty wav_filename
    df.dropna(axis=0, how='any', inplace=True)
    df.reset_index(inplace=True)

    # determine sessions do not have numpy embedding
    for s_index in range(len(df['session_id'])):
        for r in responses:
            wav_file = df.loc[s_index, r]
            wav_file = os.path.splitext(wav_file)
            fn = wav_file[0] + '.npy'
            fn = os.path.join(hubert_embedding_dir, fn)
            if not os.path.exists(fn):          # doesn't have npy_filename
                df.loc[s_index, r] = None
            else:
                df.loc[s_index, r] = wav_file[0] + '.npy'
    # dropping sessions containing empty numpy_file
    df.dropna(axis=0, how='any', inplace=True)
    df.to_csv('csv/'+csv_output, index=False)

# for i in tqdm(range(0, 13), desc='Generating'):
#     read_corpus(selected_task=i, csv_output='task'+str(i)+'.csv')
# --------------------



'''
Sleepiness Dataset
'''
from torch.utils.data import Dataset, DataLoader, random_split
class SleepinessDataset(Dataset):
    def __init__(self, device, selected_task = 1):
        if selected_task not in range(0, 13):
            raise Exception("Invalide task selected!")

        self.FEATURE_TYPE = 'Embedding' # 'Embedding' or full hubert feature 'FullFeature'
        self.device = device
        self.selected_task = selected_task   # All=0 | 1 | 2 | ...| 12
        self.dataset = pd.read_csv('csv/task'+str(selected_task)+'.csv')

    def __len__(self):
        return len(self.dataset)

    # input:    index of session
    # output:   an 2D array of (N, 1024) where N is number of responses in the selected task <--- if using HuBERT embedding
    #           an 3D array of (N, T, 1024) where N is number of responses, T is maximal length of the response's feature <--- if using HuBERT features
    def __getitem__(self, idx):
        # the response columns of selected task
        if self.selected_task != 0:
            responses = [k for k, v in response_task_map.items() if v == self.selected_task]
        else:
            responses = list(response_task_map.keys())

        features = list()
        for r in responses:
            np_file = self.dataset.loc[idx, r]
            if self.FEATURE_TYPE == 'Embedding':
                feat = np.load(os.path.join(hubert_embedding_dir, np_file))       # return (1024)
            else:
                feat = np.load(os.path.join(hubert_feature_dir, np_file))         # return (T, 1024)

            feat = torch.from_numpy(feat)
            if self.FEATURE_TYPE == 'Embedding':
                feat = torch.unsqueeze(feat, 0)  # (1024) -> (1, 1024)
            features.append(feat)

        if self.FEATURE_TYPE == 'Embedding':
            pt_features = torch.cat(features)       # return (N, 1024)
        else:
            pt_features = pad_sequence(features)    # return (MaxT, N, 1024)
            pt_features = torch.permute(pt_features, (1, 0, 2))

        sss_label = self.dataset.loc[idx, 'sss']    # sleepiness is a number --> convert to one-hot-vector
        #onehot_label = torch.zeros(1, 7)
        #onehot_label[0, int(sss_label - 1)] = 1
        lb = 0 if int(sss_label) <= 3 else 1         # 1-3 is awake; 4-7 is sleepy

        return pt_features, lb


    # The feature has shape of [T x 1024]. This funciton will help to determine the largest value of T
    # The current longest T = 3315
    # warning: This takes 22mins to finish
    def get_longest_feature_length(self):
        max_T = 0
        for i in tqdm(range(0, len(self.dataset)), desc='Calculating...'):
            feat, _ = self.__getitem__(i)
            if feat.size(0) > max_T:
                max_T = feat.size(0)
                print(f"maxlen = {max_T}")
        return max_T


# ds = SleepinessDataset('cuda:0', selected_task=10)
# X1, y1 = ds.__getitem__(90)
# X2, y2 = ds.__getitem__(33)
#
# print(X1.size(), y1)
# print(X2.size(), y2)

'''
There are error numpy files when it generated
    - embedding doesn't have shape [1, 1024]
    - features doesn't have shape [N, 1024]    
'''
# for f in tqdm(os.listdir(hubert_feature_dir), desc='Deleting error numpy files'):
#     xx = np.load(hubert_feature_dir + '/'+f)
#     x = torch.from_numpy(xx)
#     x = torch.unsqueeze(x, dim=0)
#     if x.size(-1) != 1024:
#         print(f, x.size(), ' is deleted')
#         #os.remove(hubert_feature_dir + '/'+f)
#

# re-generating embedding from feature files
# feature_files = os.listdir(hubert_feature_dir)
# embedding_files = os.listdir(hubert_embedding_dir)
# for f in tqdm(feature_files):
#     feat = np.load(hubert_feature_dir + '/' + f)
#     print(feat.shape)
#     break

