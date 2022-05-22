# ----------------
# GENERAL VARIABLES
# ----------------
hubert_embedding_dir = "../../ICASSP2022/pickle/hubert-embedding"  # --> embedding [1, 1024]
#hubert_feature_dir = "../../ICASSP2022/pickle/hubert-feature"      # --> features [T, 1024]
hubert_feature_dir = "/media/data_sdh/HuBERT-Feature/Nx1024"      # --> features [T, 1024]
corpus_file = "csv/new2.csv"

# this dictionary maps ressponseX --> task X
response_task_map = {}
for i in range(1, 10):      # task1 - task9
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
import re
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
    # the response columns of selected task
    if selected_task != 0:
        responses = [k for k, v in response_task_map.items() if v == selected_task]
    else:
        responses = list(response_task_map.keys())

    # load the original csv file
    cols = [0] # session id
    cols += [i for i in range(3, 51)]  # response samples
    cols += [79, 80, 143] # gender, age, sleepiness scale (label)
    df = pd.read_csv(corpus_file, usecols=cols)
    # change the name of columns
    col_names = ['session_id'] + ['response' + str(i) for i in range(1, 49)]  # response1,...,response48
    col_names += ['gender', 'age', 'sss']
    df.columns = col_names
    df.drop(columns=['response45', 'response47'], inplace=True)     # remove redundant columns
    # extract sleepiness level
    sss_levels = list(map(lambda x: (re.findall("\d", x)[0]), df['sss']))
    df['sss'] = sss_levels

    new_df = df.loc[:, ['session_id']+ responses +['gender', 'age', 'sss']]
    # dropping rows contain empty cell
    new_df.dropna(axis=0, how='any', inplace=True)

    # extract audio filename from responses
    for c in range(1, len(new_df.columns)-3):
        sample_ids = new_df.iloc[:, c]
        str_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
        sample_ids = list(
            map(lambda x: (re.findall(str_pattern, x)[0] + '.wav') if type(x) is str else '__', sample_ids))
        new_df.iloc[:, c] = sample_ids

    # determine session that doesn't have numpy embedding
    for idx in tqdm(new_df.index, total=len(new_df)):
        session_id = new_df.at[idx, 'session_id']
        #tqdm.write(session_id)
        for r in responses:
            #print(idx, session_id,r )
            wav_file = new_df.at[idx, r]
            npy_file = wav_file.replace('.wav', '.npy')

            fn = os.path.join(hubert_feature_dir, session_id, npy_file)
            if os.path.isfile(fn):          # doesn't have npy_filename
                new_df.at[idx, r] = npy_file
            else:
                new_df.at[idx, r] = None

    # dropping sessions containing empty numpy_file
    new_df.dropna(axis=0, how='any', inplace=True)
    new_df.to_csv('csv/'+csv_output, index=False)

# for i in tqdm(range(0, 13), desc='Generating'):
#     read_corpus(selected_task=i, csv_output='task'+str(i)+'.csv')
# --------------------

'''
Sleepiness Dataset
'''
from torch.utils.data import Dataset, WeightedRandomSampler
class SleepinessDataset(Dataset):
    def __init__(self, device, selected_task = 1):
        if selected_task not in range(0, 13):
            raise Exception("Invalide task selected!")

        self.FEATURE_TYPE = 'Embedding' # 'Embedding' or full hubert feature 'FullFeature'
        self.device = device
        self.selected_task = selected_task   # All=0 | 1 | 2 | ...| 12
        self.dataset = pd.read_csv('csv/task'+str(selected_task)+'.csv')

        # update labels to 0 & 1 based on sss value --> for binary classification
        #   - 0 : if sss in [1..3]
        #   - 1 : if sss in [4..7]
        self.dataset.loc[self.dataset['sss']>=4, 'sss'] = 7
        self.dataset.loc[self.dataset['sss']<=3, 'sss'] = 0
        self.dataset.loc[self.dataset['sss']==7, 'sss'] = 1

        #update gender: Male=1, Female=0, Other=0.5
        self.dataset.loc[self.dataset['gender'] == 'Female', 'gender'] = 0.
        self.dataset.loc[self.dataset['gender'] == 'Male', 'gender'] = 1.
        self.dataset.loc[self.dataset['gender'] == 'Other', 'gender'] = 0.5
        self.dataset.loc[self.dataset['gender'] == 'Transgender male', 'gender'] = 0.5
        self.dataset.loc[self.dataset['gender'] == 'Transgender female', 'gender'] = 0.5
        self.dataset.loc[self.dataset['gender'] == 'Prefer not to answer', 'gender'] = 0.5


        #self.dataset = self.dataset.iloc[:200, :]


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
            session_id = self.dataset.loc[idx, 'session_id']
            np_file = self.dataset.loc[idx, r]
            feat = np.load(os.path.join(hubert_feature_dir, session_id, np_file))  # return (T, 1024)
            if self.FEATURE_TYPE == 'Embedding':
                feat = np.load(os.path.join(hubert_feature_dir, session_id, np_file))
                feat = np.squeeze(feat)  # reform (1, T, 1024) to (T, 1024)

                # somehow the generated embedding contains NaN values ---> we need to replace them by 0
                x = np.isnan(feat)
                feat[x] = 0

                feat = np.mean(feat, axis=0)       # returns (1024, )
                feat = np.reshape(feat, (1, 1024)) # reform to (1, 1024)
            feat = torch.from_numpy(feat)
            features.append(feat)

        if self.FEATURE_TYPE == 'Embedding':
            pt_features = torch.cat(features)       # return (N, 1024)
        else:
            pt_features = pad_sequence(features)    # return (MaxT, N, 1024)
            pt_features = torch.permute(pt_features, (1, 0, 2))

        sss_label = self.dataset.loc[idx, 'sss']
        age_label = self.dataset.loc[idx, 'age']
        gender_label = self.dataset.loc[idx, 'gender']

        #onehot_label = torch.zeros(1, 7)
        #onehot_label[0, int(sss_label - 1)] = 1

        return pt_features, sss_label, age_label, gender_label

    ''' This function return weight of each class of the dataset'''
    def get_class_weights(self, indecies=None):
        if indecies != None:
            selected_rows = self.dataset.iloc[indecies]
        else:
            selected_rows = self.dataset

        labels_unique, counts = np.unique(selected_rows['sss'], return_counts=True)
        class_weights = [counts[c]/np.sum(counts) for c in range(len(counts))]

        return torch.FloatTensor(class_weights), counts


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


"""
This function plots training curve and attention.
    - The training histories are stored json file under folder image/
    - the attention maps (c1, c2, c3) are stored as tensor files under model/
"""
def plotting_training_curve():
    import json
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MultipleLocator

    cc1 = torch.load('model/c1.pt').detach().cpu()
    cc2 = torch.load('model/c2.pt').detach().cpu()
    cc3 = torch.load('model/c3.pt').detach().cpu()

    jsonfile = open('image/train-attention-hist0.0001.json', 'r')
    train_history = json.load(jsonfile)
    avg_train_accuracy = np.mean(train_history['train_acc']) * 100
    avg_test_accuracy = np.mean(train_history['test_acc']) * 100

    plt.plot(train_history['train_loss'], label='Training loss')
    plt.plot(train_history['train_acc'], label='Training Accurracy')
    plt.plot(train_history['test_acc'], label='Test Accuracy')
    plt.xlabel('epoch')
    plt.title(f'Avg. Train Acc={avg_train_accuracy:.2f}%; Avg. Test Acc=: {avg_test_accuracy:.2f}%')
    plt.legend()
    plt.savefig('image/train-history.png')

    for i in range(0, cc1.size(0)):
        c1 = torch.squeeze(cc1[i]).t()      #
        c2 = torch.squeeze(cc2[i]).t()
        c3 = torch.squeeze(cc3[i]).t()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
        ax1.set_title('L1')
        ax2.set_title('L2')
        ax3.set_title('L3')

        im1 = ax1.imshow(c1, aspect='auto')
        im2 = ax2.imshow(c2, aspect='auto')
        im3 = ax3.imshow(c3, aspect='auto')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.2)

        # cbar1 = plt.colorbar(im1, cax=ax1, ticks=MultipleLocator(0.2), format="%.2f")
        # cbar2 = plt.colorbar(im2, cax=ax2, ticks=MultipleLocator(0.2), format="%.2f")
        cbar3 = plt.colorbar(im3, cax=cax3, format="%.05f")
        ax1.yaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        ax1.set_xlabel('Response index')
        ax2.set_xlabel('Response index')
        ax3.set_xlabel('Response index')
        ax1.set_xticks(np.arange(0, 46, 5))
        ax2.set_xticks(np.arange(0, 46, 5))
        ax3.set_xticks(np.arange(0, 46, 5))

        plt.title('Attention scores')
        plt.tight_layout()
        plt.savefig('image/train-attention-' + str(i) + '.png')

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

