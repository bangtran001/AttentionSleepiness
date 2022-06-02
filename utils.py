# ----------------
# GENERAL VARIABLES
# ----------------
#hubert_embedding_dir = "../../ICASSP2022/pickle/hubert-embedding"  # --> embedding [1, 1024]
#hubert_feature_dir = "../../ICASSP2022/pickle/hubert-feature"      # --> features [T, 1024]
import json

hubert_feature_dir = "/media/data_sdh/HuBERT-Feature/Nx1024"      # --> features [T, 1024]
gemaps_feature_dir = "/media/data_sdh/GeMAPS-Feature"
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
def read_corpus(selected_task = 0, csv_output='dataset.csv'):
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
The dataset for Sleepiness using HuBERT Embedding.
'''
from torch.utils.data import Dataset, WeightedRandomSampler
class HuBERTEmbedDataset(Dataset):
    def __init__(self, device, selected_task = 0):
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

        # Solving imbalance problem with undersampling method
        # sleepy_ds = self.dataset[self.dataset['sss']==1]
        # tmp_ds = self.dataset[self.dataset['sss']==0]
        # awake_ds = tmp_ds.sample(n=len(sleepy_ds))
        # self.dataset = pd.concat([sleepy_ds, awake_ds], ignore_index=True)

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


    # calculating statistic information of the dataset
    def statistic(self):
        print(f'Number of sessions: {len(self.dataset)}')
        genders, gender_count =  np.unique(self.dataset['gender'], return_counts=True)
        print(f'Genders: {genders}, {gender_count}')

        # calculate gender vs. sleepy
        print('Gender\tNone-Sleepy\tSleepy')
        for i, sex in enumerate(genders):
            df = self.dataset[self.dataset['gender'] == sex]
            # gender
            sss_vals, count = np.unique(df['sss'], return_counts=True)
            print(f'Gender: {sex}')
            for j, sss in enumerate(sss_vals):
                print(f'\tsss_val={sss} has {count[j]}')
            print(f'\ttotal= {np.sum(count)}')

        print('==================')
        # calculate age_group vs. sleepy
        age_separator = [39, 64, 100]
        lower_bound = 18
        for upper_bound in age_separator:
            df1 = self.dataset[self.dataset['age'] >= lower_bound]
            df = df1[df1['age'] <= upper_bound]
            sss_vals, count = np.unique(df['sss'], return_counts=True)
            print(f'age [{lower_bound} - {upper_bound}]')
            for j, sss in enumerate(sss_vals):
                print(f'\tsss_val={sss} has {count[j]}')
            lower_bound = upper_bound+1

        print('==================')
        # calculate age_group vs. gender
        for i, sex in enumerate(genders):
            df = self.dataset[self.dataset['gender'] == sex]
            #age
            ages, count = np.unique(df['age'], return_counts=True)
            print(f'Gender: {sex}')
            tt = 0
            TTT = 0
            for j, age in enumerate(ages):
                print(f'\tage={age} has {count[j]}')
                tt = tt + count[j]
                TTT += count[j]
                if age%10 == 9:
                    print(f'\t--- total={tt}')
                    tt = 0

            print(f'\ttotal of gender = {TTT}')



# ds = HuBERTEmbedDataset(device=torch.device('cpu'))
# ds.statistic()

"""
The dataset for Sleepiness using GeMAPS Embedding.
"""
class GeMAPSDataset(Dataset):
    def __init__(self, device, selected_task = 0):
        if selected_task not in range(0, 13):
            raise Exception("Invalide task selected!")

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

        # Solving imbalance problem with undersampling method
        sleepy_ds = self.dataset[self.dataset['sss']==1]
        tmp_ds = self.dataset[self.dataset['sss']==0]
        awake_ds = tmp_ds.sample(n=len(sleepy_ds))
        self.dataset = pd.concat([sleepy_ds, awake_ds], ignore_index=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # the response columns of selected task
        if self.selected_task != 0:
            responses = [k for k, v in response_task_map.items() if v == self.selected_task]
        else:
            responses = list(response_task_map.keys())

        features = list()
        for r in responses:
            session_id = self.dataset.loc[idx, 'session_id']
            js_file = self.dataset.loc[idx, r]
            js_file = str(js_file).replace('.npy', '.json')
            f = open(os.path.join(gemaps_feature_dir, session_id, js_file), 'r')
            data = json.load(f)
            data = data.values()
            feat = list(data)[0]
            feat = list(feat.values())
            feat = np.array(feat)
            feat = torch.from_numpy(feat)
            feat = torch.unsqueeze(feat, dim=0)     # (1, 48)
            features.append(feat)

        pt_features = torch.cat(features)       # (N, 88)
        sss_label = self.dataset.loc[idx, 'sss']
        age_label = self.dataset.loc[idx, 'age']
        gender_label = self.dataset.loc[idx, 'gender']

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


# ds = GeMAPSDataset(device=torch.device('cuda'))
# x, label, age, gender = ds.__getitem__(10)
# print(x.size())

"""
This function plots training curve and attention.
    - The training histories are stored json file under folder image/
    - the attention maps (c1, c2, c3) are stored as tensor files under model/
"""
def plotting_training_curve(
        in_histfile = 'image/train-attention-hist-lr-0.0001-hubert.json',
        out_figfile = 'image/train-attention-hist-lr-0.0001-hubert.png',
        attention=True,
        feature='hubert',
        out_attmap = 'image/attention-map-hubert.png'
):
    import json
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MultipleLocator


    jsonfile = open(in_histfile, 'r')
    train_history = json.load(jsonfile)
    avg_train_accuracy = np.mean(train_history['train_acc']) * 100
    avg_test_accuracy = np.mean(train_history['test_acc']) * 100

    plt.plot(train_history['train_loss'], label='Training loss', marker='s', markevery=10)
    plt.plot(train_history['train_acc'], label='Training Accurracy', marker='v', markevery=10)
    plt.plot(train_history['test_acc'], label='Test Accuracy', marker='o', markevery=10)
    plt.xlabel('epoch')
    #plt.title(f'Avg. Train Acc={avg_train_accuracy:.2f}%; Avg. Test Acc=: {avg_test_accuracy:.2f}%')
    plt.legend()
    plt.savefig(out_figfile)

    if not attention:
        return

    cc1 = torch.load('model/c1-'+ feature +'.pt').detach().cpu()
    cc2 = torch.load('model/c2-'+ feature +'.pt').detach().cpu()
    cc3 = torch.load('model/c3-'+ feature +'.pt').detach().cpu()
    for i in range(0, cc1.size(0)):
        c1 = torch.squeeze(cc1[i]).t()
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
        plt.savefig(out_attmap)

def plotting_comparing_curves():
    import json
    import matplotlib.pyplot as plt
    historical_files = list()

    #HuBERT + Age + Gender
    historical_files.append('image/json/train-attention-hist-lr-0.0001-HuBERT.json')
    historical_files.append('image/json/train-noattention-hist-lr-0.0001-HuBERT.json')

    # HuBERT only
    historical_files.append('image/json/train-attention-hist-lr-0.0001-HuBERT-nogender.json')
    historical_files.append('image/json/train-noattention-hist-lr-0.0001-HuBERT-nogender.json')

    # GeMAPS only
    historical_files.append('image/json/train-attention-hist-lr-0.0001-GeMAPS-nogender.json')
    historical_files.append('image/json/train-noattention-hist-lr-0.0001-GeMAPS-nogender.json')

    data = list()
    for i, fn in enumerate(historical_files):
        data.append(json.load(open(fn, 'r')))

    plt.plot(data[0]['test_acc'], label='ATT + HuBERT + AG', marker='s', markevery=10)
    plt.plot(data[2]['test_acc'], label='ATT + HuE', marker='s', markevery=10)
    plt.plot(data[4]['test_acc'], label='ATT + GeMAPS', marker='s', markevery=10)

    plt.plot(data[1]['test_acc'], label='HuBERT + AG', marker='o', markevery=15)
    plt.plot(data[3]['test_acc'], label='HuBERT', marker='o', markevery=15)
    plt.plot(data[5]['test_acc'], label='GeMAPS', marker='o', markevery=15)

    plt.xlabel('Epoch', fontsize=16)
    plt.ylim([0, 1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12)

    # plt.title(f'Avg. Train Acc={avg_train_accuracy:.2f}%; Avg. Test Acc=: {avg_test_accuracy:.2f}%')
    plt.savefig('image/comparing-test-accur.png')
    plt.tight_layout()
    plt.show()

    print('Test accuracy (ATT + HuBERT + AG)', np.max(data[0]['test_acc']))
    print('Test accuracy (ATT + HuBERT)',np.max(data[2]['test_acc']))
    print('Test accuracy (ATT + GeMAPS)',np.mean(data[4]['test_acc']))
    print('Test accuracy (HuBERT + AG)',np.max(data[1]['test_acc']))
    print('Test accuracy (HuBERT)',np.max(data[3]['test_acc']))
    print('Test accuracy (GeMAPS)',np.max(data[5]['test_acc']))


''' ----------------------- '''
# plotting_comparing_curves()
# plotting_training_curve(
#         in_histfile = 'image/json/train-noattention-hist-lr-0.0001-GeMAPS-nogender.json',
#         out_figfile = 'image/train-noattention-hist-lr-0.0001-GeMAPS-nogender.png',
#         attention=False,
#         feature='GeMAPS',
#         out_attmap = 'image/attention-map-GeMAPS.png'
# )


# '''
# There are error numpy files when it generated
#     - embedding doesn't have shape [1, 1024]
#     - features doesn't have shape [N, 1024]
# '''
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