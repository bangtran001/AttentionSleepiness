"""
(Bang Tran - May, 2022)

Command:
    python3 generate_feature.py --feature=HuBERT|GeMAPS

-----------------------
The corpus of Voiceome Dataset is demonstrated in 'new2.csv'
    - Each row corresponds to 1 interview session
    - Each session has up to 46 useful speech responses (see Voiceome Dataset document to know more)

Returns:
    (1) HuBERT embedding speech responses. The embedding is an array [1 x T x 1024] and is stored in .npy file,
        where T is number of frames (or time dimension)
        To load this array from the disk, simply use:
            import numpy as np
            x = np.load('path-the-npy-file')
    (2) Extract ComParE_2016 features (>6K features) http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf
        The GeMAPS and eGeMAPS come in variants v01a, v01b and v02 (only eGeMAPS) are subsets of ComParE_2016

        List of features is described here: https://audeering.github.io/opensmile-python/usage.html

            ComParE_2016	65 / 65 / 6373
            GeMAPSv01a	18 / - / 62
            GeMAPSv01b	18 / - / 62
            eGeMAPSv01a	23 / - / 88
            eGeMAPSv01b	23 / - / 88
            eGeMAPSv02	25 / - / 88

        Useful codes of feature extracting can be found here!
         https://github.com/jim-schwoebel/allie/tree/master/features

"""

import argparse
import json

from tqdm import tqdm
import pandas as pd
import os
import re
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from transformers import Wav2Vec2Processor, HubertModel
import opensmile

raw_csv_corpus = '/media/data_sdc/HealthData/home2/biogen-clean-private/new2.csv'
raw_audio_dir1 = '/media/data_sdc/HealthData/home2/biogen-clean-private/sessions/'   # 3417
raw_audio_dir2 = '/media/data_sdf/HealthData/home2/biogen-clean-private/sessions/'   # 735

def generate_hubert_embedding(args):
    output_features_dir = '/media/data_sdh/HuBERT-Feature/Nx1024'

    # the main CSV file from SondeHealth
    df = pd.read_csv(raw_csv_corpus, usecols=[i for i in range(0, 51)])
    df.drop(columns=df.columns[[1, 2, 47, 49]], inplace=True) # drop redundant columns
    col_names = ['session_id'] + ['response' + str(i) for i in range(1, 47)]  # response1,...,response46
    df.columns = col_names

    # extract audio filename from responses
    # there're some empty cells within response columns
    #    --> Replace these empty cells by '--'; otherwise just change it to '.wav'
    for c in range(1, len(df.columns)):
        sample_ids = df.iloc[:, c]
        str_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
        sample_ids = list(
            map(lambda x: (re.findall(str_pattern, x)[0] + '.wav') if type(x) is str else '__', sample_ids))
        df.iloc[:, c] = sample_ids

    # Load pretrain HuBERT models
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    # generating .npy files
    for i in tqdm(range(args.start_index, len(df)), total=len(df)-args.start_index):
        session_id = df.iloc[i, 0]
        tqdm.write(f'\nworking session: \'{session_id}\'')

        # skip non-existed sessions
        if os.path.isdir(raw_audio_dir1 + '/' + session_id):
            session_path = os.path.join(raw_audio_dir1, session_id)
        elif os.path.isdir(raw_audio_dir2 + '/' + session_id):
            session_path = os.path.join(raw_audio_dir2, session_id)
        else:
            tqdm.write(f'session \'{session_id}\' is not existed ')
            continue

        for j in col_names[1:]:
            wav_file = df.loc[i, j]

            # skip unexisted files
            if not os.path.exists(os.path.join(session_path, wav_file)):
                print(f'\t \'{wav_file}\' was not found.')
                continue

            # skip files having size of 0
            if os.path.getsize(os.path.join(session_path, wav_file)) == 0:
                print(f'\t \'{wav_file}\' was has size of 0.')
                continue


            origin_wav, sr = torchaudio.load(os.path.join(session_path, wav_file))
            if origin_wav.size(1) == 0:     # skip audio files having length = 0
                print(f'\t \'{wav_file}\' has length = 0')
                continue

            # down sampling to 16KHz if needed
            wav_16Khz = origin_wav
            if sr != 16000:
                wav_16Khz = F.resample(origin_wav, sr, 16000, resampling_method="sinc_interpolation")

            # merging channels if needed
            if origin_wav.size(0) > 1:
                wav_16Khz = torch.mean(wav_16Khz, dim=0)

            # generate HuBERT feature
            input_values = processor(wav_16Khz, return_tensors="pt", sampling_rate=16000).input_values
            hubert_features = hubert_model(input_values).last_hidden_state
            npy_features = hubert_features.cpu().detach().numpy()

            # save to npy file
            if not os.path.isdir(output_features_dir + '/' + session_id):
                os.mkdir(output_features_dir + '/' + session_id)
            np_file = str(wav_file).replace('.wav', '.npy')
            np_file = os.path.join(output_features_dir, session_id, np_file)
            np.save(np_file, npy_features)

"""
This function generate GeMAP feature.
    GeMaps - older script version here (used in ICASSP2022 paper):
        https://github.com/jim-schwoebel/allie/blob/master/features/audio_features/opensmile_features.py
    They also have a new python api which is a bit more convenient to use. 
        https://github.com/audeering/opensmile-python
"""
def generate_GeMAPS_features(args):
    output_features_dir = '/media/data_sdh/GeMAPS-Feature'

    # the main CSV file from SondeHealth
    df = pd.read_csv(raw_csv_corpus, usecols=[i for i in range(0, 51)])
    df.drop(columns=df.columns[[1, 2, 47, 49]], inplace=True)  # drop redundant columns
    col_names = ['session_id'] + ['response' + str(i) for i in range(1, 47)]  # response1,...,response46
    df.columns = col_names

    # extract audio filename from responses
    # there're some empty cells within response columns
    #    --> Replace these empty cells by '--'; otherwise just change it to '.wav'
    for c in range(1, len(df.columns)):
        sample_ids = df.iloc[:, c]
        str_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
        sample_ids = list(
            map(lambda x: (re.findall(str_pattern, x)[0] + '.wav') if type(x) is str else '__', sample_ids))
        df.iloc[:, c] = sample_ids

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # generating .npy files
    for i in tqdm(range(args.start_index, len(df)), total=len(df)-args.start_index):
        session_id = df.iloc[i, 0]
        tqdm.write(f'\nworking session: \'{session_id}\'')

        # skip non-existed sessions
        if os.path.isdir(raw_audio_dir1 + '/' + session_id):
            session_path = os.path.join(raw_audio_dir1, session_id)
        elif os.path.isdir(raw_audio_dir2 + '/' + session_id):
            session_path = os.path.join(raw_audio_dir2, session_id)
        else:
            tqdm.write(f'session \'{session_id}\' is not existed ')
            continue

        for j in col_names[1:]:
            wav_file = df.loc[i, j]

            # skip unexisted files
            if not os.path.exists(os.path.join(session_path, wav_file)):
                print(f'\t \'{wav_file}\' was not found.')
                continue

            # skip files having size of 0
            if os.path.getsize(os.path.join(session_path, wav_file)) == 0:
                print(f'\t \'{wav_file}\' was has size of 0.')
                continue

            origin_wav, sr = torchaudio.load(os.path.join(session_path, wav_file))
            if origin_wav.size(1) == 0:  # skip audio files having length = 0
                print(f'\t \'{wav_file}\' has length = 0')
                continue

            # down sampling to 16KHz if needed
            wav_16Khz = origin_wav
            if sr != 16000:
                wav_16Khz = F.resample(origin_wav, sr, 16000, resampling_method="sinc_interpolation")

            # merging channels if needed
            if origin_wav.size(0) > 1:
                wav_16Khz = torch.mean(wav_16Khz, dim=0)

            torchaudio.save('temp.wav', torch.unsqueeze(wav_16Khz, dim=0), 16000)

            # generate GeMAPS feature
            GeMAPS_features = smile.process_file('temp.wav')
            os.remove('temp.wav')

            # save to json file
            if not os.path.isdir(output_features_dir + '/' + session_id):
                os.mkdir(output_features_dir + '/' + session_id)
            json_file = str(wav_file).replace('.wav', '.json')
            json_file = os.path.join(output_features_dir, session_id, json_file)
            GeMAPS_features.to_json(path_or_buf=json_file, orient='index', indent=False)

if __name__=='__main__':
    custom_parser = argparse.ArgumentParser(description='Generating features speech responses of Voiceome Dataset')
    custom_parser.add_argument('--feature', type=str, help='chose \'hubert\' or \'gemaps\'', required=True)
    custom_parser.add_argument('--start_index', type=int, default=0,
                                help='use this parameter to skip previous sessions')
    custom_args, _ = custom_parser.parse_known_args()

    if str(custom_args.feature).lower() in ['gemap', 'gemaps', 'egemap', 'egemaps'] :
        generate_GeMAPS_features(custom_args)
    elif str(custom_args.feature).lower() in ['hubert']:
        generate_hubert_embedding(custom_args)
    else:
        print('just a daft')
        import json
        # jsonfile = open('image/json/train-attention-hist-lr-0.0001-GeMAPS-nogender.json', 'r')
        # train_history = json.load(jsonfile)
        # print(train_history.keys())
        # pd.DataFrame({'train_loss':train_history['train_loss'],
        #               'train_acc': train_history['train_acc'],
        #               'test_loss': train_history['test_loss'],
        #               'test_acc': train_history['test_acc'],
        #               }).to_csv('temp/temp.csv', index=False)

        newh = pd.read_csv('temp/temp.csv')
        jsonfile = open('image/json/train-attention-hist-lr-0.0001-GeMAPS-nogender.json', 'w')
        json.dump({'train_loss':list(newh['train_loss']),
                      'train_acc': list(newh['train_acc']),
                      'test_loss': list(newh['test_loss']),
                      'test_acc': list(newh['test_acc']),
                      }, jsonfile)
        jsonfile.close()




        # '''
        # This code help to determine the index of session containing crashed wav file.
        # '''
        # df = pd.read_csv(raw_csv_corpus, usecols=[i for i in range(0, 51)])
        # df.drop(columns=df.columns[[1, 2, 47, 49]], inplace=True)  # drop redundant columns
        # col_names = ['session_id'] + ['response' + str(i) for i in range(1, 47)]  # response1,...,response46
        # df.columns = col_names
        #
        # for i in range(len(df)):
        #     session_id = df.iloc[i, 0]
        #     if session_id == 'a4242dc0-a5cb-11ea-bdc8-6733306edde1':
        #         print('index = ', i)
        #         break
        #
        # for c in range(1, len(df.columns)):
        #     sample_ids = df.iloc[:, c]
        #     str_pattern = "nlx-\w*-\w*-\w*-\w*-\w*"
        #     sample_ids = list(
        #         map(lambda x: (re.findall(str_pattern, x)[0] + '.wav') if type(x) is str else '__', sample_ids))
        #     df.iloc[:, c] = sample_ids
        #
        # if os.path.isdir(raw_audio_dir1 + '/' + session_id):
        #     session_path = os.path.join(raw_audio_dir1, session_id)
        # elif os.path.isdir(raw_audio_dir2 + '/' + session_id):
        #     session_path = os.path.join(raw_audio_dir2, session_id)
        # for j in col_names[1:]:
        #     wav_file = df.loc[1745, j]
        #
        #     # skip unexisted files
        #     if not os.path.exists(os.path.join(session_path, wav_file)):
        #         print(f'\t \'{wav_file}\' was not found.')
        #         continue
        #
        #     # skip files having size of 0
        #     if os.path.getsize(os.path.join(session_path, wav_file)) == 0:
        #         print(f'\t \'{wav_file}\' was has size of 0.')
        #         continue
        #
        #     origin_wav, sr = torchaudio.load(os.path.join(session_path, wav_file))
        #     if origin_wav.size(1) == 0:  # skip audio files having length = 0
        #         print(f'\t \'{wav_file}\' has length = 0')
        #         continue
        #
        #     # down sampling to 16KHz if needed
        #     wav_16Khz = origin_wav
        #     if sr != 16000:
        #         wav_16Khz = F.resample(origin_wav, sr, 16000, resampling_method="sinc_interpolation")
        #
        #     # merging channels if needed
        #     if origin_wav.size(0) > 1:
        #         wav_16Khz = torch.mean(wav_16Khz, dim=0)
        #
        #     print(wav_file, origin_wav.size(), wav_16Khz.size())
        #     torchaudio.save('temp.wav', torch.unsqueeze(wav_16Khz, dim=0), 16000)








