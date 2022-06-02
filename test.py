'''
------
C_1 -->top 3:tensor([6, 8, 7]); top 5:tensor([42,  9,  6,  8,  7]); top 10:tensor([ 5, 37, 44, 41, 43, 42,  9,  6,  8,  7])
C_2 -->top 3:tensor([6, 8, 7]); top 5:tensor([42,  9,  6,  8,  7]); top 10:tensor([ 1, 46, 44, 41, 43, 42,  9,  6,  8,  7])
C_3 -->top 3:tensor([6, 8, 7]); top 5:tensor([ 9, 42,  6,  8,  7]); top 10:tensor([20, 37, 44, 41, 43,  9, 42,  6,  8,  7])
------
C_1 -->top 3:tensor([3, 1, 2]); top 5:tensor([8, 4, 3, 1, 2]); top 10:tensor([10,  6,  5,  9,  7,  8,  4,  3,  1,  2])
C_2 -->top 3:tensor([3, 1, 2]); top 5:tensor([7, 4, 3, 1, 2]); top 10:tensor([10,  6,  9,  5,  8,  7,  4,  3,  1,  2])
C_3 -->top 3:tensor([8, 1, 2]); top 5:tensor([7, 3, 8, 1, 2]); top 10:tensor([ 5,  4, 10,  6,  9,  7,  3,  8,  1,  2])
'''

# top 3:
# 8 - 4 times
# 1, 2, 6, 7 - 3 times
# 3 - 2 times
#
# -----
#
# top5:
# 7, 8 - 5 times
# 6, 1, 2, 42, 9, 3 - 3 times
# 4 - 2 times
#
# -------
# top 10:
# 8, 7, 6, 9 - 6 times
# 1, 5 - 4 times
# 2, 3, 42, 43, 41, 44, 10, 4  - 3 times
# 37 - 2 times
# 46, 20 - 1 times
#
#
# =========> Final rank
# ===> 6, 7 ,8, 1, 2, 3: are rank 1
# ===> 9, 4, 5, 42
# ===> 10, 20, 37, 41, 43, 44, 46

import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from model import Adaptive_AttnSDM, GeMAPS_AttnSDM

import utils
from train_origin import OriginalAttentionModel
from utils import HuBERTEmbedDataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def run_test(model_name, attention_map_name, data_type='HuBERT', age_gender=True):
    print(f'Run testing for trained model \'{model_name}\'')

    device = torch.device('cuda')
    loss_func = nn.CrossEntropyLoss()

    ds = None
    if data_type.lower() in ['hubert']:
        ds = HuBERTEmbedDataset(device=device, selected_task=0)
    elif data_type.lower() in ['egemaps', 'gemaps']:
        ds = utils.GeMAPSDataset(device=device, selected_task=0)
    else:
        print('Invalide datatype')
        return
    testloader = DataLoader(dataset=ds, batch_size=64, shuffle=True, drop_last=False, num_workers=16)

    if data_type.lower() in ['hubert']:
        model = Adaptive_AttnSDM(num_classes=2, age_gender=age_gender)
    elif data_type.lower() in ['gemaps', 'egmaps']:
        model = GeMAPS_AttnSDM(num_classes=2)
    else:
        model = OriginalAttentionModel(num_classes=2).to(device)

    model.load_state_dict(torch.load('model/'+model_name))
    model = model.to(device=device)
    model.eval()

    with torch.no_grad():
        total = 0
        correct = 0
        avg_loss = 0.0
        print("Runing test with trained model...")
        c1_list = list()
        c2_list = list()
        c3_list = list()
        for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            x_test, labels_test, ages, genders = data
            x_test, labels_test = x_test.to(device), labels_test.to(device)
            x_test = torch.unsqueeze(x_test, dim=1)  # reshape from (?, 46, 1024) --> (?, 1, 46, 1024)

            ages, genders = ages.float().to(device), genders.float().to(device)
            ages = torch.unsqueeze(ages, dim=1)
            genders = torch.unsqueeze(genders, dim=1)

            # x_test = torch.unsqueeze(x_test, dim=2)
            # x_test = x_test.repeat(1, 1, 46, 1)
            # x_test = torch.permute(x_test, (0, 3, 1, 2))        # (?, 1024, 46, 46)

            if data_type.lower() in ['hubert']:
                pred_test, c1, c2, c3 = model(x_test, ages, genders)
            elif data_type.lower() in ['gemaps', 'egmaps']:
                pred_test, c1, c2, c3 = model(x_test.float().to(device))
            else:
                pred_test, c1, c2, c3 = model(x_test.to(device), ages, genders)

            c1_list.append(c1)
            c2_list.append(c2)
            c3_list.append(c3)

            predict = torch.argmax(pred_test, 1)
            total += labels_test.size(0)
            correct += torch.eq(predict, labels_test).sum().double().item()
            avg_loss += loss_func(pred_test, labels_test).item() / len(testloader)

        print(f"\tAccuracy on test data: {100 * correct / total:.2f}%")
        print(f"\tAvg Loss on test data: {avg_loss:.2f}")

        c1 = torch.concat(c1_list, dim=0)
        c2 = torch.concat(c2_list, dim=0)
        c3 = torch.concat(c3_list, dim=0)
        print(c1.size(), c2.size(), c3.size())

        torch.save(c1.detach().cpu(), 'model/c1-'+attention_map_name)
        torch.save(c2.detach().cpu(), 'model/c2-'+attention_map_name)
        torch.save(c3.detach().cpu(), 'model/c3-'+attention_map_name)

if __name__ == '__main__':
    custom_parser = argparse.ArgumentParser(description='Testing trained model')
    custom_parser.add_argument('--save_attention_map', type=bool, required=False)
    custom_args, _ = custom_parser.parse_known_args()

    custom_args.save_attention_map = False
    if custom_args.save_attention_map == True :
        run_test(data_type='GeMAPS',
                 #model_name='attn_model_weights_lr_0.0001-HuBERT.pth',
                 #model_name='attn_model_weights_lr_0.0001-HuBERT-nogender.pth',
                 model_name='attn_model_weights_lr_0.0001-GeMAPS.pth',
                 attention_map_name = 'GeMAPS.pt',
                 age_gender = False)
        exit(0)

    C = list()
    logC = list()
    C.append(torch.squeeze(torch.load('model/c1-HuBERT-AG.pt')))
    C.append(torch.squeeze(torch.load('model/c2-HuBERT-AG.pt')))
    C.append(torch.squeeze(torch.load('model/c3-HuBERT-AG.pt')))
    C.append(torch.squeeze(torch.load('model/c1-HuBERT.pt')))
    C.append(torch.squeeze(torch.load('model/c2-HuBERT.pt')))
    C.append(torch.squeeze(torch.load('model/c3-HuBERT.pt')))
    for i in range(len(C)):
        C[i] = torch.mean(C[i], dim=0)      # mean over (?, 46, 32) --> (46, 32)
        C[i] = C[i].t()
        logC.append(torch.log10(C[i]))


    # ----------- draw bar graphs comparation  ---------------
    for i in range(len(C)):
        sumEnergy = torch.sum(C[i], dim=0)
        ranks = torch.argsort(sumEnergy, dim=0)
        if i % 3 == 0:
            print('------')
        print(f'C_{i%3+1} -->top 3:{ranks[-3:]+1}; top 5:{ranks[-5:]+1}; top 10:{ranks[-10:]+1}')

    # ----------- draw heatmap & bar graphs ---------------
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    ax[0][0].set_title(r'$\mathcal{C}_1$', fontsize=20)
    ax[0][1].set_title(r'$\mathcal{C}_2$', fontsize=20)
    ax[0][2].set_title(r'$\mathcal{C}_3$', fontsize=20)

    ax[1][0].set_title(r'$\Downarrow$ Total energy', fontsize=18)
    ax[1][1].set_title(r'$\Downarrow$ Total energy', fontsize=18)
    ax[1][2].set_title(r'$\Downarrow$ Total energy', fontsize=18)
    for i in range(0, 3):
        ax[1][i].set_xlabel('Response index', fontsize=16)

    for i in range(0, 2):
        for j in range(0, 3):
            # if i == 0 and j > 0:
            #     ax[i][j].yaxis.set_visible(False)
            for label in ax[i][j].get_xticklabels():
                 label.set_fontsize(15)
            for label in ax[i][j].get_yticklabels():
                 label.set_fontsize(15)

    im_list = list()
    log_im = list()
    col = 0   # 0 for HuBERT+Age+Gender; 1 for HuBERT-only
    for j in range(0, 3):
        im = ax[0][j].imshow(C[col*3 + j], aspect='auto', cmap="plasma")
        im_list.append(im)
        # eliminate negative values
        tmpC = torch.where(C[col*3 + j].double() < 0.0, 0.0, C[col*3 + j].double())
        sumCol = torch.sum(tmpC, dim=0)
        y_max = torch.abs(sumCol).max()
        ax[1][j].bar(range(0, 46), sumCol)

    sup_title = '(a) Generated by HuBERT + Age + Gender' if col==0  else  '(b) Generated by HuBERT only'
    plt.suptitle(sup_title, fontsize=22)
    plt.tight_layout()
    plt.savefig('image/attention-map-HuBERT-AG.png') if col==0 else plt.savefig('image/attention-map-HuBERT.png')
    plt.show()
    exit(0)

    #--------------- draw heatmap & log10 heatmap ---------------
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))

    ax[0][0].set_title(r'$\mathcal{C}_1$', fontsize=20)
    ax[0][1].set_title(r'$\mathcal{C}_2$', fontsize=20)
    ax[0][2].set_title(r'$\mathcal{C}_3$', fontsize=20)
    c1, c2, c3 = C[0], C[1], C[2]

    ax[1][0].set_title(r'$log_{10}(\mathcal{C}_1)$', fontsize=20)
    ax[1][1].set_title(r'$log_{10}(\mathcal{C}_2)$', fontsize=20)
    ax[1][2].set_title(r'$log_{10}(\mathcal{C}_3$)', fontsize=20)
    for i in range(0, 3):
        ax[1][i].set_xlabel('Response index', fontsize=16)
    for i in range(0, 2):
        for j in range(0, 3):
            ax[i][j].yaxis.set_visible(False)
            for label in ax[i][j].get_xticklabels():
                 label.set_fontsize(15)
    im_list = list()
    log_im = list()
    col = 0         # 0 for HuBERT+Age+Gender; 1 for HuBERT-only
    for j in range(0, 3):
        im = ax[0][j].imshow(C[col*3 + j], aspect='auto', cmap="plasma")
        im_list.append(im)
        logimg = ax[1][j].imshow(logC[col*3 + j], aspect='auto', cmap="plasma")
        log_im.append(logimg)

    plt.tight_layout()
    # plt.savefig('image/attention-map-HuBERT.png')
    plt.show()


    # divider3 = make_axes_locatable(ax[0][2])
    # cax3 = divider3.append_axes("right", size="3%", pad=0.2)
    # im1 = ax1.imshow(c1, aspect='auto')
    # im2 = ax2.imshow(c2, aspect='auto')
    # im3 = ax3.imshow(c3, aspect='auto')
    # divider3 = make_axes_locatable(ax3)
    # cax3 = divider3.append_axes("right", size="3%", pad=0.2)
    #
    # # cbar1 = plt.colorbar(im1, cax=ax1, ticks=MultipleLocator(0.2), format="%.2f")
    # # cbar2 = plt.colorbar(im2, cax=ax2, ticks=MultipleLocator(0.2), format="%.2f")
    # # cbar3 = plt.colorbar(im3, cax=cax3, format="%.f")

    # # ax1.set_xticks(np.arange(0, 46))
    # # ax2.set_xticks(np.arange(0, 46))
    # # ax3.set_xticks(np.arange(0, 46))
    # # ax1.set_xticks([1, 10, 20, 30, 40, 45])
    #
    # ax1.set_xlabel('Response ', fontsize=16)
    # ax2.set_xlabel('Response', fontsize=16)
    # ax3.set_xlabel('Response', fontsize=16)
    # for label in ax1.get_xticklabels():
    #     label.set_fontsize(14)
    # for label in ax2.get_xticklabels():
    #     label.set_fontsize(14)
    # for label in ax2.get_xticklabels():
    #     label.set_fontsize(14)
    # plt.yticks(fontsize=15)

    # plt.title('HuBERT + Age + Gender')
    # plt.tight_layout()
    # plt.savefig('image/attention-map-HuBERT-AG.png')
    # plt.show()


