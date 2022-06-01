'''
This file trains the model withe inputs of size (32 x 32 x 46)
'''


import json
import sys

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import utils


class Adaptive_ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(Adaptive_ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

class Adaptive_ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(Adaptive_ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        return self.op(inputs)

class Adaptive_LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(Adaptive_LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l + g)  # batch_size x 1 x W x H
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, W, H), g

def weights_init_xavierNormal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, val=0.)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)

class Adaptive_AttnSDM(nn.Module):
    def __init__(self, num_classes, normalize_attn=True,
                 selected_task=0,
                 apply_attention=True,      # using attention block
                 data_type='HuBERT',        # input data for training
                 age_gender=True            # use age & gender at classifier block
                 ):

        super(Adaptive_AttnSDM, self).__init__()
        # using attention block
        self.apply_attention = apply_attention
        self.data_type = data_type              # HuBERT or GeMAPS
        self.age_gender = age_gender

        if selected_task != 0:
            self.input_channels = len([k for k, v in response_task_map.items() if v == selected_task])
        else:
            self.input_channels = len(response_task_map.keys())

        self.cv1 = Adaptive_ConvBlock(in_features=46, out_features=64, num_conv=1, pool=True)
        self.cv2 = Adaptive_ConvBlock(in_features=64, out_features=128, num_conv=1, pool=True)
        self.cv3 = Adaptive_ConvBlock(in_features=128, out_features=256, num_conv=1, pool=True)
        self.cv4 = Adaptive_ConvBlock(in_features=256, out_features=512, num_conv=1)
        self.cv5 = Adaptive_ConvBlock(in_features=512, out_features=512, num_conv=1)
        self.cv6 = Adaptive_ConvBlock(in_features=512, out_features=512, num_conv=1)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)

        # Attention
        self.projector = Adaptive_ProjectorBlock(256, 512)
        self.attn1 = Adaptive_LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn2 = Adaptive_LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn3 = Adaptive_LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)

        # padding & flatten when does not use attention
        self.pad = nn.Sequential(
                        # (?, 512, 46, 32)
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm2d(num_features=512, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(3, 3), stride=3, padding=0),
                        # (?, 512, 15, 10)
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm2d(num_features=512, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(3, 3), stride=3, padding=0),
                        # (?, 512, 5, 3)
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
                        nn.BatchNorm2d(num_features=512, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(3, 3), stride=3, padding=0),
                        # (?, 512, 1, 1)
                     )



        # padding & flatten when does not use attention
        self.pad = nn.Sequential(
                        # (?, 512, 4, 4)
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, bias=True),
                        nn.BatchNorm2d(num_features=512, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0),
                        # (?, 512, 2, 2)
                    )

        # Final Classification Layer
        self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)

        # weight = U [-(1/sqrt(n)), 1/sqrt(n)]
        weights_init_xavierNormal(self)

    def forward(self, x, age, gender):
        x = self.cv1(x)         # (?, 46, 32, 32) --> (?, 64, 46, 256)
        x = self.cv2(x)         # (?, 64, 46, 512) --> (?, 128, 46, 64)
        x = self.cv3(x)         # (?, 128, 46, 256) --> (?, 256, 46, 32)

        l1 = x
        x = self.cv4(x)         # (?, 256, 46, 128) --> (?, 512, 46, 32)
        l2 = x
        x = self.cv5(x)         # (?, 512, 46, 128) --> (?, 512, 46, 32)
        l3 = x
        x = self.cv6(x)         # (?, 512, 46, 128) --> (?, 512, 46, 32)
        g = self.dense(x)       # (?, 512, 46, 32)


        g = self.pad(g)
        g = g.view(g.size(0), -1)  # Flatten
        g = torch.cat((g, g, g), dim=1)  # (?, 512*3)

        # classification Block
        x = self.classify(g)  # batch_size x num_classes
        return x, 0, 0, 0

'''
#####################
'''


response_task_map = {}
for i in range(1, 10):      # task1 - task10
    response_task_map['response' + str(i)] = i
for i in range(10, 35):     #  task 10 (Confrontational naming)
    response_task_map['response' + str(i)] = 10
for i in range(35, 45):     # task 11 (non-word)
    response_task_map['response' + str(i)] = 11
response_task_map['response46'] = 12    # task 12 (sentence repeat)
response_task_map['response48'] = 12

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
N_EPOCHS = 200
print('----------Train the customized model with 32x32x46--------------')

import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

my_model = Adaptive_AttnSDM(num_classes=2,
                            apply_attention=False,
                            age_gender=False)
device_ids = [0, ]
loss_func = nn.CrossEntropyLoss()
model = nn.DataParallel(my_model, device_ids=device_ids).to(device)
loss_func.to(device)
# optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

ds = utils.HuBERTEmbedDataset(device=device, selected_task=0)
summary(my_model, (BATCH_SIZE, 46, 32, 32),
        age=torch.ones(BATCH_SIZE, 1),
        gender=torch.zeros(BATCH_SIZE, 1))
n_samples = len(ds)
n_test_samples = int(n_samples * .20)
n_train_samples = n_samples - n_test_samples
train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])
train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8)
validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8)
print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

training_accuracy = list()
training_loss = list()
testing_accuracy = list()
testing_loss = list()
# use classweight in the case don't use undersampling method
class_weights, class_counts = ds.get_class_weights(train_ds.indices)
best_loss = sys.float_info.max
for epoch in range(N_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{N_EPOCHS} @ lr={optimizer.param_groups[0]['lr']} ------")
    avg_loss = 0.0
    avg_acc = 0.0
    for i, (inputs, labels, ages, genders) in tqdm(enumerate(train_dl, start=0), total=len(train_dl),
                                                   desc='Training step'):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        inputs = torch.unsqueeze(inputs, dim=1)  # reshape from (?, 46, 1024) --> (?, 1, 46, 1024)
        inputs = inputs.view(-1, 46, 32, 32)
        inputs, labels = inputs.to(device), labels.to(device)


        ages, genders = ages.float().to(device), genders.float().to(device)
        ages = torch.unsqueeze(ages, dim=1)
        genders = torch.unsqueeze(genders, dim=1)

        # forward
        pred, __, __, __ = model(inputs, ages, genders)

        # backward
        loss = loss_func(pred, labels)
        loss.backward()
        optimizer.step()
        # print(f"  > step {i+1}/{len(train_dl)} loss={loss.item():.4f}")

        avg_loss += loss.item() / len(train_dl)

        model.eval()
        pred, _, _, _ = model(inputs.float(), ages, genders)

        # use classweight in the case unused undersampling method
        predict = torch.argmax(pred - class_weights.to(device), 1)
        # predict = torch.argmax(pred, 1)
        total = labels.size(0)
        correct = torch.eq(predict, labels).sum().double().item()
        avg_acc += (correct / total) / len(train_dl)
    print(f"\tavg. train loss={avg_loss:.4f} \ttrain accuracy={100 * avg_acc:.2f}%")
    # -- save the model
    if avg_loss < best_loss:
        torch.save(my_model.state_dict(), 'model/noattention_hubertonly_323246.pth')

    # logging
    training_accuracy.append(avg_acc)
    training_loss.append(avg_loss)

    model.eval()
    with torch.no_grad():
        test_total = 0
        test_correct = 0
        avg_loss = 0.0
        for i, (inputs_test, labels_test, ages_test, genders_test) in enumerate(validation_dl, 0):
            inputs_test = torch.unsqueeze(inputs_test, dim=1)  # reshape from (?, 46, 1024) --> (?, 1, 46, 1024)
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

            inputs_test = inputs_test.view(-1, 46, 32, 32)

            ages_test, genders_test = ages_test.float().to(device), genders_test.float().to(device)
            ages_test = torch.unsqueeze(ages_test, dim=1)
            genders_test = torch.unsqueeze(genders_test, dim=1)

            pred_test, _, _, _ = model(inputs_test.float(), ages_test, genders_test)

            avg_loss += loss_func(pred_test, labels_test).item() / len(validation_dl)
            # use classweight in the case don't use undersampling method
            predict = torch.argmax(pred_test - class_weights.to(device), 1)
            # predict = torch.argmax(pred_test, 1)
            test_total += labels_test.size(0)
            test_correct += torch.eq(predict, labels_test).sum().double().item()
    print(f"\tavg. test loss={avg_loss:.4f} \ttest accuracy={100 * (test_correct / test_total):.2f}%")
    # logging
    testing_loss.append(avg_loss)
    testing_accuracy.append(test_correct / test_total)

print("Training is Done!")

jsonfile = open('image/json/noattention_hubertonly_323246.json', 'w')

# -- save training history to json file
json.dump({'train_loss': training_loss,
           'train_acc': training_accuracy,
           'test_loss': testing_loss,
           'test_acc': testing_accuracy}, jsonfile)
jsonfile.close()

# --- plotting the train
import matplotlib.pyplot as plt

plt.plot(training_loss, label='Training loss')
plt.plot(training_accuracy, label='Training Accurracy')
plt.plot(testing_accuracy, label='Test Accuracy')
plt.xlabel('epoch')
plt.legend()