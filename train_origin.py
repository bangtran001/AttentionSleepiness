import argparse
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.nn import functional as F
from tqdm import tqdm
import json
import torch
import utils
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
import sys

'''
Model construction
'''
"""
    ## Implementation: “Learn To Pay Attention” published in ICLR 2018 conference
    ## https://nivedwho.github.io/blog/posts/attncnn/
    ## https://colab.research.google.com/github/nivedwho/Colab/blob/main/SelfAttnCNN.ipynb#scrollTo=BYnr1NuQFFJk
"""
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i + 1], kernel_size=3, padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i + 1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

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

class OriginalAttentionModel(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True):
        super(OriginalAttentionModel, self).__init__()
        self.attention = attention
        self.memory = {}
        self.cv1 = ConvBlock(1024, 512, 2)
        self.cv2 = ConvBlock(512, 512, 2)
        self.cv3 = ConvBlock(512, 512, 3)
        self.cv4 = ConvBlock(512, 512, 3)
        self.cv5 = ConvBlock(512, 512, 3)
        self.cv6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True)

        # Attention = True
        self.projector = ProjectorBlock(256, 512)
        self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # Final Classification Layer
        #self.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        self.classify = nn.Sequential(
                            nn.Linear(in_features=512 * 3, out_features=1024, bias=True),
                            nn.ReLU(),
                            nn.Linear(in_features=1024, out_features=512, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=512, out_features=256, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=256, out_features=128, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=128, out_features=64, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=64, out_features=num_classes, bias=True))

            # weight = U [-(1/sqrt(n)), 1/sqrt(n)]
        weights_init_xavierNormal(self)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        # self.memory[]
        l1 = self.cv3(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)

        l2 = self.cv4(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)

        l3 = self.cv5(x)
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)

        x = self.cv6(x)
        g = self.dense(x)

        # Attention part
        #c1, g1 = self.attn1(self.projector(l1), g)
        c1, g1 = self.attn1(l1, g)
        c2, g2 = self.attn2(l2, g)
        c3, g3 = self.attn3(l3, g)
        g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
        np = c1.detach().cpu().numpy()

        # classification layer
        x = self.classify(g)  # batch_sizexnum_classes
        return [x, c1, c2, c3]



'''
Training part
'''

ds = utils.HuBERTEmbedDataset(device=torch.device("cuda"), selected_task=0)
n_samples = len(ds)
n_test_samples = int(n_samples * .25)
n_train_samples = n_samples - n_test_samples
train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])

LEARNING_RATE = 0.001   #initial learning rate
BATCH_SIZE = 64
epochs = 200
device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()

from torchinfo import summary

model = OriginalAttentionModel(im_size=46, num_classes=2).to(device)
criterion.to(device)
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
lr_lambda = lambda epoch: 0.5**(epoch//25)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

trainloader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=16)
testloader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=16)

summary(model, (BATCH_SIZE, 1024, 46, 46))

step = 0
running_avg_accuracy = 0
aug = 0

train_acc_hist = list()
train_loss_hist = list()
test_acc_hist = list()
best_train_lost = sys.float_info.max
for epoch in range(epochs):
    images_disp = []
    tqdm.write(f"---- Epoch {epoch} @ learning rate {optimizer.param_groups[0]['lr']} --- ")

    avg_training_loos = 0
    avg_training_acc = 0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        inputs, labels, ages, genders = data

        # (?, 46, 1024)

        ages = ages/100
        inputs = torch.unsqueeze(inputs, dim=2)
        inputs = inputs.repeat(1, 1, 46, 1)   # (?, 46, 46, 1024)
        # inputs = inputs.view(-1, 46, 46, 1024)
        inputs = torch.permute(inputs, (0, 3, 1, 2)) # (?, 1024, 46, 46)
        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        pred, _, _, _ = model(inputs)

        # backward
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        avg_training_loos += loss.item()/len(trainloader)

        model.eval()
        pred, __, __, __ = model(inputs)
        predict = torch.argmax(pred, 1)
        total = labels.size(0)
        correct = torch.eq(predict, labels).sum().double().item()
        accuracy = correct / total
        avg_training_acc += accuracy/len(trainloader)

    train_loss_hist.append(avg_training_loos)
    train_acc_hist.append(avg_training_acc)
    tqdm.write(f"\tTrain loss: {avg_training_loos}")
    tqdm.write(f"\tTrain Accuracy: {avg_training_acc*100:.2f}%")

    # save the best model based on trainloss & trainaccuracy
    if avg_training_loos < best_train_lost:
        current_lr = optimizer.param_groups[0]['lr']
        torch.save(model.state_dict(), f'model/original-attention-lr-{current_lr}.pth')
        best_train_lost = avg_training_loos

    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        # log scalars
        for i, data in enumerate(testloader, 0):
            x_test, labels_test, _, _= data
            x_test, labels_test = x_test.to(device), labels_test.to(device)
            # if i == 0:  # archive images in order to save to logs
                # images_disp.append(inputs[0:36, :, :, :])

            x_test = torch.unsqueeze(x_test, dim=2)
            x_test = x_test.repeat(1, 1, 46, 1)
            #x_test = x_test.view(-1, 46, 46, 1024)
            x_test = torch.permute(x_test, (0, 3, 1, 2))

            pred_test, __, __, __ = model(x_test)
            predict = torch.argmax(pred_test, 1)
            total += labels_test.size(0)
            correct += torch.eq(predict, labels_test).sum().double().item()

        tqdm.write(f"\tAccuracy on test data: {100*correct/total:.2f}%")
        test_acc_hist.append(correct/total)

        # I_train = utils.make_grid(images_disp[0], nrow=6, normalize=True, scale_each=True)
        # show(I_train)
        # if epoch == 0:
        # I_test = utils.make_grid(images_disp[1], nrow=6, normalize=True, scale_each=True)
        # show(I_test)


# plotting the training history
import matplotlib.pyplot as plt
plt.plot(train_acc_hist, label='Train accuracy')
plt.plot(test_acc_hist, label='Test accurracy')
plt.plot(train_loss_hist, label='Training loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('image/train-original-attention-hist.png')

