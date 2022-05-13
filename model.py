from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

""" this dictionary maps ressponseX --> task X """
response_task_map = {}
for i in range(1, 10):      # task1 - task10
    response_task_map['response' + str(i)] = i
for i in range(10, 35):     #  task 10 (Confrontational naming)
    response_task_map['response' + str(i)] = 10
for i in range(35, 45):     # task 11 (non-word)
    response_task_map['response' + str(i)] = 11
response_task_map['response46'] = 12    # task 12 (sentence repeat)
response_task_map['response48'] = 12

""" None-Attention: Sleepiness Detection Model (SDM) using (1x1024) embeding """
class SDM(nn.Module):
    def __init__(self, selected_task=1):
        super(SDM, self).__init__()
        assert(selected_task in range(0,13)), 'Selected task needs be from 0 to 12'

        if selected_task != 0:
            self.input_channels = len([k for k, v in response_task_map.items() if v == selected_task])
        else:
            self.input_channels = len(response_task_map.keys())


        # L1 input shape = (?, N, 32, 32)
        # Conv -> (?, 32, 32, 32)
        # Pool -> (?, 32, 16, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2))

        # L2 input shape = (?, 32, 16, 16)
        # Conv -> (?, 64, 16, 16)
        # Pool -> (?, 64, 8, 8)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2))

        # L3 input shape = (?, 64, 8, 8)
        # Conv -> (?, 128, 8, 8)
        # Pool -> (?, 128, 4, 4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2))

        # L4 FC 4x4x128 inputs -> 1024 outputs
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4 * 4 * 128, out_features=1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2))
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=2, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

""" Attention mechanism blocks """
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
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

class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_))  # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)  # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)  # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1, 1)).view(N, C)

        return c.view(N, 1, W, H), output

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

"""
Attention Sleepiness Detection Model
"""
class AttnSDM(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True):
        super(AttnSDM, self).__init__()
        self.attention = attention
        self.memory = {}
        self.cv1 = ConvBlock(46, 64, 2)         # 46 responses ~ 46 input channels
        self.cv2 = ConvBlock(64, 128, 2)
        self.cv3 = ConvBlock(128, 256, 3)
        self.cv4 = ConvBlock(256, 512, 3)
        self.cv5 = ConvBlock(512, 512, 3)
        self.cv6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size / 32), padding=0, bias=True)

        # Attention = True
        self.projector = ProjectorBlock(256, 512)
        self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)

        # Final Classification Layer
        self.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)

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
        c1, g1 = self.attn1(self.projector(l1), g)
        c2, g2 = self.attn2(l2, g)
        c3, g3 = self.attn3(l3, g)
        g = torch.cat((g1, g2, g3), dim=1)  # batch_size x C
        np = c1.detach().cpu().numpy()

        # classification layer
        x = self.classify(g)  # batch_size x num_classes
        return [x, c1, c2, c3]





## Implementation: “Learn To Pay Attention” published in ICLR 2018 conference
## https://nivedwho.github.io/blog/posts/attncnn/
## https://colab.research.google.com/github/nivedwho/Colab/blob/main/SelfAttnCNN.ipynb#scrollTo=BYnr1NuQFFJk

## https://github.com/arundo/tsaug
## https://github.com/jim-schwoebel/allie/tree/master/visualize