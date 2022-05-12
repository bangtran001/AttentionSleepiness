from torch import nn
import torch

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


# Sleepiness Detection Model (SDM) using (1x1024) embeding
class SDM_Embedding(nn.Module):
    def __init__(self, selected_task=1):
        super(SDM_Embedding, self).__init__()
        self.selected_task = selected_task
        self.input_channels = len([k for k, v in response_task_map.items() if v == selected_task])

        # L1 input shape = (?, N, 32, 32)
        # Conv -> (?, 32, 32, 32)
        # Pool -> (?, 32, 16, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
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
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out