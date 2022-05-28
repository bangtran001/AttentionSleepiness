import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from train_origin import OriginalAttentionModel
from utils import HuBERTEmbedDataset


device = torch.device('cuda')
loss_func = nn.CrossEntropyLoss()
ds = HuBERTEmbedDataset(device=torch.device('cuda'), selected_task=0)
testloader = DataLoader(dataset=ds, batch_size=64, shuffle=True, drop_last=False, num_workers=16)

model = OriginalAttentionModel(num_classes=2).to(device)
model.load_state_dict(torch.load('model/original-attention-lr-0.001.pth'))
model.eval()


with torch.no_grad():
    total = 0
    correct = 0
    avg_loss = 0.0
    for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        x_test, labels_test, _, _ = data
        x_test, labels_test = x_test.to(device), labels_test.to(device)
        x_test = torch.unsqueeze(x_test, dim=2)
        x_test = x_test.repeat(1, 1, 46, 1)
        x_test = torch.permute(x_test, (0, 3, 1, 2))        # (?, 1024, 46, 46)

        pred_test, __, __, __ = model(x_test)
        predict = torch.argmax(pred_test, 1)
        total += labels_test.size(0)
        correct += torch.eq(predict, labels_test).sum().double().item()
        avg_loss += loss_func(pred_test, labels_test).item() / len(testloader)

    print(f"\tAccuracy on test data: {100 * correct / total:.2f}%")
    print(f"\tAvg Loss on test data: {avg_loss:.2f}")


