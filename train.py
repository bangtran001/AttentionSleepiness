from torchinfo import summary
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
import json
import torch
import utils
from torch.utils.data import DataLoader, random_split
from model import SDM_Embedding


'''
GLOBAL variables
'''
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

BATCH_SIZE = 128
EPOCH = 20
LEARNING_RATE = 1e-3
SELECTED_TASK = 10   # 0 for all task
N_RESPONSES = len([k for k, v in response_task_map.items() if v == SELECTED_TASK]) if SELECTED_TASK > 0 else len(response_task_map.keys())

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
''''''''''''''''''''''''''''''''''''''''''''''''''''''


ds = utils.SleepinessDataset(device=device, selected_task=SELECTED_TASK)
n_samples = len(ds)
n_test_samples = int(n_samples*.30)
n_train_samples = n_samples - n_test_samples
train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])

train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

# 3-Construct a model and assign it to selected DEVICE
my_model = SDM_Embedding(SELECTED_TASK).to(device)
summary(my_model, (BATCH_SIZE, N_RESPONSES, 32, 32))

loss_func = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
optimizer = torch.optim.Adam(params=my_model.parameters(), lr=LEARNING_RATE)


# 4-Train the model
print('Training Start ...')
total_batch = len(train_ds) // BATCH_SIZE

training_accuracy = list()
training_loss = list()
testing_accuracy = list()
testing_loss = list()
optimizer.zero_grad()  # <= initialization of the gradients
for epoch in range(EPOCH):
    # training
    print(f"----- epoch {epoch + 1}/{EPOCH} -----")
    train_targets = torch.empty(0)
    train_predicts = torch.empty(0)

    avg_cost = 0
    my_model.train()
    for i, (batch_X, batch_Y) in enumerate(train_dl):
        X = batch_X  # (N, 1024)
        Y = batch_Y
        train_targets = torch.cat((train_targets, Y))
        X = X.view(-1, N_RESPONSES, 32, 32)
        X = X.to(device)
        Y = Y.to(device)

        #optimizer.zero_grad()  # <= initialization of the gradients

        # forward propagation
        hypothesis = my_model(X)
        loss = loss_func(hypothesis, Y)  # <= compute the loss function

        # Backward propagation
        loss.backward()     # <= compute the gradient of the loss/cost function
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 5)

        optimizer.step()    # <= Update the gradients

        # Print some performance to monitor the training
        preds = hypothesis.data.max(dim=1)[1]
        train_predicts = torch.cat((train_predicts, preds.cpu().detach()))
        step_acc = (preds.data == Y.data).float().mean().item()
        print(f"    > [step {i + 1}/{len(train_dl)}]: loss={loss} \tacc={step_acc}")
        avg_cost += loss.data / total_batch

    train_accu = (train_predicts.data == train_targets.data).float().mean().item()
    training_accuracy.append(train_accu)
    training_loss.append(avg_cost.cpu().detach())

    #validating the model after each epoch
    my_model.eval()
    test_predicts = torch.empty(0)
    test_targets = torch.empty(0)
    for i, (batch_X, batch_Y) in enumerate(validation_dl):
        X = batch_X
        Y = batch_Y
        test_targets = torch.cat((test_targets, Y))

        X = X.view(-1, N_RESPONSES, 32, 32)
        X = X.to(device)
        Y = Y.to(device)
        hypothesis = my_model(X)
        predicts = hypothesis.data.max(dim=1)[1]
        test_predicts = torch.cat((test_predicts, predicts.cpu().detach()))
    test_accu = (test_predicts.data == test_targets.data).float().mean().item()
    testing_accuracy.append(test_accu)

    print(f"    Avg loss: {avg_cost}; Train Accuracy: {train_accu}; Test Accuracy: {test_accu}\n")


print('Training Finished!')




'''
Save & plot the history of training
'''
# jsonfile = open('model/train-hist.json', 'w')
# json.dump({'loss':training_loss, 'train_acc': training_accuracy, 'test_acc': testing_accuracy}, jsonfile)
# jsonfile.close()

# plotting the train
import matplotlib.pyplot as plt
plt.plot([i for i in range(1, EPOCH + 1)], training_loss, label='Training loss')
plt.plot([i for i in range(1, EPOCH + 1)], training_accuracy, label='Training Accurracy')
plt.plot([i for i in range(1, EPOCH + 1)], testing_accuracy, label='Test Accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig('model/train-hist.png')













