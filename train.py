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
from model import SDM, AttnSDM, ICASSP_Model
import numpy as np

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
N_EPOCHS = 100
LEARNING_RATE = 1e-4

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

""" Train the ICASSP2022 model """
def train_ICASSP(args):
    SELECTED_TASK = args.task
    if SELECTED_TASK == 0:
        N_RESPONSES = len(list(response_task_map.keys()))
    else:
        N_RESPONSES = len([k for k, v in response_task_map.items() if v == SELECTED_TASK])
    ds = utils.SleepinessDataset(device=device, selected_task=SELECTED_TASK)
    train_ds, test_ds = random_split(ds, [(len(ds)-int(len(ds)*.2)), int(len(ds)*.2)])

    #oversampling train_ds
    


    class_count = np.unique(targets, return_counts=True)[1]
    print(class_count)
    return None

    train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    # total = 0
    # positive = 0
    # negative = 0
    # for i, (X, labels) in enumerate(validation_dl):
    #     total += labels.size(0)
    #     positive+= (labels == 1).sum().double().item()
    #     negative+= (labels == 0).sum().double().item()
    # print(f'Total: {total}\nPositive/Negative: {positive}/{negative}')
    # print(f'print: {100*negative/total}%%')
    #
    # return None

    # 3-Construct a model and assign it to selected DEVICE
    my_model = ICASSP_Model(SELECTED_TASK).to(device)
    summary(my_model, (BATCH_SIZE, N_RESPONSES, 32, 32))
    # class_weights = ds.get_class_weights()

    #loss_func = torch.nn.CrossEntropyLoss()  # Softmax is internally computed.
    #optimizer = torch.optim.Adam(params=my_model.parameters(), lr=LEARNING_RATE)
    # loss_func = torch.nn.BCELoss(weight=class_weights)  # softmax() + binary CE
    loss_func = torch.nn.BCELoss()  # softmax() + binary CE
    optimizer = optim.SGD(my_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    #--------------
    def train_one_epoch(epoch):     # return (loss, accuracy)
        running_loss = 0
        correct = 0
        total = 0
        for i, (X, labels) in enumerate(train_dl):
            X = X.view(-1, N_RESPONSES, 32, 32)     # (?, N, 1024) --> (?, N, 32, 32)
            X, labels = X.to(device), labels.to(device)

            # forward propagation
            my_model.train()
            hypothesis = my_model(X)
            loss = loss_func(torch.squeeze(hypothesis), labels.float())  # <= compute the loss function
            running_loss += loss.item()

            # Backward propagation
            loss.backward()
            optimizer.step()

            # calculate current accuracy
            my_model.eval()
            pred = my_model(X)
            predict = torch.squeeze(torch.round(pred).int())
            total += labels.size(0)
            correct += predict.eq(labels).sum().double().item()

        running_loss = running_loss/len(train_dl)
        running_acc = correct/total
        return running_loss, running_acc
    #--------------
    def test_one_epoch(epoch): # return (loss, accuracy)
        my_model.eval()
        running_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (X, labels) in enumerate(validation_dl):
                X = X.view(-1, N_RESPONSES, 32, 32) # (?, N, 1024) --> (?, N, 32, 32)
                X, labels = X.to(device), labels.to(device)

                pred = my_model(X)
                loss = loss_func(torch.squeeze(pred), labels.float())
                running_loss += loss.item()
                predict = torch.squeeze(torch.round(pred).int())
                total += labels.size(0)
                correct += predict.eq(labels).sum().double().item()
            running_loss = running_loss/len(validation_dl)
            running_acc = correct/total
            return running_loss, running_acc
    #--------------

    # 4-Train the model
    print('Training Start ...')
    training_accuracy = list()
    training_loss = list()
    testing_accuracy = list()
    testing_loss = list()
    optimizer.zero_grad()  # <= initialization of the gradients
    for epoch in range(N_EPOCHS):
        # training
        print(f"----- epoch {epoch + 1}/{N_EPOCHS} -----")
        my_model.train()
        tr_loss, tr_acc = train_one_epoch(epoch)

        my_model.eval()
        te_loss, te_acc = test_one_epoch(epoch)

        print(f"    Train loss: {tr_loss}     Train accuracy: {tr_acc}")
        print(f"    Test loss: {te_loss}     Test accuracy: {te_acc}")
        training_accuracy.append(tr_acc)
        training_loss.append(tr_loss)
        testing_accuracy.append(te_acc)
        testing_loss.append(te_loss)
    print('Training Finished!')

    '''
    Save & plot the history of training
    '''
    # plotting the train
    import matplotlib.pyplot as plt
    plt.plot(training_accuracy, '-', label='Training accuracy')
    plt.plot(training_loss, '-', label='Training loss')
    plt.plot(testing_accuracy, '-', label='Test accuracy')
    plt.plot(testing_loss, '-', label='Test loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('model/train-icassp-hist.png')

""" Train none-attention-model"""
def train_SDM(args):
    assert (args.task in range(0, 12)), "The SELECTED_TASK must be between 0 and 12"

    SELECTED_TASK = args.task
    if SELECTED_TASK == 0:
        N_RESPONSES = len(list(response_task_map.keys()))
    else:
        N_RESPONSES = len([k for k, v in response_task_map.items() if v == SELECTED_TASK])

    ds = utils.SleepinessDataset(device=device, selected_task=SELECTED_TASK)
    n_samples = len(ds)
    n_test_samples = int(n_samples*.25)
    n_train_samples = n_samples - n_test_samples
    train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])

    train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    # 3-Construct a model and assign it to selected DEVICE
    my_model = SDM(SELECTED_TASK).to(device)
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
    for epoch in range(N_EPOCHS):
        # training
        print(f"----- epoch {epoch + 1}/{N_EPOCHS} -----")
        train_targets = torch.empty(0)
        train_predicts = torch.empty(0)

        avg_cost = 0
        my_model.train()
        for i, (batch_X, batch_Y) in tqdm(enumerate(train_dl)):
            X = batch_X  # (N, 1024)
            Y = batch_Y
            train_targets = torch.cat((train_targets, Y))
            X = X.view(-1, N_RESPONSES, 32, 32)
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()  # <= initialization of the gradients

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
    plt.plot([i for i in range(1, N_EPOCHS + 1)], training_loss, label='Training loss')
    plt.plot([i for i in range(1, N_EPOCHS + 1)], training_accuracy, label='Training Accurracy')
    plt.plot([i for i in range(1, N_EPOCHS + 1)], testing_accuracy, label='Test Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('model/train-hist.png')

""" Train attention-model """
def train_AttnSDM(args):
    net = AttnSDM(im_size=32, num_classes=2)
    summary(net, (BATCH_SIZE, 46, 32, 32))

    device = torch.device("cuda")
    device_ids = [0, ]

    criterion = nn.CrossEntropyLoss()
    epochs = N_EPOCHS
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    # optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ds = utils.SleepinessDataset(device=device, selected_task=0)
    n_samples = len(ds)
    n_test_samples = int(n_samples*.25)
    n_train_samples = n_samples - n_test_samples
    train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])

    train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    training_accuracy = list()
    training_loss = list()
    testing_accuracy = list()
    testing_loss = list()

    class_weights, class_counts = ds.get_class_weights(train_ds.indices)
    print('train class_weights:', class_weights, class_counts)

    cw, cc = ds.get_class_weights(test_ds.indices)
    print('test class_weights:', cw, cc)



    step = 0
    running_avg_accuracy = 0
    aug = 0

    for epoch in range(N_EPOCHS):
        print("\nepoch %d learning rate %f" % (epoch+1, optimizer.param_groups[0]['lr']))
        for i, data in enumerate(train_dl, start=0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.view(-1, 46, 32, 32)  # reshape from N x 1024 --> N x 32 x 32
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            pred, __, __, __ = model(inputs)

            # backward
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                model.eval()
                pred, __, __, __ = model(inputs)
                predict = torch.argmax(pred - class_weights.to(device), 1)
                total = labels.size(0)
                correct = torch.eq(predict, labels).sum().double().item()
                accuracy = correct / total
                running_avg_accuracy = 0.9 * running_avg_accuracy + 0.1 * accuracy
                print("\t >loss %.4f train accuracy %.2f%% running avg accuracy %.2f%%"
                      % (loss.item(), (100 * accuracy), (100 * running_avg_accuracy)))
                #logging
                training_accuracy.append(accuracy)
                training_loss.append(loss.item())
            step += 1

        model.eval()
        total = 0
        correct = 0


        with torch.no_grad():
            # log scalars
            for i, data in enumerate(validation_dl, 0):
                inputs_test, labels_test = data
                inputs_test = inputs_test.view(-1, 46, 32, 32)
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                pred_test, _, _, _ = model(inputs_test)
                predict = torch.argmax(pred_test - class_weights.to(device), 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
            print("\t > accuracy on test data: %.2f%%" % (100 * correct / total))
            testing_accuracy.append(correct / total)

    # plotting the train
    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(training_loss))], training_loss, label='Training loss')
    plt.plot([i for i in range(len(training_accuracy))], training_accuracy, label='Training Accurracy')
    plt.plot([i for i in range(len(testing_accuracy))], testing_accuracy, label='Test Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('model/train-attention-hist.png')


""" Main function """
if __name__=='__main__':
    custom_parser = argparse.ArgumentParser(description='Training Sleepiness Classification Model')
    custom_parser.add_argument('--task', type=int, default=1)
    custom_parser.add_argument('--attention', type=int, default=0)
    custom_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    custom_parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
    custom_parser.add_argument('--batch_size', type=int, default=20, help='Batch size')

    custom_args, _ = custom_parser.parse_known_args()

    if custom_args.attention > 0:
        train_AttnSDM(custom_args)
    else:
        train_SDM(custom_args)

    # train_ICASSP(custom_args)











