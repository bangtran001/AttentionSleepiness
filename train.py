import argparse
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import sys
from torch.nn import functional as F
from tqdm import tqdm
import json
import torch
import utils
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from model import Simple_SDM, Adaptive_AttnSDM, GeMAPS_AttnSDM, ICASSP_Model
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

BATCH_SIZE = 64
N_EPOCHS = 100
LEARNING_RATE = 1e-5

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

""" Train the ICASSP2022 model """
def train_ICASSP(args):
    LEARNING_RATE = 1e-5

    SELECTED_TASK = args.task
    if SELECTED_TASK == 0:
        N_RESPONSES = len(list(response_task_map.keys()))
    else:
        N_RESPONSES = len([k for k, v in response_task_map.items() if v == SELECTED_TASK])
    ds = utils.HuBERTEmbedDataset(device=device, selected_task=SELECTED_TASK)
    train_ds, test_ds = random_split(ds, [(len(ds)-int(len(ds)*.2)), int(len(ds)*.2)])
    train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    # 3-Construct a model and assign it to selected DEVICE
    my_model = ICASSP_Model(SELECTED_TASK).to(device)
    summary(my_model, (BATCH_SIZE, N_RESPONSES, 32, 32))

    #class_weights, _ = ds.get_class_weights(train_ds.indices)
    class_weights, _ = ds.get_class_weights()
    # loss_func = torch.nn.BCELoss()  # softmax() + binary CE
    loss_func = torch.nn.CrossEntropyLoss() # softmax included
    optimizer = optim.RMSprop(my_model.parameters(), lr=LEARNING_RATE)

    #--------------
    def train_one_epoch(epoch):     # return (loss, accuracy)
        running_loss = 0
        correct = 0
        total = 0
        for i, (X, labels, _, _) in enumerate(train_dl):
            X = X.view(-1, N_RESPONSES, 32, 32)     # (?, N, 1024) --> (?, N, 32, 32)
            X, labels = X.to(device), labels.to(device)

            # forward propagation
            my_model.train()
            hypothesis = my_model(X)
            loss = loss_func(hypothesis, labels)  # <= compute the loss function
            running_loss += loss.item()

            # Backward propagation
            loss.backward()
            optimizer.step()

            # calculate current accuracy
            my_model.eval()
            pred = my_model(X)
            #predict = torch.squeeze(torch.round(pred).int())
            predict = torch.argmax(pred - class_weights.to(device), 1)
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
            for i, (X, labels, _, _) in enumerate(validation_dl):
                X = X.view(-1, N_RESPONSES, 32, 32) # (?, N, 1024) --> (?, N, 32, 32)
                X, labels = X.to(device), labels.to(device)

                pred = my_model(X)
                loss = loss_func(pred, labels)
                running_loss += loss.item()
                #predict = torch.squeeze(torch.round(pred).int())
                predict = torch.argmax(pred - class_weights.to(device), 1)
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
        print(f"----- epoch {epoch + 1}/{N_EPOCHS} (learning rate = {LEARNING_RATE}-----")
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

    ds = utils.HuBERTEmbedDataset(device=device, selected_task=SELECTED_TASK)
    n_samples = len(ds)
    n_test_samples = int(n_samples*.25)
    n_train_samples = n_samples - n_test_samples
    train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])

    train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    # 3-Construct a model and assign it to selected DEVICE
    my_model = Simple_SDM(SELECTED_TASK).to(device)
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
        for i, (batch_X, batch_Y, _, _) in tqdm(enumerate(train_dl), total=len(train_dl)):
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
        for i, (batch_X, batch_Y, _, _) in enumerate(validation_dl):
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
    my_model = AttnSDM(im_size=32, num_classes=2)
    summary(my_model, (BATCH_SIZE, 46, 32, 32))

    device_ids = [0, ]
    loss_func = nn.CrossEntropyLoss()
    epochs = N_EPOCHS
    model = nn.DataParallel(my_model, device_ids=device_ids).to(device)
    loss_func.to(device)
    # optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ds = utils.HuBERTEmbedDataset(device=device, selected_task=0)
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
    step = 0
    running_avg_accuracy = 0
    aug = 0

    for epoch in range(N_EPOCHS):
        print("\nepoch %d learning rate %f" % (epoch+1, optimizer.param_groups[0]['lr']))
        for i, data in enumerate(train_dl, start=0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs, labels, _, _ = data
            inputs = inputs.view(-1, 46, 32, 32)  # reshape from N x 1024 --> N x 32 x 32
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            pred, __, __, __ = model(inputs)

            # backward
            loss = loss_func(pred, labels)
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
                inputs_test, labels_test, _, _ = data
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

""" Train Adaptive attention-model """
def train_AdaptiveAttnSDM(args):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    apply_attention = True if args.attention !=0 else False
    age_gender = True if args.age_gender !=0 else False

    if args.attention:
        if age_gender:
            saved_model_path = 'model/attn_model_weights_lr_' + str(args.learning_rate) + '-' + str(args.feature) + '.pth'
        else:
            saved_model_path = 'model/attn_model_weights_lr_' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.pth'
    else:
        if age_gender:
            saved_model_path = 'model/noattn_model_weights_lr_' + str(args.learning_rate) + '-' + str(args.feature) + '.pth'
        else:
            saved_model_path = 'model/noattn_model_weights_lr_' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.pth'

    my_model = Adaptive_AttnSDM(num_classes=2,
                                apply_attention=apply_attention,
                                age_gender=age_gender)
    device_ids = [0, ]
    loss_func = nn.CrossEntropyLoss()
    model = nn.DataParallel(my_model, device_ids=device_ids).to(device)
    loss_func.to(device)
    # optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    ds = utils.HuBERTEmbedDataset(device=device, selected_task=args.task)
    summary(my_model, (args.batch_size, 1, 46, 1024),
            age=torch.ones(args.batch_size, 1),
            gender=torch.zeros(args.batch_size, 1))
    n_samples = len(ds)
    n_test_samples = int(n_samples*.20)
    n_train_samples = n_samples - n_test_samples
    train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])
    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=16)
    validation_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=6)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    training_accuracy = list()
    training_loss = list()
    testing_accuracy = list()
    testing_loss = list()
    # use classweight in the case don't use undersampling method
    class_weights, class_counts = ds.get_class_weights(train_ds.indices)
    best_loss = sys.float_info.max
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch+1}/{args.epoch} @ lr={optimizer.param_groups[0]['lr']} ------")
        avg_loss = 0.0
        avg_acc = 0.0
        for i, (inputs, labels, ages, genders) in tqdm(enumerate(train_dl, start=0), total=len(train_dl), desc='Training step'):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs = torch.unsqueeze(inputs, dim=1) # reshape from (?, 46, 1024) --> (?, 1, 46, 1024)
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
            avg_loss += loss.item()/len(train_dl)

            model.eval()
            pred, _, _, _ = model(inputs.float(), ages, genders)

            # use classweight in the case unused undersampling method
            predict = torch.argmax(pred - class_weights.to(device), 1)
            # predict = torch.argmax(pred, 1)
            total = labels.size(0)
            correct = torch.eq(predict, labels).sum().double().item()
            avg_acc += (correct/total)/len(train_dl)

        # save the best model based on training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(my_model.state_dict(), saved_model_path)

        print(f"\tavg. train loss={avg_loss:.4f} \ttrain accuracy={100*avg_acc:.2f}%")

        # for logging
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
                ages_test, genders_test = ages_test.float().to(device), genders_test.float().to(device)
                ages_test = torch.unsqueeze(ages_test, dim=1)
                genders_test = torch.unsqueeze(genders_test, dim=1)

                pred_test, _, _, _ = model(inputs_test.float(), ages_test, genders_test)

                avg_loss += loss_func(pred_test, labels_test).item()/len(validation_dl)
                # use classweight in the case don't use undersampling method
                predict = torch.argmax(pred_test - class_weights.to(device), 1)
                #predict = torch.argmax(pred_test, 1)
                test_total += labels_test.size(0)
                test_correct += torch.eq(predict, labels_test).sum().double().item()
        print(f"\tavg. test loss={avg_loss:.4f} \ttest accuracy={100 * (test_correct/test_total):.2f}%")
        #logging
        testing_loss.append(avg_loss)
        testing_accuracy.append(test_correct / test_total)

    print("Training is Done!")
    return 0

    # -- save the training history
    if args.attention:
        if age_gender:
            jsonfile = open('image/train-attention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '.json', 'w')
        else :
            jsonfile = open('image/train-attention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.json', 'w')
    else:
        if age_gender:
            jsonfile = open('image/train-noattention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '.json', 'w')
        else:
            jsonfile = open(
                'image/train-noattention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.json', 'w')

    # -- save training history to json file
    json.dump({'train_loss': training_loss,
               'train_acc': training_accuracy,
               'test_loss': testing_loss,
               'test_acc': testing_accuracy}, jsonfile)
    jsonfile.close()

    # --- plotting the train
    import matplotlib.pyplot as plt
    plt.plot(training_loss, label='Training loss')
    #plt.plot(testing_loss, label='Testing loss')
    plt.plot(training_accuracy, label='Training Accurracy')
    plt.plot(testing_accuracy, label='Test Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    if args.attention:
        if age_gender:
            plt.savefig('image/train-attention-hist-lr-'+str(args.learning_rate)+ '-' + str(args.feature) +'.png')
        else:
            plt.savefig('image/train-attention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.png')
    else:
        if age_gender:
            plt.savefig('image/train-noattention-hist-lr-'+str(args.learning_rate)+ '-' + str(args.feature) +'.png')
        if not age_gender:
            plt.savefig('image/train-noattention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.png')




""" Train attention-model with GeMAPS input"""
def train_GEMAPSAttnSDM(args):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    apply_attention = True if args.attention !=0 else False
    age_gender = True if args.age_gender !=0 else False

    my_model = GeMAPS_AttnSDM(num_classes=2, apply_attention=apply_attention)
    device_ids = [0, ]
    loss_func = nn.CrossEntropyLoss()
    model = nn.DataParallel(my_model, device_ids=device_ids).to(device)
    loss_func.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    ds = utils.GeMAPSDataset(device=device, selected_task=args.task)
    summary(my_model, (args.batch_size, 1, 46, 88))

    n_samples = len(ds)
    n_test_samples = int(n_samples*.20)
    n_train_samples = n_samples - n_test_samples
    train_ds, test_ds = random_split(ds, [n_train_samples, n_test_samples])

    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    print(f"--------\nTraining samples:{len(train_ds)}\nValidating samples:{len(test_ds)}\n--------")

    # --- save the model ---
    if args.attention:
        saved_model_path = 'model/attn_model_weights_lr_' + str(args.learning_rate) + '-' + str(args.feature) + '.pth'
    else:
        saved_model_path = 'model/noattn_model_weights_lr_' + str(args.learning_rate) + '.pth'

    training_accuracy = list()
    training_loss = list()
    testing_accuracy = list()
    testing_loss = list()
    # use classweight in the case don't use undersampling method
    # class_weights, class_counts = ds.get_class_weights(train_ds.indices)
    c1, c2, c3 = torch.zeros(46, 11), torch.zeros(46, 11), torch.zeros(46, 11)
    best_loss = sys.float_info.max
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch+1}/{args.epoch} @ lr={optimizer.param_groups[0]['lr']} ------")
        avg_loss = 0.0
        avg_acc = 0.0
        for i, (inputs, labels, ages, genders) in tqdm(enumerate(train_dl, start=0), total=len(train_dl), desc='Training step'):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs = torch.unsqueeze(inputs, dim=1) # reshape from (?, 46, 88) --> (?, 1, 46, 88)
            inputs, labels = inputs.to(device), labels.to(device)
            ages, genders = ages.float().to(device), genders.float().to(device)

            # forward
            pred, __, __, __ = model(inputs.float())

            # backward
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            # print(f"  > step {i+1}/{len(train_dl)} loss={loss.item():.4f}")

            avg_loss += loss.item()/len(train_dl)
            model.eval()
            pred, c1, c2, c3 = model(inputs.float())
            # use classweight in the case unused undersampling method
            # predict = torch.argmax(pred - class_weights.to(device), 1)
            predict = torch.argmax(pred, 1)
            total = labels.size(0)
            correct = torch.eq(predict, labels).sum().double().item()
            avg_acc += (correct/total)/len(train_dl)
        print(f"\tavg. train loss={avg_loss:.4f} \ttrain accuracy={100*avg_acc:.2f}%")

        # save the best model based on training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(my_model.state_dict(), saved_model_path)

        #for logging
        training_accuracy.append(avg_acc)
        training_loss.append(avg_loss)

        model.eval()
        with torch.no_grad():
            test_total = 0
            test_correct = 0
            avg_loss = 0.0
            for i, (inputs_test, labels_test, ages_test, genders_test) in enumerate(validation_dl, 0):
                inputs_test = torch.unsqueeze(inputs_test, dim=1)  # reshape from (?, 46, 88) --> (?, 1, 46, 88)
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                ages_test, genders_test = ages_test.float().to(device), genders_test.float().to(device)
                ages_test = torch.unsqueeze(ages_test, dim=1)
                genders_test = torch.unsqueeze(genders_test, dim=1)
                pred_test, _, _, _ = model(inputs_test.float())
                avg_loss += loss_func(pred_test, labels_test).item()/len(validation_dl)
                # use classweight in the case don't use undersampling method
                # predict = torch.argmax(pred_test - class_weights.to(device), 1)
                predict = torch.argmax(pred_test, 1)
                test_total += labels_test.size(0)
                test_correct += torch.eq(predict, labels_test).sum().double().item()
        print(f"\tavg. test loss={avg_loss:.4f} \ttest accuracy={100 * (test_correct/test_total):.2f}%")
        #for logging
        testing_loss.append(avg_loss)
        testing_accuracy.append(test_correct / test_total)

    print("Training is Done!")

    # --- save the model ---
    if args.attention:
        jsonfile = open('image/train-attention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.json', 'w')
    # no attention
    else:
        jsonfile = open('image/train-noattention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.json', 'w')

    json.dump({'train_loss': training_loss,
               'train_acc': training_accuracy,
               'test_loss': testing_loss,
               'test_acc': testing_accuracy}, jsonfile)
    jsonfile.close()

    # --- plotting trainig history
    import matplotlib.pyplot as plt
    plt.plot(training_loss, label='Training loss')
    #plt.plot(testing_loss, label='Testing loss')
    plt.plot(training_accuracy, label='Training Accurracy')
    plt.plot(testing_accuracy, label='Test Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    if args.attention:
        plt.savefig('image/train-attention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.png')
    else:
        plt.savefig('image/train-noattention-hist-lr-' + str(args.learning_rate) + '-' + str(args.feature) + '-nogender.png')


    # invoke utils.plotting_training_curve() to plot atte

""" Main function """
if __name__=='__main__':
    custom_parser = argparse.ArgumentParser(description='Training Sleepiness Classification Model')
    custom_parser.add_argument('--task', type=int, default=0)
    custom_parser.add_argument('--attention', type=int, default=1)
    custom_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    custom_parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    custom_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    custom_parser.add_argument('--feature', type=str, default='HuBERT', help='input features')
    custom_parser.add_argument('--age_gender', type=int, default=1, help='use age and gender')
    custom_args, _ = custom_parser.parse_known_args()

    if custom_args.feature.lower() in ['hubert']:
        train_AdaptiveAttnSDM(custom_args)
    elif custom_args.feature.lower() in ['gemaps', 'egemaps']:
        train_GEMAPSAttnSDM(custom_args)


    # train_ICASSP(custom_args)
    # train_SDM(custom_args)
    # train_AttnSDM(custom_args)
    # exit(0)
















