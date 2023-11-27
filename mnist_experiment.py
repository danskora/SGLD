import argparse
import os
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

from my_helpers import *
from sgld_optimizer import NewSGLD
import data
import models


def calc_g_e(model, optimizer, criterion, dataset):  # doesn't matter but this should run in eval mode i think
    indices = torch.randperm(len(dataset))[:200].tolist()
    total = 0
    for idx in indices:
        datapoint, label = dataset[idx]
        datapoint.unsqueeze_(0)
        optimizer.zero_grad()
        loss = criterion(model(datapoint.to(device)), torch.tensor([label], device=device))
        loss.backward()
        for p in model.parameters():
            total += torch.sum(p.grad ** 2).item()
    return total/200


# def new_calc_g_e(model, optimizer, criterion, train, test):
#     i1 = random.randint(0, len(train)-1)
#     i2 = random.randint(0, len(test)-1)
#     d1 = train[i1]
#     d2 = test[i2]
#     d1.unsqueeze_(0)
#     d2.unsqueeze_(0)
#
#
#
#     # finish this, then run tests on his experiments
#     for idx in indices:
#         datapoint, label = dataset[idx]
#         datapoint.unsqueeze_(0)
#         optimizer.zero_grad()
#         loss = criterion(model(datapoint.to(device)), torch.tensor([label], device=device))
#         loss.backward()
#         for p in model.parameters():
#             total += torch.sum(p.grad ** 2).item()
#     return total/200


def train_model(model, optimizer, scheduler, criterion, train_loader, test_loader, epochs):

    train_acc = []
    test_acc = []
    g_e = []

    for epoch in range(epochs):

        # train
        model.train()
        correct = 0
        for i, (_inputs, _labels) in enumerate(train_loader):
            inputs = _inputs.to(device)
            labels = _labels.to(device)
            g_e.append(calc_g_e(model, optimizer, criterion, train_loader.dataset))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += total_correct(model(inputs), labels)
        train_acc.append(correct / len(train_loader.dataset))

        # test
        model.eval()
        correct = 0
        with torch.no_grad():  # do i need no_grad if not calling loss.backward()?  what about calling criterion()?
            for _inputs, _labels in test_loader:
                inputs = _inputs.to(device)
                labels = _labels.to(device)
                correct += total_correct(model(inputs), labels)  # is it worth converting to "device" just to pass through model?
        test_acc.append(correct / len(test_loader.dataset))

        if epoch % 100 == 99:
            print("Completed " + str(epoch+1) + " epochs.")

    return train_acc, test_acc, g_e


def total_correct(outputs, targets):
    return np.sum(outputs.cpu().detach().numpy().argmax(axis=1) == targets.data.cpu().detach().numpy())


if __name__ == '__main__':

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser(description='...')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--sched', type=int, default=None,
                        help='the number representing which learning rate scheduler we are using')
    parser.add_argument('--std_coef', type=float, default=0.1,
                        help='ratio of langevin noise standard deviation to learning rate (sigma/gamma)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='minibatch size for training and testing')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs for training')

    # experiment conditions
    parser.add_argument('--model', type=str, default='CNN3',
                        help='model type')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset used')
    parser.add_argument('--dataset_size', type=int, default=50000,
                        help='subset of dataset used')
    parser.add_argument('--noise', type=float, action='append',
                        help='amount of noise to add to labels')

    # experiment name
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='name of folder to save the results')

    args = parser.parse_args()
    print(args)

    lr = args.lr
    std_coef = args.std_coef
    batch_size = args.batch_size
    epochs = args.epochs
    model_cfg = args.model
    dataset = args.dataset
    dataset_size = args.dataset_size
    noise = args.noise if args.noise else [0.0, 0.5]
    sched = args.sched
    experiment_name = args.experiment_name

    validation_size = 0
    num_workers = 1

    # set seed
    torch.manual_seed(1)

    # make necessary subdirectories
    if not os.path.exists('experiments'):
        os.mkdir('experiments')
    if not os.path.exists('experiments/' + experiment_name):
        os.mkdir('experiments/' + experiment_name)

    # record important parameters
    with open('experiments/' + experiment_name + '/_experiment_specifications.txt', "w") as file:
        file.write("Model Architecture:  " + model_cfg + "\n")
        file.write("Dataset:  " + dataset + "\n\n")

        file.write("Dataset Size:  " + str(dataset_size) + "\n")
        file.write("Minibatch Size:  " + str(batch_size) + "\n")
        file.write("Epochs:  " + str(epochs) + "\n\n")

        if sched:
            file.write("Learning Rate Decay Format:  " + str(sched) + "\n")
        else:
            file.write("Learning Rate:  " + str(lr) + "\n")
        file.write("sigma/gamma:  " + str(std_coef) + "\n")

    # initialize figure 1 (train acc)
    fig1 = plt.figure()
    fig1ax = fig1.add_subplot()
    fig1ax.set_xlabel('Epoch')
    fig1ax.set_ylabel('Accuracy')
    fig1.suptitle('Train Accuracy')

    # initialize figure 2 (test acc)
    fig2 = plt.figure()
    fig2ax = fig2.add_subplot()
    fig2ax.set_xlabel('Epoch')
    fig2ax.set_ylabel('Accuracy')
    fig2.suptitle('Test Accuracy')

    # initialize figure 3 (gen error)
    fig3 = plt.figure()
    fig3ax = fig3.add_subplot()
    fig3ax.set_xlabel('Epoch')
    fig3ax.set_ylabel('Generalization Error')
    fig3.suptitle('Generalization Error\n'+r'$\mathcal{L}(\hat{w}, \mathcal{D}) - \mathcal{L}(\hat{w}, S)$')

    if std_coef != 0:

        # initialize figure 4 (gen error bound)
        fig4 = plt.figure()
        fig4ax = fig4.add_subplot()
        fig4ax.set_xlabel('Step')
        fig4ax.set_ylabel('Empirical Bound')
        fig4.suptitle('Bound on Expected Generalization Error')

    # initialize figure 5 (g_e)
    fig5 = plt.figure()
    fig5ax = fig5.add_subplot()
    fig5ax.set_xlabel('Step')
    fig5.suptitle('Average Squared Gradient Norm')

    # initialize figures corresponding to each noise value
    p_to_fig = {}
    for p in noise:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('Step')
        fig.suptitle('noise = ' + str(p))
        p_to_fig[p] = (fig, ax)

    # eventually we should remove discrepancy between "steps" and "epochs" and bound test accuracy/loss directly

    channels = 1 if dataset == 'MNIST' else 3

    for p in noise:
        print("Running for p = "+str(p))

        # configure datasets and dataloaders
        trainset = data.get_dataset(dataset, noise=p, size=dataset_size, train=True)
        testset = data.get_dataset(dataset, noise=p, train=False)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=num_workers)

        # initialize stuff for our algorithm
        if model_cfg == 'MLP':
            model = models.MLP(channels).to(device)
        elif model_cfg == 'AlexNet':
            model = models.AlexNet(channels).to(device)
        elif model_cfg == 'CNN2':
            model = models.CNN2(channels).to(device)
        elif model_cfg == 'CNN3':
            model = models.CNN3(channels).to(device)
        else:
            raise NotImplementedError

        if std_coef != 0:
            optimizer = NewSGLD(model.parameters(), lr=lr, std_coef=std_coef, device=device)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        if sched:  # if we specified a scheduler it will override the lr argument
            if sched == 2:
                scheduler = DecayScheduler(optimizer, 0.01, 0.95, 60, floor=None)
            elif sched == 1:
                scheduler = DecayScheduler(optimizer, 0.003, 0.995, 60, floor=0.0005)
            elif sched == 3:
                scheduler = DecayScheduler(optimizer, 0.005, 0.995, 5*dataset_size/batch_size, floor=None)
            else:
                raise NotImplementedError
        else:  # otherwise create a dummy scheduler
            scheduler = DecayScheduler(optimizer, lr, 1, 100, floor=None)

        criterion = torch.nn.CrossEntropyLoss()

        # train
        train_acc, test_acc, g_e = train_model(model, optimizer, scheduler, criterion, train_loader, test_loader, epochs)

        sum_term = [0]
        for e in g_e:
            sum_term.append(sum_term[-1] + e)
        sum_term.pop(0)

        # plot
        fig1ax.plot(range(epochs), train_acc, label='p='+str(p))
        fig2ax.plot(range(epochs), test_acc, label='p='+str(p))
        fig3ax.plot(range(epochs), [a-b for a, b in zip(train_acc, test_acc)], label='p='+str(p))
        if std_coef != 0:
            if dataset_size == batch_size:
                coef = 8.12 / dataset_size / std_coef
            else:
                coef = 2 * math.sqrt(2) / dataset_size / std_coef
            fig4ax.plot(range(len(sum_term)), [math.sqrt(i) * coef for i in sum_term], label='p='+str(p))
        fig5ax.plot(range(len(g_e)), g_e, label='p='+str(p))
        fig, ax = p_to_fig[p]
        batches = (dataset_size - 1) // batch_size + 1
        ax.plot(range(batches-1, epochs*batches, batches), train_acc, label='train accuracy')
        ax.plot(range(batches-1, epochs*batches, batches), [abs(a-b) for a, b in zip(train_acc, test_acc)], label=r'|$err_{gen}(S)$|')
        if std_coef != 0:
            ax.plot(range(len(sum_term)), [math.sqrt(i) * coef for i in sum_term], label=r'$err_{gen}$ bound')
        ax.legend()
        fig.savefig('experiments/' + experiment_name + '/noise' + str(p) + '.png')

    fig1ax.legend()
    fig2ax.legend()
    fig3ax.legend()
    if std_coef != 0:
        fig4ax.legend()
    fig5ax.legend()
    fig1.savefig('experiments/'+experiment_name+'/train_accuracy.png')
    fig2.savefig('experiments/'+experiment_name+'/test_accuracy.png')
    fig3.savefig('experiments/'+experiment_name+'/gen_error.png')
    if std_coef != 0:
        fig4.savefig('experiments/'+experiment_name+'/gen_error_bound.png')
    fig5.savefig('experiments/'+experiment_name+'/average_squared_gradient_norm.png')

