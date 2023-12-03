import argparse
import os

import torch
from torch.utils.data import DataLoader

from decay_scheduler import *
from sgld_optimizer import NewSGLD
from bounds import *
from helpers import *
import data
import models

# replace epochs with steps entirely (excpet for args/params), and maybe only save data every 10 steps or so...
# run this several times to get a better bound
# parameters for # trials for each expectation
# parameters for schedulers?
# check what yadi said about Jensen ineq and resampling z z'
# have separate file for creating plots, and also INCLUDE LOG SCALE for everything

# BUG FOR THESE default parameters (CNN3, CIFAR10)


def train_model(model, optimizer, scheduler, criterion, trainset, testset, epochs):  # make batchsize param

    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=1000, num_workers=num_workers)

    train_acc = []
    test_acc = []
    squared_gradient_norm = []
    li_summand = []
    gradient_discrepancy = []
    banerjee_summand = []

    z_ls = []       # this depends on dataset having 10 label types, so it won't work on CIFAR100 but whatever
    z_prime_ls = []
    for i in range(10):
        indices = [j for j in range(len(trainset)) if trainset[j][1] == i]
        z_ls.append(trainset[indices[0]])
        indices = [j for j in range(len(testset)) if testset[j][1] == i]
        z_prime_ls.append(testset[indices[0]])

    for epoch in range(epochs):

        # train
        model.train()
        correct = 0
        for i, (_inputs, _labels) in enumerate(train_loader):
            inputs = _inputs.to(device)
            labels = _labels.to(device)

            a, b = calc_li_summand(model, optimizer, criterion, train_loader.dataset)
            li_summand.append(a)
            squared_gradient_norm.append(b)

            a, b = calc_banerjee_summand(model, optimizer, criterion, z_ls, z_prime_ls)
            banerjee_summand.append(a)
            gradient_discrepancy.append(b)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += total_correct(model(inputs), labels)
        train_acc.append(correct / len(trainset))

        # test
        model.eval()
        correct = 0
        with torch.no_grad():  # do i need no_grad if not calling loss.backward()?  what about calling criterion()?
            for _inputs, _labels in test_loader:
                inputs = _inputs.to(device)
                labels = _labels.to(device)
                correct += total_correct(model(inputs), labels)  # is it worth converting to "device" just to pass through model?
        test_acc.append(correct / len(testset))

        if epoch % 100 == 99:
            print("Completed " + str(epoch+1) + " epochs.")

    return train_acc, test_acc, squared_gradient_norm, gradient_discrepancy, li_summand, torch.FloatTensor(banerjee_summand)


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
    path = 'experiments/' + experiment_name
    if not os.path.exists(path):
        os.mkdir(path)

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

    # eventually we should remove discrepancy between "steps" and "epochs" and bound test accuracy/loss directly

    channels = 1 if dataset == 'MNIST' else 3

    acc_dict = {}
    bound_dict = {}
    grad_dict = {}

    for p in noise:
        print("Running for p = "+str(p))

        # configure datasets and dataloaders
        trainset = data.get_dataset(dataset, noise=p, size=dataset_size, train=True)
        testset = data.get_dataset(dataset, noise=p, train=False)

        # initialize stuff for our algorithm
        if model_cfg == 'MLP':  # put this all in one thing with yadi's getattr approach
            model = models.MLP(channels).to(device)
        elif model_cfg == 'AlexNet':
            model = models.AlexNet(channels).to(device)
        elif model_cfg == 'CNN2':
            model = models.CNN2(channels).to(device)
        elif model_cfg == 'CNN3':
            model = models.CNN3(channels).to(device)
        else:
            raise NotImplementedError

        optimizer = NewSGLD(model.parameters(), lr=lr, std_coef=std_coef, device=device)

        if sched:  # if we specified a scheduler it will override the lr argument, this is messy !!
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
        train_acc, test_acc, sq_grad_norm, grad_disc, li_summand, banerjee_summand = train_model(model, optimizer, scheduler, criterion, trainset, testset, epochs)

        # plot
        acc_dict[p] = train_acc, test_acc
        bound_dict[p] = (calc_li_bound(li_summand, dataset_size, dataset_size != batch_size),
                         calc_banerjee_bound(banerjee_summand, dataset_size, dataset_size != batch_size))
        grad_dict[p] = sq_grad_norm, grad_disc

        plot_everything(path, p, dataset_size, train_acc, test_acc, li_summand, banerjee_summand)

    plot_acc(path, acc_dict)
    plot_bounds(path, bound_dict)
    plot_grad(path, grad_dict)

