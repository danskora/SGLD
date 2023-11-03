import argparse
import torch
import data
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import transforms
from sgld_optimizer import NewSGLD
from torch.utils.data import Subset
import os
import math


def make_mnist_alexnet():
    act = torch.nn.SiLU
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        act(),
        nn.MaxPool2d(2, stride=2, padding=0),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        act(),
        nn.MaxPool2d(2, stride=2, padding=0),
        # nn.Conv2d(128, 256, kernel_size=2, padding="same"),
        # act(),
        # nn.Conv2d(256, 128, kernel_size=2, padding="same"),
        # act(),
        nn.Conv2d(128, 64, kernel_size=2, padding="same"),
        act(),
        nn.Flatten(),
        nn.Linear(4096, 256),
        act(),
        nn.Linear(256, 256),
        act(),
        nn.Linear(256, 10),
    )


def g_e(model, optimizer, criterion, dataset, nsamples):
    indices = torch.randperm(len(dataset))[:nsamples].tolist()
    total = 0
    for idx in indices:
        datapoint, label = dataset[idx]
        datapoint.unsqueeze_(0)
        optimizer.zero_grad()
        loss = criterion(model(datapoint), torch.tensor([label]))
        loss.backward()
        for p in model.parameters():
            total += torch.sum(torch.mul(p.grad, p.grad))
    return total/nsamples


if __name__ == '__main__':

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # read in arguments
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--var', type=float, default=0.000004,
                        help='langevin noise variance')
    parser.add_argument('--batch_size', type=int, default=60000,
                        help='minibatch size for training and testing')
    parser.add_argument('--dataset_size', type=int, default=60000,
                        help='subset of MNIST used')
    parser.add_argument('--epochs', type=int, default=5,  # 500 in actual experiment
                        help='number of epochs for training')
    parser.add_argument('--dataset_samples', type=int, default=1,
                        help='datasets used to estimate the expectation over S')
    parser.add_argument('--parameter_samples', type=int, default=1,
                        help='models used to estimate the expectation over w')
    parser.add_argument('--train_data_samples', type=int, default=200,
                        help='datapoints used to estimate g_e')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='fraction of labels to add noise to')
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='name of folder to save the results')
    args = parser.parse_args()
    print(args)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    variance = args.var
    dataset_size = args.dataset_size
    dataset_samples = args.dataset_samples
    parameter_samples = args.parameter_samples
    train_data_samples = args.train_data_samples
    experiment_name = args.experiment_name
    noise = args.noise

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    validation_size = 0
    num_workers = 1

    # make necessary subdirectories
    if not os.path.exists('experiments'):
        os.mkdir('experiments')
    if not os.path.exists('experiments/' + experiment_name):
        os.mkdir('experiments/' + experiment_name)

    # initialize figure
    fig1 = plt.figure()
    fig1ax = fig1.add_subplot()
    fig1ax.set_xlabel('Step')
    fig1ax.set_ylabel('Error Bound')
    fig1.suptitle('generalization error bound')

    total_sum_term = [0 for i in range(epochs+1)]  # change if we aren't doing GLD

    for a in range(dataset_samples):  # this does nothing right now since I just want to do one trial with the full set

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                              transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                            transforms.ToTensor()]))
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                              transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                            transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=num_workers)  # use for g_e

        for b in range(parameter_samples):

            model = make_mnist_alexnet().to(device)
            optimizer = NewSGLD(model.parameters(), lr=lr, variance=variance)
            scheduler = None
            criterion = nn.CrossEntropyLoss()

            sum_term = [0]
            for epoch in range(epochs):
                for inputs, labels in train_loader:
                    sum_term.append(sum_term[-1] + g_e(model, optimizer, criterion, train_loader.dataset, train_data_samples))
                    total_sum_term[epoch+1] = total_sum_term[epoch+1] + sum_term[-1]  # change if we aren't doing GLD

    coef = 2 * math.sqrt(2) / dataset_size * lr / math.sqrt(variance)
    bound = [coef * math.sqrt(i / (dataset_samples*parameter_samples)) for i in total_sum_term]  # change if we aren't doing GLD
    fig1ax.plot(range(len(bound)), bound)
    fig1.savefig('experiments/' + experiment_name + '/better_gen_error_bound.png')






