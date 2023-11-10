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


class NoisyMNIST(torchvision.datasets.MNIST):
    def __init__(self, noise, size, train):
        super(NoisyMNIST, self).__init__(root='./data', train=train, download=True,
                                         transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                       transforms.ToTensor()]))

        self.newlabels = {}
        indices = torch.randperm(size)[:int(noise*size)].tolist()
        for idx in indices:
            self.newlabels[idx] = int(random.random()*10)

    def __getitem__(self, idx):
        img, label = super(NoisyMNIST, self).__getitem__(idx)
        return img, self.newlabels.get(idx, label)


def get_MNIST(noise=0.0, size=None, train=True):
    if size:
        return Subset(NoisyMNIST(noise, size, train), range(size))
    else:
        if train:
            return NoisyMNIST(noise, 60000, True)
        else:
            return NoisyMNIST(noise, 10000, False)


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


def train_model(model, optimizer, criterion, train_loader, test_loader, epochs):

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
            predictions = torch.argmax(outputs, dim=1)
            for j in range(len(predictions)):
                if predictions[j] == labels[j]:
                    correct += 1
        train_acc.append(correct / len(train_loader.dataset))

        # test
        model.eval()
        correct = 0
        with torch.no_grad():
            for _inputs, _labels in test_loader:
                inputs = _inputs.to(device)
                labels = _labels.to(device)
                predictions = torch.argmax(model(inputs), dim=1)
                for j in range(len(predictions)):
                    if predictions[j] == labels[j]:
                        correct += 1
        test_acc.append(correct / len(test_loader.dataset))

        if epoch % 100 == 99:
            print("Completed " + str(epoch+1) + " epochs.")

    return train_acc, test_acc, g_e


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


if __name__ == '__main__':

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser(description='...')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--var', type=float, default=0.000001,
                        help='langevin noise variance')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='minibatch size for training and testing')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs for training')

    # experiment conditions
    parser.add_argument('--dataset_size', type=int, default=60000,
                        help='subset of dataset used')
    parser.add_argument('--noise', type=float, action='append',
                        help='amount of noise to add to labels')

    # experiment name
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='name of folder to save the results')

    args = parser.parse_args()
    print(args)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    variance = args.var
    dataset_size = args.dataset_size
    noise = args.noise if args.noise else [0.0, 0.5]
    experiment_name = args.experiment_name

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

    for p in noise:
        print("Running for p = "+str(p))

        # configure datasets and dataloaders
        trainset = get_MNIST(noise=p, size=dataset_size, train=True)
        testset = get_MNIST(noise=p, train=False)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(trainset, batch_size=1000, num_workers=num_workers)

        # initialize stuff for our algorithm
        model = make_mnist_alexnet().to(device)
        optimizer = NewSGLD(model.parameters(), lr=lr, variance=variance, device=device)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        # train
        train_acc, test_acc, g_e = train_model(model, optimizer, criterion, train_loader, test_loader, epochs)

        sum_term = [0]
        for e in g_e:
            sum_term.append(sum_term[-1] + e)
        sum_term.pop(0)

        # plot
        fig1ax.plot(range(epochs), train_acc, label='p='+str(p))
        fig2ax.plot(range(epochs), test_acc, label='p='+str(p))
        fig3ax.plot(range(epochs), [a-b for a, b in zip(train_acc, test_acc)], label='p='+str(p))
        coef = 8.12/dataset_size * lr/math.sqrt(variance)  # depends on whether stochastic or not
        fig4ax.plot(range(len(sum_term)), [math.sqrt(i) * coef for i in sum_term], label='p='+str(p))
        fig5ax.plot(range(len(g_e)), g_e, label='p='+str(p))
        fig, ax = p_to_fig[p]
        batches = (dataset_size - 1) // batch_size + 1
        ax.plot(range(batches-1, epochs*batches, batches), train_acc, label='train accuracy')
        ax.plot(range(batches-1, epochs*batches, batches), [abs(a-b) for a, b in zip(train_acc, test_acc)], label=r'|$err_{gen}(S)$|')
        ax.plot(range(len(sum_term)), [math.sqrt(i) * coef for i in sum_term], label=r'$err_{gen}$ bound')
        ax.legend()
        fig.savefig('experiments/' + experiment_name + '/noise' + str(p) + '.png')

    fig1ax.legend()
    fig2ax.legend()
    fig3ax.legend()
    fig4ax.legend()
    fig5ax.legend()
    fig1.savefig('experiments/'+experiment_name+'/train_accuracy.png')
    fig2.savefig('experiments/'+experiment_name+'/test_accuracy.png')
    fig3.savefig('experiments/'+experiment_name+'/gen_error.png')
    fig4.savefig('experiments/'+experiment_name+'/gen_error_bound.png')
    fig5.savefig('experiments/'+experiment_name+'/average_squared_gradient_norm.png')

