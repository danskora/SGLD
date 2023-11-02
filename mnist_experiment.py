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
    def __init__(self, noise, size):
        super(NoisyMNIST, self).__init__(root='./data', train=True, download=True,
                                         transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                       transforms.ToTensor()]))

        self.newlabels = {}
        indices = torch.randperm(size)[:int(noise*size)].tolist()
        for idx in indices:
            self.newlabels[idx] = int(random.random()*10)

    def __getitem__(self, idx):
        img, label = super(NoisyMNIST, self).__getitem__(idx)
        return img, self.newlabels.get(idx, label)


def get_MNIST(noise=0.0, size=60000, train=True):
    if train:
        return Subset(NoisyMNIST(noise, size), range(size))
    else:
        return torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                        transforms.ToTensor()]))


def g_e(model, optimizer, criterion, dataset):
    # indices = torch.randperm(len(dataset))[:200].tolist()
    # total = 0
    # for idx in indices:
    #     datapoint, label = dataset[idx]
    #     datapoint.unsqueeze_(0)
    #     optimizer.zero_grad()
    #     loss = criterion(model(datapoint), torch.tensor([label]))
    #     loss.backward()
    #     for p in model.parameters():
    #         total += torch.sum(torch.mul(p.grad, p.grad))
    # return total/200
    return 1


def train_model(model, optimizer, criterion, train_loader, test_loader, epochs, path):

    with open(path + '_plotdata.txt', "r") as file:
        progress = int(next(file))
        if progress == 0:
            train_acc = []
            gen_error = []
            sum_term = [0]
        else:
            train_acc = [float(x) for x in next(file).split()]
            gen_error = [float(x) for x in next(file).split()]
            sum_term = [float(x) for x in next(file).split()]

    for epoch in range(progress, epochs, 1):

        # train
        model.train()
        correct = 0
        for i, (inputs, labels) in enumerate(train_loader):
            sum_term.append(sum_term[-1] + g_e(model, optimizer, criterion, train_loader.dataset))
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
            for inputs, labels in test_loader:
                predictions = torch.argmax(model(inputs), dim=1)
                for j in range(len(predictions)):
                    if predictions[j] == labels[j]:
                        correct += 1
        test_acc = correct / len(test_loader.dataset)
        gen_error.append(train_acc[-1] - test_acc)

        # checkpoint
        if epoch % 50 == 49:
            torch.save(model.state_dict(), path + '_modelinfo.pt')

            with open(path + '_plotdata.txt', "w") as file:
                file.write(str(epoch+1))
                file.write(''.join([str(x)+' ' for x in train_acc]))
                file.write(''.join([str(x)+' ' for x in gen_error]))
                file.write(''.join([str(x)+' ' for x in sum_term]))

    sum_term.pop(0)

    return train_acc, gen_error, sum_term


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

    # read in arguments
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--var', type=float, default=0.0,
                        help='langevin noise variance')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='minibatch size for training and testing')
    parser.add_argument('--dataset_size', type=int, default=60000,
                        help='subset of MNIST used')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs for training')
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='name of folder to save the results')
    args = parser.parse_args()
    print(args)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    variance = args.var
    dataset_size = args.dataset_size
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
    fig1.suptitle('train accuracy (lr=' + str(lr) + ')')

    # initialize figure 2 (gen error)
    fig2 = plt.figure()
    fig2ax = fig2.add_subplot()
    fig2ax.set_xlabel('Epoch')
    fig2ax.set_ylabel('Generalization Error')
    fig2.suptitle('generalization error (lr=' + str(lr) + ')')

    # initialize figure 2 (gen error bound)
    fig3 = plt.figure()
    fig3ax = fig3.add_subplot()
    fig3ax.set_xlabel('Step')
    fig3ax.set_ylabel('Empirical Bound')
    fig3.suptitle('generalization error bound (lr=' + str(lr) + ')')

    for p in [0.00, 0.50]:
        print("Running for p = "+str(p))

        # configure datasets and dataloaders
        trainset = get_MNIST(noise=p, size=dataset_size, train=True)
        testset = get_MNIST(train=False)
        train_loader, _, test_loader = data.getDataloader(trainset, testset, validation_size, batch_size, num_workers)

        # initialize stuff for our algorithm
        model = make_mnist_alexnet().to(device)
        optimizer = NewSGLD(model.parameters(), lr=lr, variance=variance)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        path = 'experiments/' + experiment_name + '/noise' + str(p)
        try:
            model.load_state_dict(torch.load(path + '_modelinfo.pt'))
        except:
            with open(path + '_plotdata.txt', "w") as file:
                file.write('0')

        # train and plot
        train_acc, gen_error, sum_term = train_model(model, optimizer, criterion, train_loader, test_loader, epochs, path)
        fig1ax.plot(range(epochs), train_acc, label='p='+str(p))
        fig2ax.plot(range(epochs), gen_error, label='p='+str(p))
        coef = 8.12/dataset_size * lr/math.sqrt(variance)
        fig3ax.plot(range(len(sum_term)), [math.sqrt(i) * coef for i in sum_term], label='p='+str(p))

    fig1ax.legend()
    fig2ax.legend()
    fig3ax.legend()
    fig1.savefig('results/'+experiment_name+'/train_accuracy.png')
    fig2.savefig('results/'+experiment_name+'/gen_error.png')
    fig3.savefig('results/'+experiment_name+'/gen_error_bound.png')

