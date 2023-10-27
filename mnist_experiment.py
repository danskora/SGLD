import argparse
import torch
import data
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import transforms
# from sgld_optimizer import NewSGLD
from torch.utils.data import Subset


class NoisyMNIST(torchvision.datasets.MNIST):
    def __init__(self, p, train):
        super(NoisyMNIST, self).__init__(root='./data', train=train, download=True,
                                         transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

        self.newlabels = {}
        if train:
            indices = torch.randperm(60000)[:int(p*60000)].tolist()
            for idx in indices:
                self.newlabels[idx] = int(random.random()*10)

    def __getitem__(self, idx):
        img, label = super(NoisyMNIST, self).__getitem__(idx)
        return img, self.newlabels.get(idx, label)


def train_model(model, optimizer, criterion, train_loader, epochs):
    train_acc = []
    train_loss = []

    for epoch in range(epochs):
        print(epoch)

        # train
        model.train()
        total_loss = 0
        correct = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(outputs, dim=1)
            for j in range(len(predictions)):
                if predictions[j] == labels[j]:
                    correct += 1
        train_loss.append(total_loss / len(train_loader.dataset))
        train_acc.append(correct / len(train_loader.dataset))

    return train_acc, train_loss


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
    parser.add_argument('--optim', type=str, default='both',
                        help='optimizer', choices=['SGD', 'SGLD', 'both'])
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='minibatch size for training and testing')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs for training')
    args = parser.parse_args()
    print(args)

    optim = args.optim
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    validation_size = 0
    num_workers = 1

    # initialize figure 1 (train acc)
    fig1 = plt.figure()
    fig1ax = fig1.add_subplot()
    fig1ax.set_xlabel('Epoch')
    fig1ax.set_ylabel('Accuracy')
    fig1.suptitle('train accuracy (lr=' + str(lr) + ')')

    for p in [0.00, 0.50]:

        # configure datasets and dataloaders
        trainset = NoisyMNIST(p, train=True)
        testset = NoisyMNIST(p, train=False)
        #trainset = Subset(trainset, range(5000))
        train_loader, _, test_loader = data.getDataloader(trainset, testset, validation_size, batch_size, num_workers)

        # initialize stuff for our algorithm
        model = make_mnist_alexnet().to(device)
        #optimizer = NewSGLD(model.parameters(), lr=lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        # train and plot
        train_acc, train_loss = train_model(model, optimizer, criterion, train_loader, epochs)
        fig1ax.plot(range(epochs), train_acc, label='p='+str(p))

    fig1ax.legend()
    fig1.savefig('results/mnist_experiment_train_accuracy.png')

