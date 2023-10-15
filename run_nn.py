import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from helpers import get_kwargs
import data
import models
from sgld_optimizer import SGLD


# returns a list of length 20 representing the accuracies in each bin
def acc_vs_confidence(model, data_loader):
    correct = [0 for i in range(21)]
    total = [0 for i in range(21)]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):

            softmax_scores = torch.softmax(model(inputs), dim=-1)
            confidence_vals, predictions = torch.max(softmax_scores, dim=-1)

            for j in range(len(predictions)):
                bin = int(confidence_vals[j] * 20)
                total[bin] = total[bin] + 1
                if predictions[j] == labels[j]:
                    correct[bin] = correct[bin] + 1

    return [(correct[i] / total[i] if total[i] > 0 else 0) for i in range(21)]


# returns a list of length "epochs" representing the accuracies after each epoch
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs):
    acc = []
    for epoch in range(epochs):

        # train
        for i, (inputs, labels) in enumerate(train_loader):
            loss = criterion(model(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate
        with torch.no_grad():
            correct = 0
            for i, (inputs, labels) in enumerate(val_loader):
                predictions = torch.argmax(model(inputs), dim=1)
                for j in range(len(predictions)):
                    if predictions[j] == labels[j]:
                        correct += 1
            acc.append(correct / len(valset))

    return acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # torch.backends.cudnn.benchmark = True

    # read in arguments
    parser = argparse.ArgumentParser(description='model parameters')
    parser.add_argument('--dataset', type=str, default='abalone',
                        help='dataset name', choices=['abalone', 'CIFAR10'])
    parser.add_argument('--net_type', type=str, default='MLP',
                        help='neural network name', choices=['MLP', 'AlexNet'])
    parser.add_argument('--optim', type=str, default='both',
                        help='optimizer', choices=['SGD', 'SGLD', 'both'])
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='minibatch size for training and testing')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs for training')
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    net_type = args.net_type
    optim = args.optim
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    seed = 1
    torch.manual_seed(seed)

    # model_cfg is the class that our model is an instance of
    model_cfg = getattr(models, net_type)

    # miscellaneous parameters
    validation_size = 0
    num_workers = 1

    # read in data and get dataloaders
    trainset, valset, features, outputs = data.getTransformedDataset(dataset, model_cfg, uci_regression=False)
    train_loader, _, val_loader = data.getDataloader(trainset, valset, validation_size, batch_size, num_workers)

    # initialize figure 1
    fig1 = plt.figure()
    ax = fig1.add_subplot()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    fig1.suptitle('Learning Curves (lr=' + str(lr)+')')

    # initialize figure 2
    fig2, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('SGD')
    ax2.set_title('SGLD')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Accuracy')
    fig2.suptitle('Accuracy vs Confidence')

    if optim == 'SGD' or optim == 'both':

        print("TRAINING MODEL WITH SGD")

        # initialize model
        model_args = list()
        model_kwargs = get_kwargs(dataset, model_cfg, False, features, outputs)
        model = model_cfg.base(*model_args, **model_kwargs).to(device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        # train model
        acc_ls = train_model(model, optimizer, criterion, train_loader, val_loader, epochs)
        ax.plot(range(epochs), acc_ls, label='sgd')

        # compute accuracy vs confidence at the end
        if optim == 'both':
            accuracy_ls = acc_vs_confidence(model, val_loader)
            ax1.bar([x/20 for x in range(21)], accuracy_ls)

    if optim == 'SGLD' or optim == 'both':

        print("TRAINING MODEL WITH SGLD")

        # initialize model
        model_args = list()
        model_kwargs = get_kwargs(dataset, model_cfg, False, features, outputs)
        model = model_cfg.base(*model_args, **model_kwargs).to(device)
        model.train()

        optimizer = SGLD(model.parameters(), lr=lr)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        # train model
        acc_ls = train_model(model, optimizer, criterion, train_loader, val_loader, epochs)
        ax.plot(range(epochs), acc_ls, label='sgld')

        # compute accuracy vs confidence at the end
        if optim == 'both':
            accuracy_ls = acc_vs_confidence(model, val_loader)
            ax2.bar([x/20 for x in range(21)], accuracy_ls)

    if optim == 'both':
        ax.legend()
    fig1.savefig('results/'+net_type+'_learning_curve_'+optim+'.png')

    if optim == 'both':
        fig2.savefig('results/'+net_type+'_accuracy_vs_confidence.png')

