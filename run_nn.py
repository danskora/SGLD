import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from helpers import get_kwargs
import data
import models
from sgld_optimizer import SGLD
import os


# returns a list of length 20 representing the accuracies in each bin
def acc_vs_confidence(model, data_loader):
    correct = [0 for i in range(21)]
    total = [0 for i in range(21)]

    model.eval()
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


# returns a list of length "epochs" representing the validation accuracies after each epoch
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs):
    acc = []
    for epoch in range(epochs):

        # train
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        with torch.no_grad():
            correct = 0
            for i, (inputs, labels) in enumerate(val_loader):
                predictions = torch.argmax(model(inputs), dim=1)
                for j in range(len(predictions)):
                    if predictions[j] == labels[j]:
                        correct += 1
            acc.append(correct / len(testset))

    return acc


if __name__ == '__main__':

    # make necessary subdirectories
    if not os.path.exists('model_info'):
        os.mkdir('model_info')
    if not os.path.exists('results'):
        os.mkdir('results')

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # torch.backends.cudnn.benchmark = True

    # read in arguments
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--dataset', type=str, default='abalone',
                        help='dataset name', choices=['abalone', 'CIFAR10'])
    parser.add_argument('--net_type', type=str, default='MLP',
                        help='neural network name', choices=['MLP', 'AlexNet'])
    parser.add_argument('--optim', type=str, default='both',
                        help='optimizer', choices=['SGD', 'SGLD', 'both'])
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='minibatch size for training and testing')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs for training')
    parser.add_argument('--key', type=str, default='defaultkey',
                        help='key to store model info for this experiment')
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    net_type = args.net_type
    optim = args.optim
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    key = args.key

    seed = 1
    torch.manual_seed(seed)

    # model_cfg is the class that our model is an instance of
    model_cfg = getattr(models, net_type)

    # miscellaneous parameters
    validation_size = 0
    num_workers = 1

    # read in data and get dataloaders
    trainset, testset, features, outputs = data.getTransformedDataset(dataset, model_cfg, uci_regression=False)
    train_loader, _, test_loader = data.getDataloader(trainset, testset, validation_size, batch_size, num_workers)

    # initialize figure 1 (validation accuracy)
    fig1 = plt.figure()
    ax = fig1.add_subplot()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    fig1.suptitle('Accuracy Curves (lr=' + str(lr) + ')')

    # initialize figure 2 (training loss)

    # initialize figure 3 (accuracy vs confidence)
    fig3, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('SGD')
    ax2.set_title('SGLD')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Confidence')
    fig3.suptitle('Accuracy vs Confidence')

    if optim == 'SGD' or optim == 'both':

        # initialize model
        model_args = list()
        model_kwargs = get_kwargs(dataset, model_cfg, False, features, outputs)
        model = model_cfg.base(*model_args, **model_kwargs).to(device)

        try:
            model.load_state_dict(torch.load('model_info/' + net_type + key + '_SGD.pt'))

            print("RECOVERED MODEL PREVIOUSLY TRAINED WITH SGD")

        except:
            print("TRAINING MODEL WITH SGD")

            # initialize training helpers
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            scheduler = None
            criterion = nn.CrossEntropyLoss()

            # train model
            acc_ls = train_model(model, optimizer, criterion, train_loader, test_loader, epochs)
            torch.save(model.state_dict(), 'model_info/' + net_type + key + '_SGD.pt')
            ax.plot(range(epochs), acc_ls, label='sgd')

        # compute accuracy vs confidence at the end
        print("TESTING SGD MODEL")
        accuracy_ls = acc_vs_confidence(model, test_loader)
        ax1.bar([x/20 for x in range(21)], accuracy_ls, width=0.05, align='edge')

    if optim == 'SGLD' or optim == 'both':

        # initialize model
        model_args = list()
        model_kwargs = get_kwargs(dataset, model_cfg, False, features, outputs)
        model = model_cfg.base(*model_args, **model_kwargs).to(device)

        try:
            model.load_state_dict(torch.load('model_info/' + net_type + key + '_SGLD.pt'))

            print("RECOVERED MODEL PREVIOUSLY TRAINED WITH SGLD")

        except:
            print("TRAINING MODEL WITH SGLD")

            # initialize training helpers
            optimizer = SGLD(model.parameters(), lr=lr)
            scheduler = None
            criterion = nn.CrossEntropyLoss()

            # train model
            acc_ls = train_model(model, optimizer, criterion, train_loader, test_loader, epochs)
            torch.save(model.state_dict(), 'model_info/' + net_type + key + '_SGLD.pt')
            ax.plot(range(epochs), acc_ls, label='sgld')

        # compute accuracy vs confidence at the end
        print("TESTING SGLD MODEL")
        accuracy_ls = acc_vs_confidence(model, test_loader)
        ax2.bar([x/20 for x in range(21)], accuracy_ls, width=0.05, align='edge')

    ax.legend()
    fig1.savefig('results/' + net_type + '_accuracy_curves.png')

    fig3.savefig('results/' + net_type + '_accuracy_vs_confidence.png')

