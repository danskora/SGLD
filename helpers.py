import numpy as np
import matplotlib.pyplot as plt

from bounds import *


__all__ = ['total_correct', 'plot_acc', 'plot_bounds', 'plot_grad']


def total_correct(outputs, targets):
    return np.sum(outputs.cpu().detach().numpy().argmax(axis=1) == targets.data.cpu().detach().numpy())


def plot_acc(path, acc_dict):  # takes dictionary because each of the three plots will display multiple noise coefficients, works with epochs on x-axis

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
    fig3.suptitle('Generalization Error\n' + r'$\mathcal{L}(\hat{w}, \mathcal{D}) - \mathcal{L}(\hat{w}, S)$')

    for p, (train_acc, test_acc) in acc_dict.items():
        x = range(1, len(train_acc)+1, 1)
        l = 'p='+str(p)

        fig1ax.plot(x, train_acc, label=l)
        fig2ax.plot(x, test_acc, label=l)
        fig3ax.plot(x, [a-b for a, b in zip(train_acc, test_acc)], label=l)

    fig1ax.legend()
    fig2ax.legend()
    fig3ax.legend()

    fig1.savefig(path + '/train_accuracy.png')
    fig2.savefig(path + '/test_accuracy.png')
    fig3.savefig(path + '/gen_error.png')


def plot_bounds(path, p, dataset_size, train_acc, test_acc, li_summand=None, banerjee_summand=None):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Step') if li_summand or banerjee_summand else ax.set_xlabel('Epoch')
    fig.suptitle('noise = ' + str(p))

    epochs = len(train_acc)
    if li_summand:
        batches = len(li_summand)
    elif banerjee_summand:
        batches = len(banerjee_summand)
    else:
        batches = 1

    ax.plot(range(batches, (epochs+1) * batches, batches), train_acc,
            label='train accuracy')
    ax.plot(range(batches, (epochs+1) * batches, batches), [a-b for a, b in zip(train_acc, test_acc)],
            label=r'$err_{gen}(S)$')
    if li_summand:
        ax.plot(range(len(li_summand)), calc_li_bound(li_summand, dataset_size, batches != 1),
                label=r'li $err_{gen}$ bound')
    if banerjee_summand:
        ax.plot(range(len(banerjee_summand)), calc_banerjee_bound(banerjee_summand, dataset_size, batches != 1),
                label=r'banerjee $err_{gen}$ bound')

    ax.legend()

    fig.savefig(path + '/noise' + str(p) + '.png')


def plot_grad(path, squared_grad=None, squared_grad_diff=None):
    pass

