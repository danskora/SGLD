import numpy as np
import matplotlib.pyplot as plt

from bounds import *


__all__ = ['total_correct', 'plot_acc', 'plot_everything', 'plot_bounds', 'plot_grad']


def total_correct(outputs, targets):
    return np.sum(outputs.cpu().detach().numpy().argmax(axis=1) == targets.data.cpu().detach().numpy())


def plot_acc(path, acc_dict):

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


def plot_everything(path, p, dataset_size, train_acc, test_acc, li_summand=None, banerjee_summand=None):

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_xlabel('Step') if li_summand or banerjee_summand else ax.set_xlabel('Epoch')
    # fig.suptitle('noise = ' + str(p))
    #
    # epochs = len(train_acc)
    # if li_summand is not None:
    #     batches = len(li_summand) // epochs
    # elif banerjee_summand is not None:
    #     batches = len(banerjee_summand) // epochs
    # else:
    #     batches = 1
    #
    # ax.plot(range(batches, (epochs+1) * batches, batches), train_acc,
    #         label='train accuracy')
    # ax.plot(range(batches, (epochs+1) * batches, batches), [a-b for a, b in zip(train_acc, test_acc)],
    #         label=r'$err_{gen}(S)$')
    # if li_summand is not None:
    #     ax.plot(range(1, len(li_summand)+1, 1), calc_li_bound(li_summand, dataset_size, batches != 1),
    #             label=r'li $err_{gen}$ bound')
    # if banerjee_summand is not None:
    #     ax.plot(range(1, len(banerjee_summand)+1, 1), calc_banerjee_bound(banerjee_summand, dataset_size, batches != 1),
    #             label=r'banerjee $err_{gen}$ bound')
    #
    # ax.legend()
    #
    # fig.savefig(path + '/noise' + str(p) + '.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Step') if li_summand or banerjee_summand else ax.set_xlabel('Epoch')
    fig.suptitle('noise = ' + str(p))

    epochs = len(train_acc)
    if li_summand is not None:
        batches = len(li_summand) // epochs
    elif banerjee_summand is not None:
        batches = len(banerjee_summand) // epochs
    else:
        batches = 1

    train_err = [1-i for i in train_acc]
    test_err = [1-i for i in test_acc]
    stretched_train_err = sum([[i]*batches for i in train_err], [])

    ax.plot(range(batches, (epochs + 1) * batches, batches), train_err,
            label='train error')
    ax.plot(range(batches, (epochs + 1) * batches, batches), test_err,
            label='test error')
    if li_summand is not None:
        ax.plot(range(1, len(li_summand) + 1, 1),
                [a+b for a,b in zip(stretched_train_err, calc_li_bound(li_summand, dataset_size, batches != 1))],
                label=r'train error + li $err_{gen}$ bound')
    if banerjee_summand is not None:
        ax.plot(range(1, len(banerjee_summand) + 1, 1),
                [a+b for a,b in zip(stretched_train_err, calc_banerjee_bound(banerjee_summand, dataset_size, batches != 1))],
                label=r'train error + banerjee $err_{gen}$ bound')

    ax.legend()

    fig.savefig(path + '/noise' + str(p) + '.png')


def plot_bounds(path, bound_dict):

    # initialize figure 1 (li bound)
    fig1 = plt.figure()
    fig1ax = fig1.add_subplot()
    fig1ax.set_xlabel('Step')
    fig1ax.set_ylabel('Bound')
    fig1.suptitle('Li Generalization Error Bound')

    # initialize figure 2 (banerjee bound)
    fig2 = plt.figure()
    fig2ax = fig2.add_subplot()
    fig2ax.set_xlabel('Step')
    fig2ax.set_ylabel('Bound')
    fig2.suptitle('Banerjee Generalization Error Bound')

    # initialize figure 3 (both bounds)
    fig3 = plt.figure()
    fig3ax = fig3.add_subplot()
    fig3ax.set_xlabel('Step')
    fig3ax.set_ylabel('Bound')
    fig3.suptitle('Comparing Generalization Error Bounds')

    for p, (li_bound, banerjee_bound) in bound_dict.items():
        x = range(1, len(li_bound)+1, 1)
        l = 'p=' + str(p)

        fig1ax.plot(x, li_bound, label=l)
        fig2ax.plot(x, banerjee_bound, label=l)
        fig3ax.plot(x, li_bound, label='li, '+l)
        fig3ax.plot(x, banerjee_bound, label='banerjee, '+l)

    fig1ax.legend()
    fig2ax.legend()
    fig3ax.legend()

    fig1.savefig(path + '/li_bound.png')
    fig2.savefig(path + '/banerjee_bound.png')
    fig3.savefig(path + '/both_bounds.png')


def plot_grad(path, grad_dict):

    # initialize figure 1 (squared gradient norm)
    fig1 = plt.figure()
    fig1ax = fig1.add_subplot()
    fig1ax.set_xlabel('Step')
    fig1.suptitle('Squared Gradient Norm')

    # initialize figure 2 (gradient discrepancy)
    fig2 = plt.figure()
    fig2ax = fig2.add_subplot()
    fig2ax.set_xlabel('Step')
    fig2.suptitle('Gradient Discrepancy')

    # initialize figure 3 (both)
    fig3 = plt.figure()
    fig3ax = fig3.add_subplot()
    fig3ax.set_xlabel('Step')
    fig3.suptitle('Comparing Gradient Metrics')

    for p, (sq_grad_norm, grad_disc) in grad_dict.items():
        x = range(1, len(sq_grad_norm)+1, 1)
        l = 'p=' + str(p)

        fig1ax.plot(x, sq_grad_norm, label=l)
        fig2ax.plot(x, grad_disc, label=l)
        fig3ax.plot(x, sq_grad_norm, label='squared gradient norm, '+l)
        fig3ax.plot(x, grad_disc, label='gradient discrepancy, '+l)

    fig1ax.legend()
    fig2ax.legend()
    fig3ax.legend()

    fig1.savefig(path + '/squared_gradient_norm.png')
    fig2.savefig(path + '/gradient_discrepancy.png')
    fig3.savefig(path + '/both_gradient_metrics.png')

