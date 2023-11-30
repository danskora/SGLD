import numpy as np


def make_plots(path, train_acc, test_acc, li_summand=None, banerjee_summand=None, squared_grad=None, squared_grad_diff=None):
    pass


def total_correct(outputs, targets):
    return np.sum(outputs.cpu().detach().numpy().argmax(axis=1) == targets.data.cpu().detach().numpy())