import torch
import random
import math


__all__ = ['calc_li_summand', 'calc_banerjee_summand', 'calc_li_bound', 'calc_banerjee_bound']

# random and device, how do I define this globally neatly across all files??

li_gld_constant = math.sqrt(8)
li_sgld_constant = 8.12
banerjee_constant = math.sqrt(20)


def calc_li_summand(model, optimizer, criterion, dataset):  # doesn't matter but this should run in eval mode i think
    device = optimizer.device

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
    return total / 200, total / 200 * optimizer.std_coef


def calc_banerjee_summand(model, optimizer, criterion, trainset, testset):
    device = optimizer.device

    i1 = random.randint(0, len(trainset) - 1)
    i2 = random.randint(0, len(testset) - 1)
    d1, l1 = trainset[i1]
    d2, l2 = testset[i2]
    d1.unsqueeze_(0)
    d2.unsqueeze_(0)
    optimizer.zero_grad()
    loss = criterion(model(d1.to(device)), torch.tensor([l1], device=device))
    loss.backward()
    p1 = []
    for p in model.parameters():
        p1.append(p.grad.detach().clone())
    optimizer.zero_grad()
    loss = criterion(model(d2.to(device)), torch.tensor([l2], device=device))
    loss.backward()
    p2 = []
    for p in model.parameters():
        p2.append(p.grad.detach().clone())
    diff = [a-b for a, b in zip(p1, p2)]
    total = 0
    for x in diff:
        total += torch.sum(x ** 2).item()
    return total, total * optimizer.std_coef


def calc_li_bound(summand_ls):
    pass


def calc_banerjee_bound(summand_ls):
    pass
