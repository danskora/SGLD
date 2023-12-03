import torch
import random
import math


__all__ = ['calc_li_summand', 'calc_banerjee_summand', 'calc_li_bound', 'calc_banerjee_bound']

# random and device, how do I define this globally neatly across all files??

li_gld_constant = 2
li_sgld_constant = 8.12 / math.sqrt(2)
banerjee_sgld_constant = math.sqrt(20)


# BUG: std_coef may be 0 for SGD
# BUG: we are still overestimating banerjee_summand

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
    return total / 200, total / 200 / (optimizer.std_coef ** 2)


def calc_banerjee_summand(model, optimizer, criterion, z_ls, z_prime_ls):
    device = optimizer.device

    res = []
    for d1, l1 in z_ls:
        for d2, l2 in z_prime_ls:
            d1.unsqueeze_(0)  # does this cause issues by adding a dimension each time this method is called
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
            res.append(total)

    return res, [i / (optimizer.std_coef ** 2) for i in res]


def calc_li_bound(summand_ls, n, stochastic=True):
    summation_ls = [0]
    for e in summand_ls:
        summation_ls.append(summation_ls[-1] + e)
    summation_ls.pop(0)

    C = li_sgld_constant if stochastic else li_gld_constant
    return [C * math.sqrt(i) / n for i in summation_ls]


def calc_banerjee_bound(summand_tensor, n, stochastic=True):
    summation_tensor = torch.cumsum(summand_tensor, dim=0)
    expectation_S = torch.mean(torch.sqrt(summation_tensor), dim=1)

    if not stochastic:
        raise Exception('Banerjee bound requires stochastic gradient!')
    C = banerjee_sgld_constant
    return [C * i / n for i in expectation_S]
