import torch
import numpy as np

device  = 'cuda:0'

def l1(model):
    l1_regularisation = 0.
    for param in model.parameters():
        l1_regularisation += param.abs().sum()
    return l1_regularisation


def add_input_noise(input, std, mean=0):
    return input + torch.randn(input.size()).to(device) * std + mean

# adapted from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)