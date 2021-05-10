import torch.nn as nn
import torch


def prototype(config, model, support):
    all_emb = model(support)
    c_k = torch.stack([torch.mean(all_emb[i:i + config.shot, :], dim=0) for i in range(0, config.shot * config.way, config.shot)]) # mean embedding for each class

    W = 2 * c_k
    b = -torch.pow(torch.norm(c_k, dim=1), 2)

    return W, b


def inner_loop():
    k = 0