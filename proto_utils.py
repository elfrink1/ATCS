import torch.nn as nn
import torch


def prototype(config, model, support):
    all_emb = model.embed(support) #only passes through bert
    c_k = torch.stack([torch.mean(all_emb[i:i + config.shot, :], dim=0) for i in range(0, config.shot * config.way, config.shot)]) # mean embedding for each class

    W = 2 * c_k
    b = torch.sum(-c_k * c_k, dim=-1) #same as [torch.dot(c_k[i,:] c_k[i,:])]
    #print(b)
    #b = -torch.linalg.norm(c_k, dim=1)

    return W, b


