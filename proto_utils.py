import torch.nn as nn
import torch



def create_labels(config, way, shot):
    return torch.flatten(torch.tensor([[i] * shot for i in range(way)])).to(config.device) # labels are [0 x shot, ..., way x shot]

def tokenize_set(config, split):
    tok_text = config.tokenizer(split, padding=True, truncation=True, return_tensors='pt')
    return tok_text.to(config.device)
