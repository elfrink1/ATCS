import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, conf):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.nr_layers = 11
        self.finetuned_layers = [str(self.nr_layers - diff) for diff in range(conf.finetuned_layers)] if conf.finetuned_layers > 0 else []
        if self.finetuned_layers != []:
            self.finetuned_layers.append("pooler")
        self.train_param = 0

        if conf.finetuned_layers != -1:
            for n, p in self.bert.named_parameters():
                if self.finetuned_layers and True in [ftl in n for ftl in self.finetuned_layers]:
                    p.requires_grad = True
                    self.train_param += 1
                else: 
                    p.requires_grad = False
        else: 
            pass # We are finetuning the entire BERT model

        self.mlp = nn.Linear(768, conf.way)
        torch.nn.init.constant_(self.mlp.weight, 0)


    def forward(self, batch):
        b = self.bert(batch['input_ids'].squeeze(), batch['attention_mask'].squeeze()).pooler_output
        c = self.mlp(b)
        return c

    def embed(self, batch):
        b = self.bert(batch['input_ids'].squeeze(), batch['attention_mask'].squeeze()).pooler_output
        return b