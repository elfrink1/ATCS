import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, conf):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.nr_layers = 11
        self.finetune_layers = [str(self.nr_layers - diff) for diff in range(conf.finetune_layers)] if conf.finetune_layers > 0 else []
        self.finetune_layers.append("pooler")

        for n, p in self.bert.named_parameters():
            if True in [ftl in n for ftl in self.finetune_layers]:
                p.requires_grad = True
            else: 
                p.requires_grad = False
        self.mlp = nn.Linear(768, conf.nr_classes)

    def forward(self, batch):
        b = self.bert(batch).pooler_output
        c = self.mlp(b)
        return c