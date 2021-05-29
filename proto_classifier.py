import torch
import torch.nn as nn
from transformers import BertModel
torch.manual_seed(42)

class BertClassifier(nn.Module):
    def __init__(self, conf):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.nr_layers = 11
        self.finetuned_layers = [str(self.nr_layers - diff) for diff in range(conf.finetuned_layers)] if conf.finetuned_layers > 0 else []
        
        if self.finetuned_layers != []:
            self.finetuned_layers.append("pooler") 
        
        if conf.finetuned_layers != -1:
            for n, p in self.bert.named_parameters():
                if self.finetuned_layers and True in [ftl in n for ftl in self.finetuned_layers]:
                    p.requires_grad = True
                else: 
                    p.requires_grad = False

        self.mlp = nn.Sequential(nn.Linear(768, conf.hidden), nn.ReLU())



    def forward(self, batch):
        b = self.bert(batch['input_ids'], batch['attention_mask']).pooler_output
        c = self.mlp(b)
        
        return c

