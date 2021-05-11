# This file implements the multitask framework with a pre-trained BERT model
# Hard sharing will be used as the multitask framework
import torch
import torch.nn as nn
from transformers import BertModel

class MultitaskBert(nn.Module):
    def __init__(self, conf):
        super(MultitaskBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.nr_layers = 11
        self.finetuned_layers = [str(self.nr_layers - diff) for diff in range(conf.finetuned_layers)] if conf.finetuned_layers > 0 else []
        self.finetuned_layers.append("pooler")


        for n, p in self.bert.named_parameters():
            if True in [ftl in n for ftl in self.finetuned_layers]:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.ag = get_task_layers(4)

        self.hp = get_task_layers(41)

        self.bbc = get_task_layers(5)

        self.ng = get_task_layers(6)

    # TODO Make Bert layer task specific
    def get_task_layers(self, num_classes):

        if conf.task_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12) #AFAIK this is default for this version of BERT
            encoder = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, conf.task_layers),
            nn.ReLU(),
            nn.Linear(768, num_classes)
            )

        else:
            encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, num_classes)
            )

        return encoder


    def forward(self, batch, task):
        b = self.bert(batch).pooler_output

        if task == 'ag':
            c = self.ag(b)

        elif task == 'bbc':
            c = self.bbc(b)

        elif task == 'hp':
            c = self.hp(b)

        elif task == 'ng':
            c = self.ng(b)
        
        return c