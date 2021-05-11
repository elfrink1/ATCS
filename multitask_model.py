# This file implements the multitask framework with a pre-trained BERT model
# Hard sharing will be used as the multitask framework
import torch
import torch.nn as nn
from transformers import BertModel
import copy

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

        embedding, encoder, pooler = [*self.bert.children()]
        tl = -conf.task_layers

        self.shared = nn.Sequential(embedding, encoder[:tl])

        self.ag = get_task_layers(encoder[tl:], pooler, 4)

        self.hp = get_task_layers(encoder[tl:], pooler, 41)

        self.bbc = get_task_layers(encoder[tl:], pooler, 5)

        self.ng = get_task_layers(encoder[tl:], pooler, 6)


    # TODO I am unsure whether we should add the pooling layer, so I have commented it out for now
    def get_task_layers(self, encoder_layers, pooling_layer, num_classes):
        encoder = copy.deepcopy(encoder_layers)
        # pooler = copy.deepcopy(pooling_layer)
        task_layers = nn.Sequential(
            encoder,
            # pooler,
            nn.ReLU()
            nn.Linear(768, num_classes)
        )
        return task_layers


    def forward(self, batch, task):
        b = self.shared(batch).last_hidden_state

        if task == 'ag':
            c = self.ag(b)

        elif task == 'bbc':
            c = self.bbc(b)

        elif task == 'hp':
            c = self.hp(b)

        elif task == 'ng':
            c = self.ng(b)
        
        return c