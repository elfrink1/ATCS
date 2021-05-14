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
        # print(embedding)
        # print(encoder.layer)
        # print(pooler)
        # jfirjfr

        self.shared = nn.Sequential(embedding, encoder.layer[:tl])

        self.ag = self.get_task_layers(encoder.layer[tl:], pooler, 4)

        self.hp = self.get_task_layers(encoder.layer[tl:], pooler, 41)

        self.bbc = self.get_task_layers(encoder.layer[tl:], pooler, 5)

        self.ng = self.get_task_layers(encoder.layer[tl:], pooler, 6)


    # TODO I am unsure whether we should add the pooling layer, so I have commented it out for now
    # Chris: I think we should, since the original BERT classifier uses it as well
    def get_task_layers(self, encoder_layers, pooling_layer, num_classes):
        encoder = copy.deepcopy(encoder_layers)
        pooler = copy.deepcopy(pooling_layer)
        task_layers = nn.Sequential(
            encoder,
            pooler,
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )
        return task_layers


    def forward(self, batch):
        print(batch['hp']['txt'])
        out_hp = self.shared(batch['hp']['txt']).last_hidden_state
        out_ag = self.shared(batch['ag']['txt']).last_hidden_state
        out_bbc = self.shared(batch['bbc']['txt']).last_hidden_state
        out_ng = self.shared(batch['ng']['txt']).last_hidden_state

        out_hp = self.hp(out_hp)
        out_ag = self.ag(out_ag)
        out_bbc = self.bbc(out_bbc)
        out_ng = self.ng(out_ng)

        return (out_hp, out_ag, out_bbc, out_ng)

class Args():
    def __init__(self):
        self.path = "models/bert"
        self.optimizer = "Adam"
        self.lr = 0.001
        self.max_epochs = 100
        self.finetuned_layers = 0
        self.task_layers = 1
        self.tokenizer = "BERT"
        self.batch_size = 64
        self.device = "gpu"
        self.seed = 20
        self.max_text_length = -1
        self.save = False
        self.load = False

if __name__ == "__main__":
    conf = Args()
    model = MultitaskBert(conf)
