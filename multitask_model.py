# This file implements the multitask framework with a pre-trained BERT model
# Hard sharing will be used as the multitask framework
import torch
import torch.nn as nn
from torch.utils import data
from transformers import BertModel
import copy
from multitask_data import LoadMultitaskData, MergeMultitaskData

class MultitaskBert(nn.Module):
    def __init__(self, conf):
        super(MultitaskBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.nr_layers = 11
        self.finetuned_layers = [str(self.nr_layers - diff) for diff in range(conf.finetuned_layers)] if conf.finetuned_layers > 0 else []
        self.finetuned_layers.append("pooler")

        if conf.finetuned_layers != -1:
            for n, p in self.bert.named_parameters():
                if True in [ftl in n for ftl in self.finetuned_layers]:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            pass

        embedding, encoder, pooler = [*self.bert.children()]
        tl = -conf.task_layers
        self.embedding = embedding
        self.bert_layer = encoder.layer[0]
        self.pooler = pooler

        self.shared_encoders = encoder.layer[:tl]

        self.ag = self.get_task_layers(encoder.layer[tl:], pooler, 4)

        self.hp = self.get_task_layers(encoder.layer[tl:], pooler, 41)

        self.bbc = self.get_task_layers(encoder.layer[tl:], pooler, 5)

        self.ng = self.get_task_layers(encoder.layer[tl:], pooler, 6)

        self.task_layers = {'hp': self.hp, 'ag':self.ag, 'bbc':self.bbc, 'ng':self.ng}


    # TODO I am unsure whether we should add the pooling layer, so I have commented it out for now
    # Chris: I think we should, since the original BERT classifier uses it as well
    def get_task_layers(self, encoder_layers, pooling_layer, num_classes):
        encoders = copy.deepcopy(encoder_layers)
        pooler = copy.deepcopy(pooling_layer)
        mlp = nn.Linear(768, num_classes)
        task_layers = nn.ModuleList([*encoders, pooler, mlp])
        return task_layers


    def apply_task_layers(self, x, task_layers):
        encoders = task_layers[:-2]
        for encoder in encoders:
            x = encoder(x)[0]        # apply transformer layer
        x = task_layers[-2](x)       # apply pooler
        x = task_layers[-1](x)       # apply MLP classifier
        return x


    def forward(self, batch):
        datasets = list(batch.keys())
        outputs = []
        for dataset in datasets:
            out = self.embedding(batch[dataset]['txt'])
            for encoder in self.shared_encoders:
                out = encoder(out)[0]

            out = self.apply_task_layers(out, self.task_layers[dataset])
            outputs.append(out)
        return outputs

# class Args():
#     def __init__(self):
#         self.path = "models/bert"
#         self.optimizer = "Adam"
#         self.lr = 0.001
#         self.max_epochs = 100
#         self.finetuned_layers = 0
#         self.task_layers = 1
#         self.tokenizer = "BERT"
#         self.batch_size = 64
#         self.device = "gpu"
#         self.seed = 20
#         self.max_text_length = -1
#
# if __name__ == "__main__":
#     conf = Args()
#     multitask_data = LoadMultitaskData(conf)
#     train_data = MergeMultitaskData(multitask_data.train)
#     loader = data.DataLoader(train_data, batch_size = conf.batch_size)
#     batch = next(iter(loader))
#     model = MultitaskBert(conf)
#     output = model(batch)
