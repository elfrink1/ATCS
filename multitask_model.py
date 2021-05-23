
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
        self.config = conf
        self.nr_layers = 11
        self.num_classes = {'hp': 41, 'ag' : 4, 'bbc': 5, 'ng' : 6, 'dbpedia' : 14}
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
        self.pooler = pooler

        self.shared_encoders = encoder.layer[:tl]

        self.task_layers = {}
        for dataset in conf.train_sets:
            self.task_layers[dataset] = self.get_task_layers(encoder.layer[tl:], pooler, self.num_classes[dataset])

        self.few_shot_head = nn.Sequential(nn.Linear(786, conf.hidden), nn.ReLU())

        # self.ag = self.get_task_layers(encoder.layer[tl:], pooler, 4)
        #
        # self.hp = self.get_task_layers(encoder.layer[tl:], pooler, 41)
        #
        # self.bbc = self.get_task_layers(encoder.layer[tl:], pooler, 5)
        #
        # self.task_layers = {'hp': self.hp, 'ag':self.ag, 'bbc':self.bbc}


    def get_task_layers(self, encoder_layers, pooling_layer, num_classes):
        encoders = copy.deepcopy(encoder_layers)
        pooler = copy.deepcopy(pooling_layer)
        mlp = nn.Linear(768, num_classes)
        task_layers = nn.ModuleList([*encoders, pooler, mlp])
        return task_layers


    # Applies the head created from the function 'get_task_layers'
    # These are either the task specific layers or the few shot evaluation head
    def apply_task_layers(self, x, task_layers):
        encoders = task_layers[:-2]
        for encoder in encoders:
            x = encoder(x)[0]        # apply transformer layer
        x = task_layers[-2](x)       # apply pooler
        x = task_layers[-1](x)       # apply MLP classifier
        return x


    def apply_shared_encoders(self, x):
        for encoder in self.shared_encoders:
            x = encoder(x)[0]
        return x


    def forward(self, batch):
        datasets = self.config.train_sets
        outputs = []
        for dataset in datasets:
            out = self.embedding(batch[dataset]['txt'])
            out = self.apply_shared_encoders(out)
            out = self.apply_task_layers(out, self.task_layers[dataset])
            outputs.append(out)
        return outputs


class Args():
    def __init__(self):
        self.path = "models/bert"
        self.optimizer = "Adam"
        self.lr = 0.001
        self.max_epochs = 100
        self.finetuned_layers = 0
        self.task_layers = 1
        self.tokenizer = "BERT"
        self.batch_size = 25
        self.device = "gpu"
        self.seed = 20
        self.max_text_length = -1
        self.sample = 100
        self.train_sets = ['hp', 'ag', 'dbpedia']
        self.hidden = 192

if __name__ == "__main__":
    conf = Args()
    multitask_data = LoadMultitaskData(conf)
    train_data = MergeMultitaskData(multitask_data.train)
    loader = data.DataLoader(train_data, batch_size=conf.batch_size)
    batch = next(iter(loader))
    model = MultitaskBert(conf)
    output = model(batch)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
