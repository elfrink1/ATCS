import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from datasets import load_dataset

def batch_data(data, bs=2):
    ''' Returns the data in a list of batches of size bs'''
    return [data[i:i+bs] for i in range(0, len(data), bs)]

def process_hp_batch(batch, tokenizer, dev):
    headlines = tokenizer(batch["headline"], padding=True, return_tensors='pt')["input_ids"]
    headlines = headlines.to(dev)
    # descr = tokenizer(batch["short_description"])["input_ids"] ## is sometimes empty, which torch does not like
    labels = torch.LongTensor(batch["category_num"])
    labels = labels.to(dev)
    return headlines, labels

def process_ag_batch(batch, tokenizer, dev):
    # has some weird \\ characters in the text
    noslash = [ex.replace("\\", " ") for ex in batch["text"]]
    headlines = tokenizer(noslash, padding=True, return_tensors='pt')["input_ids"]
    headlines = headlines.to(dev)

    labels = torch.LongTensor(batch["label"])
    labels = labels.to(dev)

    return headlines, labels

def process_bbc_batch(batch, tokenizer, dev):
    headlines = tokenizer([b[0] for b in batch], padding=True, return_tensors='pt')["input_ids"]
    headlines = headlines.to(dev)

    labels = torch.LongTensor([b[1] for b in batch])
    labels = labels.to(dev)

    return headlines, labels

class Config(object):
    # Temporary class to store the config. Will later be replaced by a 
    # argument based config
    def __init__(self):
        self.name = "bertplusmlp"
        self.output_dim = 41
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 64
        self.lr = 0.1
        self.dataset = "bbc"
        self.max_epochs = 100
        self.finetune_layers = 1

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
        self.mlp = nn.Linear(768, conf.output_dim)

    def forward(self, batch):
        b = self.bert(batch).pooler_output
        c = self.mlp(b)
        return c

def train_bert(model, tokenizer, optimizer, train_data, dev_data, process_batch, conf):
    criterion = nn.CrossEntropyLoss()
    epoch_id = 0

    while True:

        ### train ###
        model.train()
        train_correct, train_total = 0, 0

        for batch in batch_data(train_data, conf.batch_size):
            in_, labels = process_batch(batch, tokenizer, conf.device)

            out_ = model(in_)

            loss = criterion(out_, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(out_, 1)[1]

            train_correct += sum([p == l for p, l in zip(pred, labels)]).item()
            train_total += len(labels)


        ### val ###
        model.eval()
        dev_correct, dev_total = 0, 0
        with torch.no_grad():
            for batch in batch_data(dev_data, conf.batch_size):
                in_, labels = process_batch(batch, tokenizer, conf.device)
                out_ = model(in_)
                pred = torch.max(out_, 1)[1]

                dev_correct += sum([p == l for p, l in zip(pred, labels)]).item()
                dev_total += len(labels)

        print("Epoch {}/{}, train: {:.2f}, dev: {:.2f}".format(epoch_id, 10, train_correct / train_total * 100, dev_correct / dev_total * 100))

        if epoch_id >= conf.max_epochs:
            break

        epoch_id += 1
        


if __name__ == "__main__":
    conf = Config()

    print("Training ", conf.name, "on", conf.dataset)
    print("Using device:", conf.device)

    dataset, train_data, dev_data, val_data = None, None, None, None

    if conf.dataset == "hp":
        print("Getting Fraser (HuffPost) dataset")
        # splits in train, test, validate
        dataset = load_dataset("Fraser/news-category-dataset")
        train_data = dataset["train"]
        dev_data = dataset["test"]
        val_data = dataset["validate"]
        process_batch = process_hp_batch

    elif conf.dataset == "ag":
        print("Getting AG news dataset")
        # splits in train (120000) and test (7600)
        dataset = load_dataset("ag_news")
        train_data = dataset["train"]
        val_data = dataset["test"]
        process_batch = process_ag_batch

    elif conf.dataset == "bbc":
        print("Getting BBC news dataset")
        l2i = {'entertainment':0, 'business':1, 'politics':2, 'sport':3, 'tech':4}

        # first row are colum names: ['ArticleId' 'Text' 'Category']
        train_data = [line.replace("\n", "").split(",") for line in open('Data/BBC News Train.csv')][1:]
        train_data = [[ex[1][:512], l2i[ex[2]]] for ex in train_data]
        
        # first row are colum names: ['ArticleId' 'Text']
        val_ex = [line.replace("\n", "").split(",") for line in open('Data/BBC News Test.csv')][1:]
        # first row are colum names: ['ArticleId' 'Category']
        val_labels = [line.replace("\n", "").split(",") for line in open('Data/BBC News Sample Solution.csv')][1:]
        val_data = [[ex[1][:512], l2i[label[1]]] for ex, label in zip(val_ex, val_labels) if ex[0] == label[0]]

        process_batch = process_bbc_batch

    else:
        exit("!! No dataset was given. Aborting training !!")

    model = BertClassifier(conf)
    model = model.to(conf.device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optim = torch.optim.SGD(model.parameters(), conf.lr)

    out = train_bert(model, tokenizer, optim, train_data, val_data, process_batch, conf)