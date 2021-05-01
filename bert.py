import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
from datasets import load_dataset

def batch_data(data, bs=2):
    ''' Returns the data in a list of batches of size bs'''
    return [data[i:i+bs] for i in range(0, len(data), bs)]

def process_batch(batch, tokenizer, dev):
    headlines = tokenizer(batch["headline"], padding=True, return_tensors='pt')["input_ids"]
    headlines = headlines.to(dev)
    # descr = tokenizer(batch["short_description"])["input_ids"]
    labels = torch.LongTensor(batch["category_num"])
    labels = labels.to(dev)
    return headlines, labels

class Config(object):
    # Temporary class to store the config. Will later be replaced by a 
    # argument based config
    def __init__(self):
        self.name = "bertplusmlp"
        self.output_dim = 41
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 128

class BertClassifier(nn.Module):
    def __init__(self, conf):
        super(BertClassifier, self).__init__()

        # TODO add some way to configure bert
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.mlp = nn.Linear(768, conf.output_dim)

    def forward(self, batch):
        b = self.bert(batch).pooler_output
        c = self.mlp(b)
        return c

def train_bert(model, tokenizer, optimizer, train_data, dev_data, conf):
    criterion = nn.CrossEntropyLoss()
    epoch_id = 0

    while True:
        train_correct = 0
        train_total = 0
        for batch_id, batch in enumerate(batch_data(train_data, conf.batch_size)):
            in_, labels = process_batch(batch, tokenizer, conf.device)

            out_ = model(in_)

            loss = criterion(out_, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(out_, 1)[1]

            train_correct += sum([p == l for p, l in zip(pred, labels)]).item()
            train_total += len(labels)

        print("Epoch {}/{}, acc: {:.2f}".format(epoch_id, 10, train_correct / train_total * 100))

        if epoch_id > 10:
            break

        epoch_id += 1
        


if __name__ == "__main__":
    conf = Config()

    print("Training ", conf.name, "on", conf.device)

    print("Getting Fraser (HuffPost) data")

    # splits in train, test, validate
    dataset = load_dataset("Fraser/news-category-dataset")
    train_data = dataset["train"]
    dev_data = dataset["test"]

    model = BertClassifier(conf)
    model = model.to(conf.device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optim = torch.optim.SGD(model.parameters(), lr=0.2)

    out = train_bert(model, tokenizer, optim, train_data, dev_data, conf)





    







# model = BertClassifier()

#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)