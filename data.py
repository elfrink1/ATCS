import torch
from transformers import BertTokenizer
from datasets import load_dataset

tokenizers = {
    "BERT" : BertTokenizer.from_pretrained('bert-base-uncased')
}

class Dataset():
    def __init__(self, conf):
        if conf.dataset == "hp":
            self.load_hp(conf)
        elif conf.dataset == "ag":
            self.load_ag(conf)
        elif conf.dataset == "bbc":
            self.load_bcc(conf)
        else:
            pass

    def load_hp(self, conf):
        """ Loads the HuffPost (hp) dataset from [Hugging Face](https://huggingface.co/datasets/Fraser/news-category-dataset)
            Splits in train (160682), validation (10043) and test (30128). Every sample has the following features:
            `['authors', 'category', 'category_num', 'date', 'headline', 'link', 'short_description']` """
        dataset = load_dataset("Fraser/news-category-dataset")
        self.train = self.process_hp(dataset["train"], tokenizers[conf.tokenizer])
        self.val = self.process_hp(dataset["validation"], tokenizers[conf.tokenizer])
        self.test = self.process_hp(dataset["test"], tokenizers[conf.tokenizer])

    def process_hp(self, data, tokenizer):
        """ Extracts the headlines and labels from the HuffPost dataset.
            The description is empty for some samples, which is why it is not returned."""
        headlines = tokenizer(data["headline"], padding=True, return_tensors='pt')["input_ids"]
        # descr = tokenizer(batch["short_description"])["input_ids"] ## is sometimes empty, which torch does not like
        labels = torch.LongTensor(data["category_num"])
        return [{"txt" : h, "label" : l} for h, l in zip(headlines, labels)]

    def load_ag(self, conf):
        """ Loads the AG news dataset from [Hugging Face](https://huggingface.co/datasets/ag_news)
            Splits in train (120000) and test (7600). Every sample has the following features: `['label', 'text']` """
        dataset = load_dataset("ag_news")
        self.train = self.process_ag(dataset["train"], tokenizers[conf.tokenizer])
        self.val = None
        self.test = self.process_ag(dataset["test"], tokenizers[conf.tokenizer])

    def process_ag(self, data, tokenizer):
        """ Extracts the headlines and labels from the AG news dataset.
            The headlines contain '\\' characters in place of newlines. """
        noslash = [ex.replace("\\", " ") for ex in data["text"]]
        headlines = tokenizer(noslash, padding=True, return_tensors='pt')["input_ids"]
        labels = torch.LongTensor(data["label"])
        return [{"txt" : h, "label" : l} for h, l in zip(headlines, labels)]

    def load_bcc(self, conf):
        """ Loads the BBC news dataset from [Kaggle](https://www.kaggle.com/c/learn-ai-bbc)
            This dataset has to be downloaded manually using the `downloadbbcdata` file.
            Splits in train (1490) and test (735)."""
        l2i = {'entertainment':0, 'business':1, 'politics':2, 'sport':3, 'tech':4}
        # first row are colum names: ['ArticleId' 'Text' 'Category']
        train_data = [line.replace("\n", "").split(",") for line in open('Data/BBC News Train.csv')][1:]
        self.train = self.process_bbc([[ex[1][:512], l2i[ex[2]]] for ex in train_data], tokenizers[conf.tokenizer])
        self.val = None

        # first row are colum names: ['ArticleId' 'Text']
        test_ex = [line.replace("\n", "").split(",") for line in open('Data/BBC News Test.csv')][1:]
        # first row are colum names: ['ArticleId' 'Category']
        test_labels = [line.replace("\n", "").split(",") for line in open('Data/BBC News Sample Solution.csv')][1:]
        self.test = self.process_bbc([[ex[1][:512], l2i[label[1]]] for ex, label in zip(test_ex, test_labels) if ex[0] == label[0]], tokenizers[conf.tokenizer])

        assert len(self.test) == len(test_labels)
        "The BBC test datafiles were corrupted!"

    def process_bbc(self, data, tokenizer):
        """ Extracts the headlines and labels from the BBC news dataset. """
        headlines = tokenizer([b[0] for b in data], padding=True, return_tensors='pt')["input_ids"]

        labels = torch.LongTensor([b[1] for b in data])

        return [{"txt" : h, "label" : l} for h, l in zip(headlines, labels)]

    @staticmethod
    def batch_data(data, bs=2):
        ''' Returns the data in a list of batches of size bs'''
        return [[data[0][i:i+bs], data[1][i:i+bs]] for i in range(0, len(data), bs)]
