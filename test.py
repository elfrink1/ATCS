import torch
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_ng(tokenizer):
    text = []
    politics = ['18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc'] # Label 0
    science = ['18828_sci.crypt', '18828_sci.electronics', '18828_sci.med', '18828_sci.space'] # 1
    religion = ['18828_alt.atheism', '18828_soc.religion.christian', '18828_talk.religion.misc'] # 2
    computer = ['18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware',
    '18828_comp.sys.mac.hardware', '18828_comp.windows.x'] # 3
    sports = ['18828_rec.autos', '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey'] # 4
    sale = ['18828_misc.forsale'] # 5

    for i, category in enumerate([politics, science, religion, computer, sports, sale]):
        for subcat in category:
            data = load_dataset("newsgroup", subcat)['train']['text']
            text += [[ex.split('\n', 2)[2], i] for ex in data]

    
    train_val, test = train_test_split(text, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)
    train = process_ng(train, tokenizer)
    val = process_ng(val, tokenizer)
    test = process_ng(test, tokenizer)
    return train, val, test


def process_ng(data, tokenizer):
    """ Extracts the headlines and labels from the 20 newsgroups dataset. """
    text = []
    for b in data:
        text.append(b[0])

    print('37')
    text = tokenizer(text, padding=True, return_tensors='pt')["input_ids"]

    print('40')
    labels = []
    for b in data:
        labels.append(b[1])
    labels = torch.LongTensor(labels)

    print('45')

    return [{"txt" : h, "label" : l} for h, l in zip(text, labels)]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=512)
train, val, test = load_ng(tokenizer)
print('yay')
