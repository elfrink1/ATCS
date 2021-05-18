import torch.utils.data as data
import pytorch_lightning as pl
import os
import torch


from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer
from proto_data2 import ProtoDataset
from proto_trainer import ProtoTrainer


# load_dataset('Fraser/news-category-dataset', cache_dir='./Data')
# load_dataset("ag_news", cache_dir='./Data')
# load_dataset("dbpedia_14", cache_dir='./Data')
# load_dataset("yahoo_answers_topics", cache_dir='./Data')
# load bbc news


class config():
    def __init__(self):
        self.data_path = './Data/'
        self.cache_path = './Cache/'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # bert h_parameters
        self.finetuned_layers = 5
        self.mlp_in = 768
        self.hidden = 192

         # maml h_parameters
        self.inner_steps = 5
        self.meta_batch = 12
        self.eval_batch = 5
        self.inner_lr = 0.001 #TODO: change
        self.outer_lr = 0.01
        self.max_epochs = 10000

        # episode h_parameters
        self.way = 5
        self.shot = 5
        self.query_size = 10
        self.max_eval_size = 150
    

        # debug stuff
        self.debug = False

       



def train_protomaml(config, dataset):
    
    model = ProtoTrainer(config).to(config.device)

    eval_acc = []

    for i in range(config.max_epochs):
        batch = dataset.train.train_batch(config.meta_batch)
        model.meta_train(batch)
        if i%1 == 0:
            batch = dataset.val.test_batch(config.eval_batch)
            eval_acc.append(model.eval_model(batch))

    #test_result = trainer.test(model, loader['test'])

    return model, test_result




if __name__ == "__main__":
    config = config() 

    

    dataset = ProtoDataset(config, 'yahoo')


    #model, results = train_protomaml(config, dataset)


        




