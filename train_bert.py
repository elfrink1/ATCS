import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data import Dataset
import torch.utils.data as data
from berttrainer import BERTTraniner
from transformers import BertTokenizer

class Config(object):
    # Temporary class to store the config. Will later be replaced by a 
    # argument based config
    def __init__(self):
        self.name = "bertplusmlp"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = 64
        self.lr = 0.1
        self.dataset = "bbc"
        self.max_epochs = 100
        self.finetune_layers = 1
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.seed = 20
        self.optimizer = "SGD"
        self.CHECKPOINT_PATH = "models"
        self.nr_classes = 41

def train_bert(loader, conf):
    trainer = pl.Trainer(default_root_dir=os.path.join(conf.CHECKPOINT_PATH, conf.name),
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max"),
                         gpus=1 if "cuda" in str(conf.device) else 0,
                         max_epochs=conf.max_epochs,                                            
                         progress_bar_refresh_rate=1) 
    
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pl.seed_everything(conf.seed)
    model = BERTTraniner(conf.name, model_hparams={"num_classes": 10}, optimizer_name="Adam",
                         optimizer_hparams={"lr": 1e-3}, conf=conf)

    trainer.fit(model, loader['train'])

        
if __name__ == "__main__":
    conf = Config()

    print("Training ", conf.name, "on", conf.dataset)
    print("Using device:", conf.device)

    dataset = Dataset(conf)
    loader = {
        'train' : data.DataLoader(dataset.train, batch_size=conf.batch_size, shuffle=False, pin_memory=True, num_workers=4) if dataset.train != None else None,
        'val'   : data.DataLoader(dataset.val, batch_size=conf.batch_size, shuffle=False, drop_last=False, num_workers=4) if dataset.val != None else None,
        'test'  : data.DataLoader(dataset.test, batch_size=conf.batch_size, shuffle=False, drop_last=False, num_workers=4) if dataset.test != None else None
    }

    out = train_bert(loader, conf)