import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.data as data

from data import Dataset
from berttrainer import BERTTraniner
from bertutil import get_args

def train_bert(loader, conf):
    trainer = pl.Trainer(default_root_dir=os.path.join(conf.path, conf.name),
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max"),
                         gpus=1 if "gpu" in str(conf.device) else 0,
                         max_epochs=conf.max_epochs,                                            
                         progress_bar_refresh_rate=1) 
    
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pl.seed_everything(conf.seed)
    model = BERTTraniner(conf.name, model_hparams={"num_classes": 10}, optimizer_name="Adam",
                         optimizer_hparams={"lr": 1e-3}, conf=conf)

    trainer.fit(model, loader['train'])

        
if __name__ == "__main__":
    conf = get_args()

    print("Training ", conf.name, "on", conf.dataset)

    dataset = Dataset(conf)
    loader = {
        'train' : data.DataLoader(dataset.train, batch_size=conf.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4) if dataset.train != None else None,
        'val'   : data.DataLoader(dataset.val, batch_size=conf.batch_size, shuffle=False, drop_last=False, num_workers=4) if dataset.val != None else None,
        'test'  : data.DataLoader(dataset.test, batch_size=conf.batch_size, shuffle=False, drop_last=False, num_workers=4) if dataset.test != None else None
    }

    out = train_bert(loader, conf)