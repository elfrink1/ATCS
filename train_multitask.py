import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data

from multitask_data import LoadMultitaskData, MergeMultitaskData
from multitask_model import MultitaskBert
from multitask_trainer import MultitaskTrainer

from config import get_args

def train_multitask(loader, conf):
    """ Train the BERT model in a multitask framework with all datasets.
        The model that performs best on the validation set is saved."""
    os.makedirs
    trainer = pl.Trainer(default_root_dir=os.path.join(conf.path, conf.optimizer, conf.name),
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                         gpus=1 if "gpu" in str(conf.device) else 0,
                         max_epochs=conf.max_epochs,
                         progress_bar_refresh_rate=1 if conf.progress_bar else 0)

    # Not really clear what these do, but are often disabled.
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    pl.seed_everything(conf.seed)
    model = MultitaskTrainer(conf.name, model_hparams={},
                             optimizer_name=conf.optimizer,
                             optimizer_hparams={"lr" : conf.lr},
                             conf=conf)

    trainer.fit(model, loader['train'], loader['val'])
    test_result = trainer.test(model, loader['test'])

    return model, test_result

if __name__ == "__main__":
    conf = get_args()
    print("-------------- CONF ---------------")
    print(conf)
    print("-----------------------------------")

    multitask_data = LoadMultitaskData(conf)
    multitask_train = MergeMultitaskData(multitask_data.train)
    multitask_val = MergeMultitaskData(multitask_data.val)
    multitask_test = MergeMultitaskData(multitask_data.test)

    loader = {
        'train' : data.DataLoader(multitask_train, batch_size=conf.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4) if multitask_train != None else None,
        'val'   : data.DataLoader(multitask_val, batch_size=conf.batch_size, shuffle=False, drop_last=False, num_workers=4) if multitask_val != None else None,
        'test'  : data.DataLoader(multitask_test, batch_size=conf.batch_size, shuffle=False, drop_last=False, num_workers=4) if multitask_test != None else None
    }

    model, results = train_multitask(loader, conf)
    print("Results:", results)
