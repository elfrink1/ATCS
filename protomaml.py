import torch.utils.data as data
import pytorch_lightning as pl
import os


from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer
from proto_data import Dataset
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

        # bert h_parameters
        self.finetuned_layers = 1
        self.mlp_in = 768

         # maml h_parameters
        self.inner_steps = 1
        self.meta_batch = 16
        self.inner_lr = 0.1 #TODO: change
        self.outer_lr = 0.1
        self.max_epochs = 10

        # episode h_parameters
        self.way = 5
        self.shot = 5
        self.query_size = 10



       
def train_protomaml(config, loader):
    trainer = pl.Trainer(default_root_dir=os.path.join('./Model'),
                         checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_query_acc"),
                         gpus=1,
                         max_epochs=config.max_epochs,                                            
                         progress_bar_refresh_rate=1) 
    
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pl.seed_everything(config.seed)
    model = ProtoTrainer(config)
    

    trainer.fit(model, loader['train'], loader['val'])


    
    test_result = trainer.test(model, loader['test'])

    return model, test_result




if __name__ == "__main__":
    config = config()

    

    dataset = Dataset(config, 'hp')


    for sample in iter(dataset.train):
        print(sample)

    loader = {
        'train' : data.DataLoader(dataset.train, batch_size=config.meta_batch, sampler=1),
        'val'   : data.DataLoader(dataset.val, batch_size=config.meta_batch),
        'test'  : data.DataLoader(dataset.test, batch_size=config.meta_batch)
    }
    
    model, results = train_protomaml(config, loader)


        




