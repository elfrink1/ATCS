import torch.utils.data as data
import os
import torch
import proto_utils
import numpy as np

from transformers import BertTokenizer
from proto_data import DataLoader
from proto_trainer import ProtoTrainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import product


class config():
    def __init__(self):
        self.data_path = './Data/'
        self.cache_path = './Cache/'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # bert h_parameters
        self.finetuned_layers = 4
        self.mlp_in = 768
        self.hidden = 192
        self.max_tokens = 256

         # maml h_parameters
        self.inner_steps = 5
        self.meta_batch = 8
        self.inner_lr = 0.001 
        self.outer_lr = 0.01
        self.max_epochs = 51

        # episode h_parameters
        self.min_way = 3
        self.max_way = 6
        self.shot = 5
        self.query_size = 10
        
        # eval parameters
        self.eval_perm = 3 # evaluate model with 3 different support sets
        self.eval_way = 4 
        self.class_eval = 32 # amount of query samples per class in eval

        self.query_batch = self.query_size # size of query batches


       

def train_protomaml(config, train_data, test_data, writer):
    
    model = ProtoTrainer(config).to(config.device)

    for epoch in tqdm(range(config.max_epochs), desc='Epochs'):
        
        batch = proto_utils.get_train_batch(config, train_data)
        t_acc, t_loss = model.meta_train(batch)
        writer.add_scalar("Train Loss", t_loss, epoch)
        writer.add_scalar("Train Accuracy", t_acc, epoch)

        if epoch%5 == 0:
            batch = proto_utils.get_test_batch(config, test_data)
            e_acc, e_loss = model.eval_model(batch)
            writer.add_scalar("Eval Loss", e_loss, epoch)
            writer.add_scalar("Eval Accuracy", e_acc, epoch)


if __name__ == "__main__":
    config = config()

    train_data = [DataLoader(config, set) for set in ['hp', 'ag', 'dbpedia']]
    test_data = DataLoader(config, 'bbc')
    
    tune_params = dict(
        inner_lr = [0.001, 0.0001],
        outer_lr = [0.01],
        inner_steps = [5, 10],
        meta_batch = [8, 16])

    param_values = [v for v in tune_params.values()]
    for inner_lr, outer_lr, inner_steps, meta_batch in product(*param_values):
        config.inner_lr = inner_lr
        config.outer_lr = outer_lr
        config.inner_steps = inner_steps
        config.meta_batch = meta_batch
        
        save_name = f'./runs/proto/in={inner_lr} out={outer_lr} steps={inner_steps} meta={meta_batch}'  
        if not os.path.isfile(save_name):
            writer = SummaryWriter(save_name)
            train_protomaml(config, train_data, test_data, writer)


    

