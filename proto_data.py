from datasets import load_dataset
from torch.utils.data import Dataset
import random
import torch
import numpy as np
import os
import shutil






# Access splits with Dataset.train/val/test
# split[class_id] = [sentences]
class ProtoDataset():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.set_path = os.path.join(self.config.data_path, dataset)

        self.train = self.Split(config)
        self.val = self.Split(config)
        self.test = self.Split(config)

        if dataset == "hp":
            self.load_data(self.process_hp)
        if dataset == "bbc":
            self.load_data(self.process_bbc)
        if dataset == "ag":
            self.load_data(self.process_ag)
        if dataset == "yahoo":
            self.load_data(self.process_yahoo)
        if dataset == "dbpedia":
            self.load_data(self.process_dbpedia)

    class Split(Dataset):
        def __init__(self, config):
            self.config = config
            self.data = {} # dict of class_id: [sentences]

        def train_batch(self, batch_size):
            return [self.sample_episode() for i in range(batch_size)]

        def test_batch(self, batch_size):
            return [self.testing_episode() for i in range(batch_size)]



        def testing_episode(self):
            class_ids = random.sample(self.data.keys(), self.config.way) # sample classes, n = way
            query, support = [], []
            
            smallest_set = min([len(self.data[class_id]) for class_id in class_ids])
            eval_size = min(smallest_set, self.config.max_eval_size)

            for class_id in class_ids: # for all classes
                sample_ids = random.sample(range(len(self.data[class_id])), eval_size + self.config.shot)
                
                support.extend([self.data[class_id][sample_id] for sample_id in sample_ids[0:self.config.shot]]) 
                query.extend([self.data[class_id][sample_id] for sample_id in sample_ids[self.config.shot:]]) 
        
            return support, query, eval_size


        # returns two lists of [tokenized_sentences]
        def sample_episode(self):
            #config.way = random.randint(config.min_way, min(config.max_way, len(self.data))) # amount of ways, min_ways <= n <= min(max_ways, n_classes)
            class_ids = random.sample(self.data.keys(), self.config.way) # sample classes, n = way
            query, support = [], []
            
            for class_id in class_ids: # for all classes
                sample_ids = random.sample(range(len(self.data[class_id])), self.config.query_size + self.config.shot) # number of sampled ids = n_query + n_support
                
                query.extend([self.data[class_id][sample_id] for sample_id in sample_ids[:self.config.query_size]]) # class_query = first sampled ids, n = query_size
                support.extend([self.data[class_id][sample_id] for sample_id in sample_ids[self.config.query_size:]]) # class_support = last sampled ids, n = shot

            return support, query#, class_ids







    # Download, process, save, remove raw dataset
    def load_data(self, process_fn):
        if os.path.exists(self.set_path):
            self.load_splits()
        else:
            #os.mkdir(self.config.cache_path)
            process_fn()
            #shutil.rmtree(self.config.cache_path) # remove cache


    def save_splits(self):
        os.mkdir(self.set_path)
        torch.save(self.train.data, os.path.join(self.set_path, 'train.pt'))
        torch.save(self.val.data, os.path.join(self.set_path, 'val.pt'))
        torch.save(self.test.data, os.path.join(self.set_path, 'test.pt'))     

    def load_splits(self):
        self.train.data = torch.load(os.path.join(self.set_path, 'train.pt'))
        self.val.data = torch.load(os.path.join(self.set_path, 'val.pt'))
        self.test.data = torch.load(os.path.join(self.set_path, 'test.pt'))  


    """DATASET SPECIFIC"""
    def process_hp(self):
        #TODO: change to use all data?
        dataset = load_dataset('Fraser/news-category-dataset', split='train', cache_dir=self.config.cache_path) #train, test, validation

        tasks = {i: [] for i in range(41)} 
        for example in dataset:
            if self.config.debug == True:
                text = example["headline"]
            else:
                text = example["headline"] + ' ' + example["short_description"]
            tasks[example['category_num']].append(text)
        
        #TODO: informed splits
        self.train.data = {i: tasks[i] for i in range(0,29)}
        self.val.data =  {i: tasks[i]  for i in range(29,35)}
        self.test.data =  {i: tasks[i] for i in range(35,41)}
        
        self.save_splits()   
    

    def process_ag(self):
        #TODO: change to use all data?
        dataset = load_dataset('ag_news', cache_dir=self.config.cache_path)
        print(dataset)
        
        tasks = {i: [] for i in range(41)} 
        for example in dataset:
            if self.config.debug == True:
                text = example["headline"]
            else:
                text = example["headline"] + ' ' + example["short_description"]
            tasks[example['category_num']].append(text)
        
        #TODO: informed splits
        self.train.data = {i: tasks[i] for i in range(0,2)}
        self.val.data =  {2: tasks[2]}
        self.test.data =  {3: tasks[3]}
        
        self.save_splits()   

