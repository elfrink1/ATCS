from datasets import load_dataset
import random
import torch
import numpy as np
import os
import shutil






# Access splits with Dataset.train/val/test
# split[class_id] = [sentences]
class Dataset():
    def __init__(self, config, dataset):
        self.config = config
        self.cache_path = config.cache_path
        self.set_path = os.path.join(config.data_path, dataset)
        self.tokenizer = config.tokenizer

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

    class Split():
        def __init__(self, config):
            self.config = config
            self.data = {} # dict of class_id: [sentences]

        def __iter__(self):
            return self
        
        def __next__(self):
            return self.sample_episode() # sample_dataset when more datasets are used

        def __len__(self):
            return 50

        # returns two lists of [tokenized_sentences]
        def sample_episode(self):
            #config.way = random.randint(config.min_way, min(config.max_way, len(self.data))) # amount of ways, min_ways <= n <= min(max_ways, n_classes)
            class_ids = random.sample(range(len(self.data)), self.config.way) # sample classes, n = way
            query, support = [], []
            
            for class_id in class_ids: # for all classes
                sample_ids = random.sample(range(len(self.data[class_id])), self.config.query_size + self.config.shot) # number of sampled ids = n_query + n_support
                
                query.extend([self.data[class_id][sample_id] for sample_id in sample_ids[:self.config.query_size]]) # class_query = first sampled ids, n = query_size
                support.extend([self.data[class_id][sample_id] for sample_id in sample_ids[self.config.query_size:]]) # class_support = last sampled ids, n = shot

            tok_query = self.config.tokenizer(query, padding=True, truncation=True, return_tensors='pt')
            tok_support = self.config.tokenizer(support, padding=True, truncation=True, return_tensors='pt')

            return tok_support, tok_query#, class_ids




    # Download, process, save, remove raw dataset
    def load_data(self, process_fn):
        if os.path.exists(self.set_path):
            self.load_splits()
        else:
            #os.mkdir(self.cache_path)
            process_fn()
            #shutil.rmtree(self.cache_path) # remove cache


    def save_splits(self):
        os.mkdir(self.set_path)
        torch.save(self.train.data, os.path.join(self.set_path, 'train.pt'))
        torch.save(self.train.data, os.path.join(self.set_path, 'val.pt'))
        torch.save(self.train.data, os.path.join(self.set_path, 'test.pt'))     

    def load_splits(self):
        self.train.data = torch.load(os.path.join(self.set_path, 'train.pt'))
        self.val.data = torch.load(os.path.join(self.set_path, 'val.pt'))
        self.test.data = torch.load(os.path.join(self.set_path, 'test.pt'))  


    """DATASET SPECIFIC"""
    def process_hp(self):
        #TODO: change to use all data?
        dataset = load_dataset('Fraser/news-category-dataset', split='train', cache_dir=self.cache_path)
        
        tasks = {i: [] for i in range(41)} 
        for example in dataset:
            text = example["headline"] + ' ' + example["short_description"]
            tasks[example['category_num']].append(text)
        
        #TODO: informed splits
        self.train.data = [tasks[i] for i in range(0,29)]
        self.val.data = [tasks[i]  for i in range(29,35)]
        self.test.data = [tasks[i] for i in range(35,41)]
        
        self.save_splits()   
    
