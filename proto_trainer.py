import torch
import copy
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import proto_utils
from proto_classifier import BertClassifier

class ProtoTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config # needs to be adapted for random-way or adaptive lr
        self.save_hyperparameters()
        self.model = BertClassifier(config)
        self.episode_model = BertClassifier(config)
        self.loss_module = nn.CrossEntropyLoss()
        self.automatic_optimization = False

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        #inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.config.inner_lr)
        outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
        return outer_opt    


    def training_step(self, batch, batch_idx):
        s_labels = torch.flatten(torch.tensor([[i] * self.config.shot for i in range(self.config.way)])).to(self.config.device) # labels are [0 x shot, ..., way x shot]
        q_labels = torch.flatten(torch.tensor([[i] * self.config.shot for i in range(self.config.query_size)])).to(self.config.device)
        first_episode = True

        for (support, query) in batch: # support/query: [class_sample_0 * shot, ..., class_sample_way * shot] = N x K
        
            # create task specific model
            self.tune_task(support, s_labels)
            
            #STEP 7
            init_out = self.model(query)
            init_q_loss = self.loss_module(init_out, q_labels)     
            init_grad = torch.autograd.grad(init_q_loss, filter(lambda p: p.requires_grad, self.model.bert.parameters()))

            epi_out = self.episode_model(query) 
            epi_q_loss = self.loss_module(epi_out, q_labels) 
            epi_grad = torch.autograd.grad(epi_q_loss, filter(lambda p: p.requires_grad, self.episode_model.bert.parameters()))
            
            #assert self.model.train_param == len(epi_grad) and self.model.train_param == len(init_grad)
       
            #STEP 8
            if first_episode:
                gradient_holder = [init_grad[i] + epi_grad[i] for i in range(self.model.train_param)]
                first_episode = False
            else:
                for i in range(self.model.train_param):
                    gradient_holder[i] += init_grad[i] + epi_grad[i]

            # logging
            q_acc = (epi_out.argmax(dim=-1) == q_labels).float().mean()            
            self.log('train_query_acc', q_acc)
            self.log('train_query_loss', epi_q_loss)
            print(q_acc)

        # outer step
        for i, param in enumerate(filter(lambda p: p.requires_grad, self.model.bert.parameters())):
            if param.requires_grad == True:
                param.data -= self.config.outer_lr * gradient_holder[i]#.to(self.config.device)
        self.model.zero_grad()
 



    def validation_step(self, batch, batch_idx):
       
        outer_opt = self.optimizers()
        s_labels = torch.flatten(torch.tensor([[i] * self.config.shot for i in range(self.config.way)])).to(self.config.device)
        q_labels = torch.flatten(torch.tensor([[i] * self.config.shot for i in range(self.config.query_size)])).to(self.config.device)

        for (support, query) in batch: # support/query: [class_sample_0 * shot, ..., class_sample_way * shot] = N x K
        
            # create task specific model
            self.tune_task(task, support, s_labels)
            out = self.episode_model(query) #STEP 7
            q_loss = self.loss_module(out, q_labels)

            # logging
            q_acc = (out.argmax(dim=-1) == q_labels).float().mean()            
            self.log('val_query_acc', q_acc)
            self.log('val_query_loss', q_loss)


    def test_step(self, batch, batch_idx):
        inner_opt, outer_opt = self.optimizers()
        s_labels = torch.flatten(torch.tensor([[i] * self.config.shot for i in range(self.config.way)])).to(self.config.device)
        q_labels = torch.flatten(torch.tensor([[i] * self.config.shot for i in range(self.config.query_size)])).to(self.config.device)

        for (support, query) in batch: # support/query: [class_sample_0 * shot, ..., class_sample_way * shot] = N x K
        
            # create task specific model
            self.tune_task(task, support, s_labels)

            out = self.episode_model(query) #STEP 7
            q_loss = self.loss_module(out, q_labels)

            # logging
            q_acc = (out.argmax(dim=-1) == q_labels).float().mean()            
            self.log('test_query_acc', q_acc)
            self.log('test_query_loss', q_loss)
    

    

    def tune_task(self, support, s_labels):
        torch.set_grad_enabled(True) #for gradients in eval/test

        self.episode_model = copy.deepcopy(self.model) #STEP 2
        self.episode_model.zero_grad() # episode model should not inherit base gradients
        inner_opt = torch.optim.SGD(self.episode_model.parameters(), lr=self.config.inner_lr)

        # init prototype parameters
        proto_W, proto_b = proto_utils.prototype(self.config, self.model, support) #STEP 3
        self.episode_model.mlp.weight.data = proto_W.detach() #STEP 4
        self.episode_model.mlp.bias.data = proto_b.detach() 

        # inner loop
        for i in range(self.config.inner_steps): #STEP 5
            inner_opt.zero_grad()
            out = self.episode_model(support)
            s_loss = self.loss_module(out, s_labels)
            self.manual_backward(s_loss)
            inner_opt.step()
        
        self.episode_model.zero_grad()
        # add prototypes back to the computation graph
        self.episode_model.mlp.weight.data = 2 * proto_W + (self.episode_model.mlp.weight.data - (2 * proto_W)).detach() #STEP 6
        self.episode_model.mlp.bias.data = 2 * proto_b + (self.episode_model.mlp.bias.data - (2 * proto_b)).detach()



    def prototype(config, model, support):
        all_emb = model.embed(support) #only passes through bert
        c_k = torch.stack([torch.mean(all_emb[i:i + config.shot, :], dim=0) for i in range(0, config.shot * config.way, config.shot)]) # mean embedding for each class

        W = 2 * c_k
        b = torch.sum(-c_k * c_k, dim=-1) #same as [torch.dot(c_k[i,:] c_k[i,:])]
        #b = -torch.linalg.norm(c_k, dim=1)

        return W, b
