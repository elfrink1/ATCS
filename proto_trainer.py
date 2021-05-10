import torch
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
        inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.config.inner_lr)
        outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
        return inner_opt, outer_opt    


    def training_step(self, batch, batch_idx):

        inner_opt, outer_opt = self.optimizers()
        labels = torch.tensor([[i] * self.config.shot for i in range(self.config.way)]) # labels are [0, ..., way]?

        for (support, query) in batch: # support/query: [class_sample_0 * shot, ..., class_sample_way * shot] = N x K
        
            self.episode_model = copy.deepcopy(self.model) #STEP 2
            self.episode_model.zero_grad() # episode model should not inherit base gradients
            #inner_opt = torch.optim.Adam(episode_model.parameters(), lr=self.config.inner_lr)

            # init prototype parameters
            proto_W, proto_b = proto_utils.prototype(self.config, self.episode_model.embed, support) #STEP 3
            self.episode_model.mlp.weights.data = proto_W.detach() #STEP 4
            self.episode_model.mlp.bias.data = proto_b.detach() 

            # inner loop
            for i in range(self.config.inner_steps): #STEP 5
                out = self.episode_model(support)
                loss = self.loss_module(out, labels)
                inner_opt.zero_grad()
                self.manual_backward(loss)
                inner_opt.step()
            
            # add prototypes back to the computation graph
            self.episode_model.mlp.weights.data = 2 * proto_W + (self.episode_model.mlp.weights.data - (2 * proto_W)).detach() #STEP 6
            self.episode_model.mlp.bias.data = 2 * proto_b + (self.episode_model.mlp.bias.data - (2 * proto_b)).detach()

            # gradients from query
            out = self.episode_model(query) #STEP 7
            loss = self.loss_module(out, labels)

            theta = torch.autograd.grad(loss, self.model.parameters())
            theta_k = torch.autograd.grad(loss, self.episode_model.parameters())
            self.model.grad.data = self.model.grad.data + theta + theta_k #STEP 8

            # logging
            acc = (out.argmax(dim=-1) == labels).float().mean()            
            self.log('train_query_acc', acc)
            self.log('train_query_loss', loss)

        outer_opt.step() #OUTER STEP
        self.optimizers.zero_grad()

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # need gradients for inner loop learning
        inner_opt, _ = self.optimizers()
        labels = torch.tensor([[i] * self.config.shot for i in range(self.config.way)]) # labels are [0, ..., way]?

        for (support, query) in batch: # support/query: [class_sample_0 * shot, ..., class_sample_way * shot] = N x K
        
            self.episode_model = copy.deepcopy(self.model)
            self.episode_model.zero_grad() # episode model should not inherit base gradients
            #inner_opt = torch.optim.Adam(episode_model.parameters(), lr=self.config.inner_lr)

            # init prototype parameters
            proto_W, proto_b = proto_utils.prototype(self.config, self.episode_model.embed, support)
            self.episode_model.mlp.weights.data = proto_W.detach()
            self.episode_model.mlp.bias.data = proto_b.detach() 

            # inner loop
            for i in range(self.config.inner_steps):
                out = self.episode_model(support)
                loss = self.loss_module(out, labels)
                inner_opt.zero_grad()
                self.manual_backward(loss)
                inner_opt.step()
            
            # add prototypes back to the computation graph
            self.episode_model.mlp.weights.data = 2 * proto_W + (self.episode_model.mlp.weights.data - (2 * proto_W)).detach() #STEP 6
            self.episode_model.mlp.bias.data = 2 * proto_b + (self.episode_model.mlp.bias.data - (2 * proto_b)).detach()

            # gradients from query
            out = self.episode_model(query)
            loss = self.loss_module(out, labels)

            # logging
            acc = (out.argmax(dim=-1) == labels).float().mean()            
            self.log('val_query_acc', acc)
            self.log('val_query_loss', loss)


    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True) # need gradients for inner loop learning
        inner_opt, _ = self.optimizers()
        labels = torch.tensor([[i] * self.config.shot for i in range(self.config.way)]) # labels are [0, ..., way]?

        for (support, query) in batch: # support/query: [class_sample_0 * shot, ..., class_sample_way * shot] = N x K
        
            self.episode_model = copy.deepcopy(self.model) #STEP 2
            self.episode_model.zero_grad() # episode model should not inherit base gradients
            #inner_opt = torch.optim.Adam(episode_model.parameters(), lr=self.config.inner_lr)

            # init prototype parameters
            proto_W, proto_b = proto_utils.prototype(self.config, self.episode_model.embed, support) #STEP 3
            self.episode_model.mlp.weights.data = proto_W.detach() #STEP 4
            self.episode_model.mlp.bias.data = proto_b.detach() 

            # inner loop
            for i in range(self.config.inner_steps): #STEP 5
                out = self.episode_model(support)
                loss = self.loss_module(out, labels)
                inner_opt.zero_grad()
                self.manual_backward(loss)
                inner_opt.step()
            
            # add prototypes back to the computation graph
            self.episode_model.mlp.weights.data = 2 * proto_W + (self.episode_model.mlp.weights.data - (2 * proto_W)).detach() #STEP 6
            self.episode_model.mlp.bias.data = 2 * proto_b + (self.episode_model.mlp.bias.data - (2 * proto_b)).detach()

            # gradients from query
            out = self.episode_model(query) #STEP 7
            loss = self.loss_module(out, labels)


            # logging
            #acc = (out.argmax(dim=-1) == labels).float().mean()            
            #self.log('val_query_acc', acc)
            #self.log('val_query_loss', loss)

