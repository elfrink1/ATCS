import torch
import itertools
import copy
import torch.nn as nn
import torch.optim as optim
import proto_utils
from proto_classifier import BertClassifier


class ProtoTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config # needs to be adapted for random-way or adaptive lr
        self.model = BertClassifier(config)
        self.loss_module = nn.CrossEntropyLoss()


    def meta_train(self, batch):

        q_labels = proto_utils.create_labels(self.config, self.config.way, self.config.query_size)     
        grad_sum = None
        
        for (support, query) in batch: # support/query: [class_samples_0 * shot, ..., class_samples_way * shot] = N * K
            # create task specific model
            episode_model, weight, bias = self.proto_task(support)
             
            #STEP 7    
            text = proto_utils.tokenize_set(self.config, query)                   
            out = episode_model(text) 
            out = self.output_layer(out, weight, bias)
            loss = self.loss_module(out, q_labels) 
            
            meta_grad = torch.autograd.grad(loss, [p for p in episode_model.parameters() if p.requires_grad], retain_graph=True)
            proto_grad = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad])
            grad = [mg + pg for (mg, pg) in zip(meta_grad, proto_grad)]

            #STEP 8
            if grad_sum is None:
                grad_sum = [g for g in grad]
            else:
                grad_sum = [gs + g for (gs, g) in zip(grad_sum, grad)]

            # logging
            q_acc = (out.argmax(dim=-1) == q_labels).float().mean()            
        
        for param, grad in zip([p for p in self.model.parameters() if p.requires_grad], grad):
            with torch.no_grad():
                param.data -= self.config.outer_lr * (grad / self.config.meta_batch)


    def eval_model(self, batch):
        
        accs = []
        for (support, query, eval_size) in batch:   
            episode_model, weight, bias = self.proto_task(support)

            with torch.no_grad():
                q_labels = proto_utils.create_labels(self.config, self.config.way, eval_size)
                text = proto_utils.tokenize_set(self.config, query)                    
                
                out = episode_model(text) 
                out = self.output_layer(out, weight, bias)
                loss = self.loss_module(out, q_labels) 
                accs.extend((out.argmax(dim=-1) == q_labels).float())
        
        mean_acc = torch.mean(torch.tensor(accs, requires_grad=False))
        print(mean_acc)   
        return mean_acc
      

    def proto_task(self, support):
        episode_model = copy.deepcopy(self.model) #STEP 2
        episode_model.zero_grad()
        inner_opt = torch.optim.SGD([p for p in episode_model.parameters() if p.requires_grad], lr=self.config.inner_lr)

        text = proto_utils.tokenize_set(self.config, support)
        s_labels = proto_utils.create_labels(self.config, self.config.way, self.config.shot)
        
        # init prototype parameters
        proto_W, proto_b = self.prototype(text)
        weight = proto_W.clone().detach().requires_grad_(True)
        bias = proto_b.clone().detach().requires_grad_(True)
       
        # inner loop
        for i in range(self.config.inner_steps): #STEP 5        
            inner_opt.zero_grad()
            out = episode_model(text)
            out = self.output_layer(out, weight, bias)
            loss = self.loss_module(out, s_labels)
            
            [weight_grad, bias_grad] = torch.autograd.grad(loss, [weight, bias], retain_graph=True)
            with torch.no_grad():
                weight.data -= self.config.inner_lr * weight_grad
                bias.data -= self.config.inner_lr * bias_grad
            loss.backward()
            inner_opt.step()
          
        # add prototypes back to the computation graph
        weight = 2 * proto_W + (weight - 2 * proto_W).detach() #STEP 6
        bias = 2 * proto_b + (bias - 2 * proto_b).detach() 

        return episode_model, weight, bias


    def output_layer(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)

    def prototype(self, support):
        all_emb = self.model(support)
        c_k = torch.stack([torch.mean(all_emb[i:i + self.config.shot, :], dim=0) for i in range(0, self.config.shot * self.config.way, self.config.shot)]) # mean embedding for each class

        W = 2 * c_k
        b = -torch.linalg.norm(c_k, dim=1)**2
        
        return W, b



