import torch
import itertools
import copy
import torch.nn as nn
import torch.optim as optim
import proto_utils
from proto_classifier import BertClassifier
from tqdm import tqdm


class ProtoTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BertClassifier(config)
        self.loss_module = nn.CrossEntropyLoss()
    

    def meta_train(self, batch):
        grad_sum = None
        accs, losses = [], []

        for episode in batch:
            episode_model, weight, bias = self.proto_task(episode.support) # create task specific model

            for (text, labels) in episode.query.get_batch(self.config, batch_size=self.config.query_batch):
                
                out = episode_model(text) 
                out = self.output_layer(out, weight, bias)
                loss = self.loss_module(out, labels) 
                
                proto_grad = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], retain_graph=True)
                meta_grad = torch.autograd.grad(loss, [p for p in episode_model.parameters() if p.requires_grad])
                
                grad = [mg + pg for (mg, pg) in zip(meta_grad, proto_grad)]

                if grad_sum is None:
                    grad_sum = [g.detach() for g in grad]
                else:
                    grad_sum = [gs + g.detach() for (gs, g) in zip(grad_sum, grad)]

                # logging
                accs.extend((out.argmax(dim=-1) == labels).float().detach().cpu())
                losses.append(loss.item())
        
        # update meta learner
        for param, grad in zip([p for p in self.model.parameters() if p.requires_grad], grad_sum):
            with torch.no_grad():
                param.data -= self.config.outer_lr * (grad / len(losses))


        train_acc = torch.mean(torch.tensor(accs)).cpu()
        train_loss = torch.mean(torch.tensor(losses)).cpu()
        return train_acc, train_loss


    def eval_model(self, batch):
        torch.cuda.empty_cache()
        accs, losses = [], []

        for episode in batch:
            episode_model, weight, bias = self.proto_task(episode.support)
            with torch.no_grad():         
                for (text, labels) in episode.query.get_batch(self.config, batch_size=self.config.query_batch):
                    out = episode_model(text) 
                    out = self.output_layer(out, weight, bias)
                    loss = self.loss_module(out, labels) 
                    losses.append(loss.item())
                    accs.extend((out.argmax(dim=-1) == labels).float())
                            
        eval_acc = torch.mean(torch.tensor(accs))
        eval_loss = torch.mean(torch.tensor(losses))
        return eval_acc, eval_loss









    def proto_task(self, support):
        episode_model = copy.deepcopy(self.model) #STEP 2
        episode_model.zero_grad()
        inner_opt = torch.optim.SGD([p for p in episode_model.parameters() if p.requires_grad], lr=self.config.inner_lr)

        # init prototype parameters
        proto_W, proto_b = self.prototype(support)
        weight = proto_W.clone().detach().requires_grad_(True)
        bias = proto_b.clone().detach().requires_grad_(True)

        # inner loop
        for i in range(self.config.inner_steps): #STEP 5        
            #support.shuffle() #reshuffles support set for each inner update  
            inner_opt.zero_grad()
            text, labels = next(support.get_batch(self.config))
                      
            out = episode_model(text)
            out = self.output_layer(out, weight, bias)
            loss = self.loss_module(out, labels)

            [wg, bg] = torch.autograd.grad(loss, [weight, bias], retain_graph=True)
            loss.backward()

            # update weights
            with torch.no_grad():
                weight.data -= self.config.inner_lr * wg
                bias.data -= self.config.inner_lr * bg   
            inner_opt.step()

        episode_model.zero_grad()
        # add prototypes back to the computation graph
        weight = 2 * proto_W + (weight - 2 * proto_W).detach() #STEP 6
        bias = 2 * proto_b + (bias - 2 * proto_b).detach() 
    
        return episode_model, weight, bias



    def prototype(self, support):
        sup_batch = support.len
        n_classes = support.n_classes

        text, _ = next(support.get_batch(self.config))
        emb = self.model(text)
        c_k = torch.stack([torch.mean(emb[i:i + self.config.shot, :], dim=0) for i in range(0, self.config.shot * n_classes, self.config.shot)])
        
        W = 2 * c_k
        b = -torch.linalg.norm(c_k, dim=1)**2
        support.shuffle()
        return W, b


    def output_layer(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)


