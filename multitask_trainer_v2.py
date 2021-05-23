import torch.nn as nn
import torch.optim as optim

from multitask_model import MultitaskBert
from proto_data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import product

class MultitaskTrainer(nn.Module):
    def __init__(self, conf):
        """ Initialize the Multitask model and the the loss module. """
        super().__init__()
        self.config = conf
        self.model = MultitaskBert(conf)
        self.loss_module = nn.CrossEntropyLoss()


    def train_model(self, train_data):
        self.model.train()
        if self.config.optimizer == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)


        for epoch in self.config.max_epochs:
            for batch in train_data:
                out_ = self.model(batch)
                losses, accs = []
                for i, dataset in enumerate(self.config.train_sets):
                    loss = self.loss_module(out_[i], batch[dataset]["label"])
                    acc = (out_[i].argmax(dim=-1) == batch[dataset]["label"]).float().mean()
                    accs.append(acc.item())
                    losses.append(loss.item())
                train_loss = torch.mean(torch.Tensor(losses))
                train_acc = torch.mean(torch.Tensor(accs))

                # loss_hp = self.loss_module(out_[0], batch['hp']["label"])
                # loss_ag = self.loss_module(out_[1], batch['ag']["label"])
                # loss_bbc = self.loss_module(out_[2], batch['bbc']["label"])
                # loss_ng = self.loss_module(out_[2], batch['ng']["label"])
                # avg_loss = (loss_hp + loss_ag + loss_bbc + loss_ng) / 3
                #
                # acc_hp = (out_[0].argmax(dim=-1) == batch['hp']["label"]).float().mean()
                # acc_ag = (out_[1].argmax(dim=-1) == batch['ag']["label"]).float().mean()
                # acc_bbc = (out_[2].argmax(dim=-1) == batch['bbc']["label"]).float().mean()
                # acc_ng = (out_[2].argmax(dim=-1) == batch['ng']["label"]).float().mean()
                # avg_train_acc = (acc_hp + acc_ag + acc_bbc + acc_ng) / 3

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            scheduler.step()
        return train_loss, train_acc


    def eval_model(self, batch):
        self.model.eval()
        torch.cuda.empty_cache()
        accs, losses = [], []

        for episode in batch:
            episode_model, weight, bias = self.proto_task(episode.support)
            with torch.no_grad():
                for (text, labels) in episode.query.get_batch(self.config, batch_size=self.config.query_batch):
                    out = episode_model.bert(text['input_ids'], text['attention_mask']).pooler_output
                    out = episode_model.few_shot_head(out)
                    out = self.output_layer(out, weight, bias)
                    loss = self.loss_module(out, labels)
                    losses.append(loss.item())
                    accs.extend((out.argmax(dim=-1) == labels).float())

        eval_acc = torch.mean(torch.tensor(accs))
        eval_loss = torch.mean(torch.tensor(losses))
        return eval_acc, eval_loss


    def proto_task(self, support):
        episode_model = copy.deepcopy(model) #STEP 2
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

            out = episode_model.bert(text['input_ids'], text['attention_mask']).pooler_output
            out = episode_model.few_shot_head(out)
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
        emb = self.model.bert(batch['input_ids'], batch['attention_mask']).pooler_output
        emb = self.model.few_shot_head(emb)
        c_k = torch.stack([torch.mean(emb[i:i + self.config.shot, :], dim=0) for i in range(0, self.config.shot * n_classes, self.config.shot)])

        W = 2 * c_k
        b = -torch.linalg.norm(c_k, dim=1)**2
        support.shuffle()
        return W, b


    def output_layer(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)
