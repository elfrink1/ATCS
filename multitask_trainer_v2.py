import torch.nn as nn
import torch.optim as optim

from multitask_model import MultitaskBert

class MultitaskTrainer(nn.Module):
    def __init__(self, conf):
        """ Initialize the Multitask model and the the loss module. """
        super().__init__()
        self.config = conf
        self.model = MultitaskBert(conf).to(self.config.device)
        self.loss_module = nn.CrossEntropyLoss()

    def train_model(self, data):
        self.model.train()
        if self.config.optimizer == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

        for epoch in self.config.max_epochs:
            for batch in data:
                out_ = self.model(batch)
                loss_hp = self.loss_module(out_[0], batch['hp']["label"])
                loss_ag = self.loss_module(out_[1], batch['ag']["label"])
                # loss_bbc = self.loss_module(out_[2], batch['bbc']["label"])
                loss_ng = self.loss_module(out_[2], batch['ng']["label"])
                avg_loss = (loss_hp + loss_ag + loss_bbc + loss_ng) / 3

                acc_hp = (out_[0].argmax(dim=-1) == batch['hp']["label"]).float().mean()
                acc_ag = (out_[1].argmax(dim=-1) == batch['ag']["label"]).float().mean()
                # acc_bbc = (out_[2].argmax(dim=-1) == batch['bbc']["label"]).float().mean()
                acc_ng = (out_[2].argmax(dim=-1) == batch['ng']["label"]).float().mean()
                avg_train_acc = (acc_hp + acc_ag + acc_bbc + acc_ng) / 3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def eval_model(self, data):
        self.model.eval()

        with torch.no_grad():
            for batch in data:
                # TODO: define few shot evaluation step here
                pass
