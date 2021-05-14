import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from multitask_model import MultitaskBert

class MultitaskTrainer(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, conf):
        """ Initialize the Multitask model and the the loss module. """
        super().__init__()
        self.save_hyperparameters()
        self.model = MultitaskBert(conf)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        """ Supported optimizers are Adam and Stochastic Gradient Descent. Scheduler is hard coded for preliminary results.
            it is not expected to reach epoch `100` during training. """
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

        return [optimizer], [scheduler]


# TODO Weighted sum of losses

    def training_step(self, batch, batch_idx):
        out_ = self.model(batch)
        loss_hp = self.loss_module(out_[0], batch['hp']["label"])
        loss_ag = self.loss_module(out_[1], batch['ag']["label"])
        loss_bbc = self.loss_module(out_[2], batch['bbc']["label"])
        loss_ng = self.loss_module(out_[3], batch['ng']["label"])
        total_loss = loss_hp + loss_ag + loss_bbc + loss_ng

        acc_hp = (out_[0].argmax(dim=-1) == batch['hp']["label"]).float().mean()
        acc_ag = (out_[1].argmax(dim=-1) == batch['ag']["label"]).float().mean()
        acc_bbc = (out_[2].argmax(dim=-1) == batch['bbc']["label"]).float().mean()
        acc_ng = (out_[3].argmax(dim=-1) == batch['ng']["label"]).float().mean()
        avg_train_acc = (acc_hp + acc_ag + acc_bbc + acc_ng) / 4

        self.log('train_acc_hp', acc_hp, on_step=False, on_epoch=True)
        self.log('train_acc_ag', acc_ag, on_step=False, on_epoch=True)
        self.log('train_acc_bbc', acc_bbc, on_step=False, on_epoch=True)
        self.log('train_acc_ng', acc_ng, on_step=False, on_epoch=True)
        self.log('train_acc', avg_train_acc, on_step=False, on_epoch=True)

        self.log('train_loss_hp', loss_hp)
        self.log('train_loss_ag', loss_ag)
        self.log('train_loss_bbc', loss_bbc)
        self.log('train_loss_ng', loss_ng)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """ Simply calculate the accuracy and log it. """
        out_ = self.model(batch)
        acc_hp = (out_[0].argmax(dim=-1) == batch['hp']["label"]).float().mean()
        acc_ag = (out_[1].argmax(dim=-1) == batch['ag']["label"]).float().mean()
        acc_bbc = (out_[2].argmax(dim=-1) == batch['bbc']["label"]).float().mean()
        acc_ng = (out_[3].argmax(dim=-1) == batch['ng']["label"]).float().mean()
        avg_val_acc = (acc_hp + acc_ag + acc_bbc + acc_ng) / 4

        self.log('val_acc_hp', acc_hp)
        self.log('val_acc_ag', acc_ag)
        self.log('val_acc_bbc', acc_bbc)
        self.log('val_acc_ng', acc_ng)
        self.log('val_acc', avg_val_acc)

    def test_step(self, batch, batch_idx):
        """ Simply calculate the accuracy and log it. """
        out_ = self.model(batch)
        acc_hp = (out_[0].argmax(dim=-1) == batch['hp']["label"]).float().mean()
        acc_ag = (out_[1].argmax(dim=-1) == batch['ag']["label"]).float().mean()
        acc_bbc = (out_[2].argmax(dim=-1) == batch['bbc']["label"]).float().mean()
        acc_ng = (out_[3].argmax(dim=-1) == batch['ng']["label"]).float().mean()
        avg_test_acc = (acc_hp + acc_ag + acc_bbc + acc_ng) / 4

        self.log('val_acc_hp', acc_hp)
        self.log('val_acc_ag', acc_ag)
        self.log('val_acc_bbc', acc_bbc)
        self.log('val_acc_ng', acc_ng)
        self.log('test_acc', avg_test_acc)
