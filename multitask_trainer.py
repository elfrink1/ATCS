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

    def forward(self, batch, task):
        return self.model(batch, task)

    def configure_optimizers(self):
        """ Supported optimizers are Adam and Stochastic Gradient Descent. Scheduler is hard coded for preliminary results.
            it is not expected to reach epoch `100` during training. """
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx, task):
        out_ = self.model(batch["txt"], task)
        loss = self.loss_module(out_, batch["label"])
        acc = (out_.argmax(dim=-1) == batch["label"]).float().mean()

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, task):
        """ Simply calculate the accuracy and log it. """
        out_ = self.model(batch["txt"], task)
        acc = (out_.argmax(dim=-1) == batch["label"]).float().mean()
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx, task):
        """ Simply calculate the accuracy and log it. """
        out_ = self.model(batch["txt"], task)
        acc = (out_.argmax(dim=-1) == batch["label"]).float().mean()
        self.log('test_acc', acc)