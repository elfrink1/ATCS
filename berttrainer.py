import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from bertclassifier import BertClassifier

class BERTTraniner(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, conf):
        super().__init__()
        self.save_hyperparameters()

        self.model = BertClassifier(conf)

        self.loss_module = nn.CrossEntropyLoss()

        print("Trainer init done")

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        print("training")
        out_ = self.model(batch["txt"])
        loss = self.loss_module(out_, batch["label"])
        acc = (out_.argmax(dim=-1) == batch["label"]).float().mean()

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        out_ = self.model(batch["txt"])
        acc = (batch["label"] == out_).float().mean()
        self.log('test_acc', acc)


