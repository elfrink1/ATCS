import torch
import os
from torch.utils import data
import torch.optim as optim

from multitask_data import LoadMultitaskData, MergeMultitaskData
from multitask_model import MultitaskBert
from multitask_trainer import MultitaskTrainer
import proto_data
import proto_utils

from config import get_args
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_multitask(conf, train_loader, test_data, writer):
    model = MultitaskTrainer(conf).to(conf.device)

    if conf.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=conf.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    for epoch in tqdm(range(conf.max_epochs), desc="Epoch"):
        train_acc, train_loss = model.train_model(train_data, optimizer)
        writer.add_scalar("Train loss", train_loss, epoch)
        writer.add_scalar("Train accuracy", train_acc, epoch)
        if epoch % 5 == 0:
            batch = proto_utils.get_test_batch(conf, test_data)
            val_acc, val_loss = model.eval_model(batch)
            writer.add_scalar("Val loss", val_loss, epoch)
            writer.add_scalar("Val accuracy", val_acc, epoch)
        scheduler.step()

    print("Testing multitask model...")
    batch = proto_utils.get_test_batch(conf, test_data)
    test_acc, test_loss = model.eval_model(batch)
    writer.add_scalar("Test loss", test_loss, epoch)
    writer.add_scalar("Test accuracy", test_acc, epoch)


if __name__ == "__main__":
    conf = get_args()
    print("-------------- CONF ---------------")
    print(conf)
    print("-----------------------------------")

    print("Loading data...")
    multitask_data = LoadMultitaskData(conf)
    multitask_train = MergeMultitaskData(multitask_data.train)
    train_data = data.DataLoader(multitask_train, batch_size=conf.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=conf.num_workers) if multitask_train != None else None
    test_data = proto_data.DataLoader(conf, conf.test_set)

    print("Training multitask model...")
    save_name = f'./{conf.path}/runs/{conf.name}'
    writer = SummaryWriter(save_name)
    train_multitask(conf, train_data, test_data, writer)
