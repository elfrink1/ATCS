from argparse import ArgumentParser

# class Config(object):
#     # Temporary class to store the config. Will later be replaced by a 
#     # argument based config
#     def __init__(self):
#         self.name = "bertplusmlp"
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         self.batch_size = 64
#         self.lr = 0.1
#         self.dataset = "bbc"
#         self.max_epochs = 100
#         self.finetune_layers = 1
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.seed = 20
#         self.optimizer = "SGD"
#         self.CHECKPOINT_PATH = "models"
#         self.nr_classes = 41


arg_defaults = {
    "path" : "models/bert",
    "optimizer" : "Adam",
    "lr" : 0.1,
    "max_epochs" : 100,
    "finetuned_layers" : 0,
    "tokenizer" : "BERT",
    "batch_size" : 64,
    "device" : "gpu",
    "seed" : 20,
    "max_text_length": -1
}

def get_args():
    parser = ArgumentParser(description="BERT baseline training")
    parser.add_argument("name", type=str, 
                        help="name of the model")
    parser.add_argument("dataset", type=str, choices=["hp", "ag", "bbc"],
                        help="the dataset used for training")
    parser.add_argument("nr_classes", type=int,
                        help="the number of classes of the dataset")
    parser.add_argument("--path", type=str, default=arg_defaults["path"],
                        help="the path to save the model checkpoints and logs")
    parser.add_argument("--optimizer", type=str, default=arg_defaults["optimizer"], 
                        choices=["Adam", "SGD"], help="the optimizer to use for training")
    parser.add_argument("--lr", type=float, default=arg_defaults["lr"],
                        help="the learning rate for the optimizer")
    parser.add_argument("--max_epochs", type=int, default=arg_defaults["max_epochs"],
                        help="the number of epochs after which to stop")
    parser.add_argument("--finetuned_layers", type=int, default=arg_defaults["finetuned_layers"],
                        help="the number of transformer layers of BERT to finetune")
    parser.add_argument("--tokenizer", type=str, default=arg_defaults["tokenizer"],
                        choices=["BERT"], help="the tokenizer to use on the text")
    parser.add_argument("--batch-size", type=int, default=arg_defaults["batch_size"],
                        help="size of the batches")
    parser.add_argument("--device", type=str, default=arg_defaults["device"],
                        choices=["cpu", "gpu"], help="the device to use")
    parser.add_argument("--seed", type=int, default=arg_defaults["seed"],
                        help="the random seed used by pytorch lightning")
    parser.add_argument("--progress_bar", action="store_true", default=False,
                        help="show the progress bar")
    parser.add_argument("--max_text_length", type=int, default=arg_defaults["max_text_length"],
                        help="the max text length in characters (-1: no limit)")
    
    
    args = parser.parse_args()
    return args