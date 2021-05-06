from argparse import ArgumentParser

arg_defaults = {
    "path" : "models",
    "optimizer" : "Adam",
    "lr" : 0.1,
    "max_epochs" : 100,
    "finetuned_layers" : 0,
    "tokenizer" : "BERT",
    "batch_size" : 64,
    "device" : "gpu",
    "seed" : 20
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


    args = parser.parse_args()
    return args
