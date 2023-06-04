import torch
import torch.optim as optim
import argparse

from dc2_g22.lstm.model import LSTMForcaster
from dc2_g22.lstm.train_test import train_test_loop
from dc2_g22.lstm.dataset_loaders import BatchSampler, WardCrimeDataset

from typing import Dict, Any

import dvclive
from dvclive import Live

def find_device():
    DEBUG = False
    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
    return device
 



def main(args):
    n_wards = 23

    # Initialize dvc logging

    device = find_device()
    model = LSTMForcaster(n_wards,
                          args.hidden_size,
                          n_wards,
                          args.num_layers,
                          device)
    model.to(device)

    dataset = WardCrimeDataset(args.dataset_path, args.lags, device)
    sampler = BatchSampler(args.batch_size, args.lags, dataset, args.train_ratio)
    criterion = torch.nn.MSELoss()

    # Define the optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Invalid optimizer choice. Available options: 'adam', 'sgd', 'rmsprop'")


    with Live(save_dvc_exp=True) as live:
        live.log_params(vars(args))
        train_test_loop(model,
                        args.num_epochs,
                        sampler,
                        optimizer,
                        criterion,
                        live,
                        device)

def start():
    """
    starts the little framework we have here.
    """
    parser = argparse.ArgumentParser(description="LSTM Training")
    parser.add_argument("--dataset_path", "-d", type=str, default="../../data/street.parquet",
                        help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train set ratio")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--lags', type=int, default=12, help='Number of lags (default: 12)')
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"],
                        help="Optimizer choice: adam, sgd, or rmsprop")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")

    args = parser.parse_args()
    main(args)



if __name__ == "__main__":
    start()
