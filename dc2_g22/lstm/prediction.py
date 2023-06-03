import argparse
from pathlib import Path

from dc2_g22.lstm.model import LSTMForcaster
from dc2_g22.lstm.dataset_loaders import WardCrimeDataset
from dc2_g22.lstm.feature_extraction import DataPrepper

import dvc.api
import pandas as pd
import numpy as np
import torch


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


def predict(args, model, dataset):
    print(dataset.data.index)
    row_add = lambda df, data: \
        pd.concat([df,
                   pd.DataFrame(data,
                                columns=df.columns,
                                index=[df.index[-1] + pd.DateOffset(months=1)])])
    if not args["future"]:
        dataset.data.drop(dataset.data.index[-args["chain_len"]:], inplace=True)

    dataset.data = row_add(dataset.data, np.nan)
    dataset.split_targets()

    for link in range(args["chain_len"]):
        points, _ = dataset[-1]
        with torch.no_grad():
            raw_prediction = model(points).to("cpu").numpy()
        prediction = raw_prediction.reshape(1, raw_prediction.shape[0])
        dataset.data.drop(dataset.data.index[-1], inplace=True)
        dataset.data = row_add(dataset.data, prediction)
        dataset.data = row_add(dataset.data, np.nan)
        dataset.split_targets()

    dataset.data.drop(dataset.data.index[-1], inplace=True)
    return dataset.data.iloc[-args["chain_len"]:]

def main(args):
    n_wards=23
    dp = DataPrepper(args["raw_path"], args["scaling"])

    last_exp = dvc.api.exp_show()[0]
    hidden_size = int(last_exp["hidden_size"])
    lstm_layers = int(last_exp["num_layers"])
    lags = last_exp["lags"]

    dataset = WardCrimeDataset(args["dataset_path"], lags=lags)

    model = LSTMForcaster(
        n_wards,
        hidden_size,
        n_wards,
        lstm_layers,
        find_device()
    )
    model.load_state_dict(torch.load(args["model_path"])["state_dict"])
    model.eval()

    raw_prediction = predict(args, model, dataset)
    scaled = pd.DataFrame(
        dp.scaler.inverse_transform(raw_prediction),
        columns=raw_prediction.columns,
        index=raw_prediction.index)
    print(scaled)

    output_path = Path(args["output_path"])
    scaled.to_parquet(output_path / Path(
        "predictions_%dmonths_from_%s" % (
            args["chain_len"], dataset.data.index[-args["chain_len"]])))


def start():
    parser = argparse.ArgumentParser(
        description="LSTM Data Preperation and Feature extraction")
    parser.add_argument("--raw_path", type=str,
                        help="Path to the linearmodel dataset")
    parser.add_argument("--dataset_path", "-d", type=str,
                        help="Path to the lstm dataset")
    parser.add_argument("--model_path", type=str, help="Output of predictions per month")
    parser.add_argument("--output_path", type=str, help="Output of predictions per month")
    parser.add_argument("--chain_len", type=int, help="Months to predict at the end of the data")
    parser.add_argument("--scaling", type=str,
                        choices = ["robust", "standard", "minmax"],
                        help="Kind  of scaling")
    parser.add_argument("--future", action="store_true",
                        help="Start prediction in the future")

    args = parser.parse_args()
    main(vars(args))


if __name__ == "__main__":
    start()
