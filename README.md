# Data challenge 2 Group 22

## Documentation

Techincal documentation and structure of our modules could be found by opening
`docs/dc2_g22.html` in your preferred browser.

## Installation

You must have python 3.8.16 installed

``` shell
poetry install
poetry shell
```

It will take a few minutes to build the environment.

Once completed with no errors and the poetry shell is Installed proceed to usage.
All commands described in the usage section are available exclusively in the poetry shell.

## Usage

### LSTM

This repository implements 3 specific commands to try out our code.

`lstm-train`, `lstm-prep` and `lstm-predict`


Important to run training from lstm-predict

- `lstm-prep` transforms a street.parquet data set to a lstm ready version
- `lstm-train` trains an lstm model and records that experiment with dvc
- `lstm-predicts` uses the best model to predict an year ahead.

passing and --help flag to there commands shows intended usage.

#### Examples experiment 

``` shell
lstm-train --dataset_path data/lstm.parquet --num_epochs 150 --num_layers 2 --hidden_size 64 --batch_size 32 --lags 24 --learning_rate 0.0001 --optimizer adam
```

#### Example prediction

``` shell
lstm-predict --raw_path data/linearmodel.parquet -d data/lstm.parquet --model_path model_checkpoints/model_best.pth --chain_len 3 --output_path data/ --scaling robust
``` 

### Viewing experiments

To see experiments you have performed

``` shell
dvc show exp
```

To see plots of metrics for those experiments against epochs, please refer too.

``` shell
dvc plots diff $(dvc show exp --names-only)
```

### Exploring notebooks

In `dc2_g22/notebooks` you can find `*.ipynb` files that we used to process data and try out secondary models such as Prophet or Linear Regression, that didn't rely on pytorch.

