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

- `lstm-prep` transforms a street.parquet data set to a lstm ready version
- `lstm-train` trains an lstm model and records that experiment with dvc
- `lstm-predicts` uses the best model to predict an year ahead.

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

In `dc2_g22/notebooks` you can find `*.ipynb` files that we used to process data and try out models that didn't rely on pytorch.

