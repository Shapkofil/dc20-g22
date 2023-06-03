import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

from typing import Tuple, Generator, Union, List
from pathlib import Path

import random

class WardCrimeDataset(Dataset):

    def __init__(self,
                 X: Union[str, os.PathLike, Path],
                 lags: int = 12,
                 device: str = "cpu") -> None:
        X = Path(X)
        self.data = WardCrimeDataset.slurp_data_from_cached(X)
        self.points = self.data[:-1]
        self.targets = self.data[1:]
        self.lags = lags
        self.device = device

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: Union[int, np.integer, slice, np.ndarray]) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the X and y
        """

        def nplag(x, y):
            return np.stack([x - y + i - 1 for i in range(y + 1)], axis=-1)
        if isinstance(idx, int) or isinstance(idx, np.integer):
            X_idx = np.arange(idx - self.lags, idx + 1)
            # print(idx, X_idx)
        elif isinstance(idx, slice):
            X_idx = nplag(np.arange(idx.start, idx.stop, idx.step),
                          self.lags)
            # print(idx, X_idx)
        elif isinstance(idx, np.ndarray):
            X_idx = nplag(idx,
                          self.lags)
        else:
            raise RuntimeError("idx type not supported")

        return torch.Tensor(self.points[X_idx]).to(self.device), \
            torch.Tensor(self.targets[idx]).to(self.device)

    @staticmethod
    def slurp_data_from_cached(X: Path) -> np.ndarray:
        """
        Get the data aggregated by month(rows) and wards (columns)
        """
        df = pd.read_parquet(X)
        return df.to_numpy()


class BatchSampler():
    def __init__(self,
                 batch_size: int,
                 lags: int,
                 dataset: WardCrimeDataset,
                 train_percent: float,
                 ):
        self.lags = lags
        self.batch_size = batch_size
        self.train_percent = train_percent
        self.dataset = dataset

        datlen = len(dataset)
        self.batch_n = int(datlen // self.batch_size)

        # Generate Index
        self.index = np.arange(self.lags, len(self.dataset) - 1)
        self.shuffle()

    def shuffle(self) -> None:
        # np.random.shuffle(self.index)

        self.train_index, self.test_index = np.split(
            self.index,
            [int(self.batch_n * self.train_percent) * self.batch_size],
            axis=0)

    def __len__(self) -> int:
        return self.batch_n


class BatchLoader():
    def __init__(self,
                 sampler: BatchSampler,
                 test: bool = False):
        self.sampler = sampler
        if test:
            if not hasattr(self.sampler, "test_index"):
                raise RuntimeError(
                    "test index are not defined generator cannot be initialized")
            self.index = self.sampler.test_index
        else:
            if not hasattr(self.sampler, "train_index"):
                raise RuntimeError(
                    "train index are not defined generator cannot be initialized")
            self.index = self.sampler.train_index

    def __len__(self) -> int:
        return len(self.index) // self.sampler.batch_n

    def __iter__(self, test_gen=False) -> \
            Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for batch_index in range(0, len(self.index), self.sampler.batch_size):
            batch = self.sampler.dataset[
                batch_index: batch_index + self.sampler.batch_size + 1]
            yield batch
