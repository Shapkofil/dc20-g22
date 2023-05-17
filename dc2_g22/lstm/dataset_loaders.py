import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Tuple
from pathlib import Path

import random

class WardCrimeDataset(Dataset):

    def __init__(self, X:Path)->None:
        self.data = WardCrimeDataset.slurp_data_from_cached(X)
        self.points = self.data[1:]
        self.targets = self.data[:-1]

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor, np.ndarray]:
        # In case index is out of bounds bring it back in
        idx = idx % len(self.points)
        return self.points[idx], self.targets[idx]

    @staticmethod
    def slurp_data_from_cached(X:Path)->np.ndarray:
        raise NotImplemented


class BatchSampler():
    def __init__(self,
                 batch_size: int,
                 dataset: WardCrimeDataset,
                 test_percent: float):
        self.batch_size = batch_size
        self.dataset = dataset

        datlen = len(dataset)
        train_batch_n = int(( test_percent * datlen ) // self.batch_size)
        self.index = [i for i in range(test_batch_n * self.batch_size)]
        self.test_index = [i for i in range(test_batch_n * self.batch_size, datlen)]


    def shuffle(self) -> None:
        random.shuffle(self.index)

    def __len__(self) -> int:
        return (len(self.index) // self.batch_size) + 1

    
